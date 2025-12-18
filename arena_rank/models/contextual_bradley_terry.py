"""Contextual Bradley-Terry rating system implementation in JAX."""

from functools import partial
import math
from copy import deepcopy
import multiprocessing as mp
from typing import Tuple, Dict, Any
import jax
import jax.nn as nn
import jax.numpy as jnp
from jax import jit, tree_util
from jaxtyping import PyTree

from arena_rank.models.rating_system import RatingSystem
from arena_rank.utils.data_utils import ContextualPairDataset
from arena_rank.utils.math_utils import assemble_parwise_matrix

jax.config.update("jax_enable_x64", True)


# needs to be at the top level to be pickle-able for multiprocessing
def fit_single_bootstrap_sample(
    key: jnp.ndarray,
    model: RatingSystem,
    dataset: ContextualPairDataset,
) -> PyTree:
    """Helper function to fit a single bootstrap sample."""
    n_obs = dataset.pairs.shape[0]
    idxs = jax.random.randint(key, shape=(n_obs,), minval=0, maxval=n_obs)

    boot_dataset = ContextualPairDataset(
        competitors=dataset.competitors,
        pairs=dataset.pairs[idxs],
        outcomes=dataset.outcomes[idxs],
        features=dataset.features[idxs],
        weights=dataset.weights[idxs],
    )

    boot_model = deepcopy(model)
    boot_model.params = {
        "ratings": jnp.zeros_like(boot_model.params["ratings"], dtype=model.dtype),
        "coeffs": jnp.zeros_like(boot_model.params["coeffs"], dtype=model.dtype),
    }
    boot_model.fit(boot_dataset)
    return boot_model.params


class ContextualBradleyTerry(RatingSystem):
    """
    Bradley-Terry rating system with contextual features.
    """

    def __init__(
        self,
        n_competitors: int,
        n_features: int,
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        reg: float = 1.0,
        hessian_reg: float = 1e-5,
        max_iter: int = 1000,
        ftol: float = 1e-9,
        gtol: float = 1e-9,
        dtype=jnp.float64,
        verbose: bool = False,
    ):
        self.n_competitors = n_competitors
        self.n_features = n_features
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.reg = reg
        self.hessian_reg = hessian_reg
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
        self.dtype = dtype
        self.verbose = verbose
        self.alpha = scale / math.log(base)
        self.params = {
            "ratings": jnp.zeros(n_competitors, dtype=dtype),
            "coeffs": jnp.zeros(n_features, dtype=dtype),
        }

    @partial(jit, static_argnames=["self"])
    def loss_function(self, params: PyTree, data: PyTree) -> float:
        """
        Computes the Contextual Bradley-Terry loss.
        """
        ratings = params["ratings"]
        coeffs = params["coeffs"]
        matchups = data["pairs"]
        features = data["features"]
        outcomes = data["outcomes"]
        weights = data["weights"]

        matchup_ratings = ratings[matchups]
        bt_logits = matchup_ratings[:, 0] - matchup_ratings[:, 1]
        context_logits = jnp.dot(features, coeffs)
        total_logits = bt_logits + context_logits
        log_likelihood = total_logits * outcomes - nn.softplus(total_logits)
        weighted_ll = jnp.mean(weights * log_likelihood)
        reg_loss = 0.5 * self.reg * jnp.sum(coeffs**2)
        return -(weighted_ll) + reg_loss

    @staticmethod
    @partial(jit, static_argnames=["n_obs", "n_competitors", "n_features"])
    def compute_hessian_and_covariance(
        ratings: jnp.ndarray,
        coeffs: jnp.ndarray,
        matchups: jnp.ndarray,
        features: jnp.ndarray,
        outcomes: jnp.ndarray,
        weights: jnp.ndarray,
        reg: float,
        hessian_reg: float,
        n_competitors: int,
        n_features: int,
        n_obs: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the Hessian and Gradient Covariance matrices for the
        sandwich estimator within the Contextual Bradley-Terry framework.
        """
        matchup_ratings = ratings[matchups]
        bt_logits = matchup_ratings[:, 0] - matchup_ratings[:, 1]
        context_logits = jnp.dot(features, coeffs)
        probs = jax.nn.sigmoid(bt_logits + context_logits)

        # row level second derivative values
        hess_vals = (probs * (1.0 - probs) * weights) / n_obs

        idx_a = matchups[:, 0]
        idx_b = matchups[:, 1]

        # rating block of the hessian
        hess_ratings = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
        hess_ratings = assemble_parwise_matrix(hess_ratings, hess_vals, idx_a, idx_b)
        hess_ratings = hess_ratings + jnp.eye(n_competitors) * hessian_reg

        # feature block of the hessian
        hess_features = jnp.dot(features.T, (hess_vals[:, None] * features))
        hess_features = hess_features + jnp.eye(n_features) * hessian_reg

        # "cross" block of the hessian, deriv w.r.t. ratings and features
        hess_weighted_feats = hess_vals[:, None] * features  # (n_obs, n_features)
        hess_cross = jnp.zeros((n_competitors, n_features), dtype=ratings.dtype)
        hess_cross = hess_cross.at[idx_a, :].add(hess_weighted_feats)
        hess_cross = hess_cross.at[idx_b, :].add(-hess_weighted_feats)

        hessian = jnp.block(
            [
                [hess_ratings, hess_cross],
                [hess_cross.T, hess_features],
            ]
        )

        # row level gradient covariance values
        grad_cov_vals = (((outcomes - probs) * weights) ** 2) / n_obs

        # rating block of the gradient covariance
        grad_cov_ratings = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
        grad_cov_ratings = assemble_parwise_matrix(grad_cov_ratings, grad_cov_vals, idx_a, idx_b)

        # feature block of the gradient covariance
        grad_cov_features = jnp.dot(features.T, (grad_cov_vals[:, None] * features))

        # cross block of the gradient covariance
        grad_cov_weighted_feats = grad_cov_vals[:, None] * features  # (n_obs, n_features)
        grad_cov_cross = jnp.zeros((n_competitors, n_features), dtype=ratings.dtype)
        grad_cov_cross = grad_cov_cross.at[idx_a, :].add(grad_cov_weighted_feats)
        grad_cov_cross = grad_cov_cross.at[idx_b, :].add(-grad_cov_weighted_feats)

        grad_cov = jnp.block(
            [
                [grad_cov_ratings, grad_cov_cross],
                [grad_cov_cross.T, grad_cov_features],
            ]
        )

        # correct for L2 regularization in gradient covariance
        params_vec = jnp.concatenate([jnp.zeros(n_competitors), coeffs])
        reg_correction = (reg**2 / n_obs) * jnp.outer(params_vec, params_vec)
        grad_cov = grad_cov - reg_correction
        return hessian, grad_cov

    def compute_ratings_and_cis(
        self,
        dataset: ContextualPairDataset,
        significance_level: float = 0.05,
        ci_method: str = "sandwich",
        num_bootstrap: int = 100,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """
        Calculates ratings, coefficients, and confidence intervals.

        Args:
            ci_method: "sandwich" for central limit theorem estimates or "bootstrap" for resampling.
            num_bootstrap: Number of bootstrap samples (only used if ci_method="bootstrap").
            seed: Random seed for bootstrapping.
            n_jobs: Number of workers for multiprocessing (only used if ci_method="bootstrap").
        """
        if not self.fitted:
            self.fit(dataset)

        ratings = self.params["ratings"]  # unscaled ratings from the fitted model
        coeffs = self.params["coeffs"]

        alpha = self.alpha
        offset = self.init_rating

        def scale(x):
            return x * alpha + offset

        scaled_ratings = scale(ratings)
        rating_lower = None
        rating_upper = None
        scaled_variances = None

        if ci_method == "sandwich":
            features = dataset.features
            n_obs = dataset.pairs.shape[0]

            hessian, gradient_cov = self.compute_hessian_and_covariance(
                ratings,
                coeffs,
                dataset.pairs,
                features,
                dataset.outcomes,
                dataset.weights,
                self.reg,
                self.hessian_reg,
                self.n_competitors,
                self.n_features,
                n_obs,
            )

            hessian_inv = jnp.linalg.inv(hessian)
            asymptotic_variance = hessian_inv @ gradient_cov @ hessian_inv
            param_variances = jnp.diag(asymptotic_variance) / n_obs
            rating_variances = param_variances[: self.n_competitors]

            std_errs = jnp.sqrt(rating_variances)
            z = jax.scipy.stats.norm.ppf(1.0 - significance_level / 2.0)
            widths = z * std_errs  # in *unscaled* rating units

            rating_lower = scale(ratings - widths)
            rating_upper = scale(ratings + widths)
            scaled_variances = rating_variances * (alpha**2)

        elif ci_method == "bootstrap":
            master_key = jax.random.PRNGKey(seed)
            keys = jax.random.split(master_key, num_bootstrap)
            worker_args = [(keys[i], self, dataset) for i in range(num_bootstrap)]
            n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

            # ctx = mp.get_context("spawn")
            # with ctx.Pool(processes=n_jobs) as pool:
            #     results = pool.starmap(fit_single_bootstrap_sample, worker_args)
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.starmap(fit_single_bootstrap_sample, worker_args)

            bootstrap_params = tree_util.tree_map(lambda *args: jnp.stack(args), *results)
            bootstrap_ratings = bootstrap_params["ratings"]  # [B, n_competitors] unscaled
            scaled_samples = scale(bootstrap_ratings)

            rating_lower = jnp.quantile(scaled_samples, significance_level / 2.0, axis=0)
            rating_upper = jnp.quantile(scaled_samples, 1.0 - significance_level / 2.0, axis=0)
            scaled_variances = jnp.var(scaled_samples, axis=0)

        else:
            raise ValueError(f"Unknown ci_method: {ci_method}")

        return {
            "competitors": dataset.competitors,
            "ratings": scaled_ratings,
            "coeffs": coeffs,
            "rating_lower": rating_lower,
            "rating_upper": rating_upper,
            "variances": scaled_variances,
        }
