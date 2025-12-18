"""Bradley-Terry rating system implementation in JAX."""

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
from arena_rank.utils.data_utils import PairDataset
from arena_rank.utils.math_utils import assemble_parwise_matrix

jax.config.update("jax_enable_x64", True)


# needs to be at the top level to be pickle-able for multiprocessing
def fit_single_bootstrap_sample(
    boot_counts: jnp.ndarray,
    model: RatingSystem,
    dataset: PairDataset,
) -> PyTree:
    """Helper function to fit a single bootstrap sample with given counts."""
    # opt_weights = counts * weights (combining occurrence counts and reweighting)
    boot_opt_weights = boot_counts * dataset.weights

    # Create bootstrap dataset with resampled counts
    boot_dataset = PairDataset(
        competitors=dataset.competitors,
        pairs=dataset.pairs,
        outcomes=dataset.outcomes,
        counts=boot_counts,
        weights=dataset.weights,
        opt_weights=boot_opt_weights,  # same dataset but with weights based on the bootstrap counts
    )

    boot_model = deepcopy(model)
    boot_model.params = {"ratings": jnp.zeros_like(boot_model.params["ratings"], dtype=model.dtype)}
    boot_model.fit(boot_dataset)
    return boot_model.params


class BradleyTerry(RatingSystem):
    """Bradley-Terry rating system implementation."""

    def __init__(
        self,
        n_competitors,
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        hessian_reg: float = 1e-5,
        max_iter: int = 1000,
        ftol: float = 1e-9,
        gtol: float = 1e-9,
        dtype=jnp.float64,
        verbose: bool = False,
    ):
        self.n_competitors = n_competitors
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.hessian_reg = hessian_reg
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
        self.dtype = dtype
        self.verbose = verbose
        self.fitted = False
        self.alpha = scale / math.log(base)
        self.params = {"ratings": jnp.zeros(n_competitors, dtype=dtype)}

    @staticmethod
    @jit
    def loss_function(params, data: PyTree) -> jnp.ndarray:
        ratings = params["ratings"]
        matchups = data["pairs"]
        weights = data["opt_weights"]
        outcomes = data["outcomes"]
        rating_diffs = ratings[matchups[:, 0]] - ratings[matchups[:, 1]]
        loss = -jnp.sum(weights * (outcomes * rating_diffs - nn.softplus(rating_diffs))) / jnp.sum(weights)
        return loss

    @staticmethod
    @partial(jit, static_argnames=["n_competitors"])
    def compute_hessian_and_covariance(
        ratings: jnp.ndarray,
        matchups: jnp.ndarray,
        outcomes: jnp.ndarray,
        counts: jnp.ndarray,
        opt_weights: jnp.ndarray,
        hessian_reg: float,
        n_competitors: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the Hessian and Gradient Covariance matrices for the sandwich estimator in the Bradley-Terry framework.

        Args:
            ratings: Competitor ratings, shape (n_competitors,)
            matchups: Matchup pairs, shape (n_pairs, 2)
            outcomes: Outcomes of matchups, shape (n_pairs,)
            counts: Counts of each unique pair, shape (n_pairs,)
            opt_weights: Optimization weights for each matchup, incorporates both occurrence counts and reweighting, shape (n_pairs,)
            hessian_reg: Regularization term for Hessian diagonal.
            n_competitors: Total number of competitors.
        """
        matchup_ratings = ratings[matchups]
        logits = matchup_ratings[:, 0] - matchup_ratings[:, 1]
        probs = jax.nn.sigmoid(logits)

        grad_vals = (outcomes - probs) * opt_weights
        # variance per unique pair bucket
        var_grad_vals = (grad_vals**2) / counts
        hess_vals = probs * (1.0 - probs) * opt_weights

        idx_a = matchups[:, 0]
        idx_b = matchups[:, 1]

        hessian = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
        grad_cov = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)

        hessian = assemble_parwise_matrix(hessian, hess_vals, idx_a, idx_b)
        grad_cov = assemble_parwise_matrix(grad_cov, var_grad_vals, idx_a, idx_b)

        # regularize the diagonal of the Hessian for numerical stability
        hessian = hessian + (jnp.eye(n_competitors) * hessian_reg)
        return hessian, grad_cov

    def compute_ratings_and_cis(
        self,
        dataset: PairDataset,
        significance_level: float = 0.05,
        ci_method: str = "sandwich",
        num_bootstrap: int = 100,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """
        Fits the model (if needed), calculates confidence intervals.

        Args:
            dataset: PairDataset containing the matchup data.
            significance_level: Significance level for confidence intervals.
            ci_method: "sandwich" for central limit theorem estimates or "bootstrap" for resampling.
            num_bootstrap: Number of bootstrap samples (only used if ci_method="bootstrap").
            seed: Random seed for bootstrapping.
            n_jobs: Number of workers for multiprocessing (only used if ci_method="bootstrap").
        """
        if not self.fitted:
            self.fit(dataset)

        ratings = self.params["ratings"]  # unscaled ratings from the fitted model

        alpha = self.alpha
        offset = self.init_rating

        def scale(x):
            return x * alpha + offset

        scaled_ratings = scale(ratings)
        rating_lower = None
        rating_upper = None
        scaled_variances = None

        total_battles = jnp.sum(dataset.counts)
        if ci_method == "sandwich":
            is_unweighted = jnp.allclose(dataset.weights, 1.0)
            reg_factor = jnp.where(is_unweighted, total_battles, 1.0)
            effective_reg = self.hessian_reg * reg_factor

            hessian, gradient_cov = self.compute_hessian_and_covariance(
                ratings,
                dataset.pairs,
                dataset.outcomes,
                dataset.counts,
                dataset.opt_weights,
                effective_reg,
                self.n_competitors,
            )

            hessian_inv = jnp.linalg.inv(hessian)
            covariance_matrix = hessian_inv @ gradient_cov @ hessian_inv
            variances = jnp.diag(covariance_matrix)

            std_errs = jnp.sqrt(variances)
            z_score = jax.scipy.stats.norm.ppf(1 - significance_level / 2)
            interval_widths = z_score * std_errs

            rating_lower = scale(ratings - interval_widths)
            rating_upper = scale(ratings + interval_widths)
            scaled_variances = variances * (alpha**2)

        elif ci_method == "bootstrap":
            # generate all bootstrap count samples using multinomial distribution
            key = jax.random.PRNGKey(seed)
            boot_counts_all = jax.random.multinomial(
                key=key,
                n=int(total_battles),
                p=dataset.counts / total_battles,
                shape=(num_bootstrap, dataset.counts.shape[0]),
                dtype=self.dtype,
            )

            # prepare arguments for multiprocessing
            worker_args = [(boot_counts_all[i], self, dataset) for i in range(num_bootstrap)]
            n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

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
            "rating_lower": rating_lower,
            "rating_upper": rating_upper,
            "variances": scaled_variances,
        }
