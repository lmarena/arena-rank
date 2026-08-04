"""Composite contextual Bradley-Terry rating system implementation in JAX.

Combines the composite multi-loss objective of :class:`CompositeBradleyTerry`
with the per-observation feature regression of :class:`ContextualBradleyTerry`.
Each loss component supplies its own (potentially empty) feature matrix and gets
its own coefficient vector, while a single latent rating vector is shared across
all components::

    L(theta, {beta_k}) = sum_k w_k * L_k(theta, beta_k)
                         + 0.5 * reg * sum_k ||beta_k||^2

where each component logit is ``r_i - r_j + f^T beta_k``. A component with zero
features has an empty ``beta_k`` and a ``(n_obs, 0)`` feature matrix, so its
context contribution is identically zero and it reduces to a plain Bradley-Terry
term. This lets a contextual dataset (e.g. style-controlled human votes) be
jointly fit with a plain one (e.g. factuality battles) in a single optimization.

As in :class:`ContextualBradleyTerry`, the ratings carry no penalty in the fit;
``hessian_reg`` only stabilizes the Hessian diagonal for confidence intervals,
while ``reg`` is a genuine L2 penalty on the feature coefficients.

Unlike the non-contextual preprocessing path, observations are not aggregated
by matchup because features vary per row.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax import jit
from jaxtyping import PyTree

from arena_rank.utils.math_utils import assemble_parwise_matrix, lbfgs_minimize

jax.config.update("jax_enable_x64", True)

__all__ = [
    "CompositeContextualBradleyTerry",
]


class CompositeContextualBradleyTerry:
    """Composite Bradley-Terry model with per-component contextual features.

    The objective jointly optimizes a shared rating vector and per-component
    feature coefficients::

        L = sum_k w_k * L_k(theta, beta_k) + 0.5 * reg * sum_k ||beta_k||^2

    where ``L_k`` is a standard BT log-likelihood on component ``k``'s pairwise
    comparisons with logit ``r_i - r_j + f^T beta_k``, normalized by its own
    total weight mass. Components with zero features carry an empty ``beta_k``
    and reduce to plain BT terms. The ratings carry no penalty in the fit;
    ``hessian_reg`` only stabilizes the Hessian diagonal for confidence
    intervals.

    Attributes:
        n_features_per_loss: Per-component feature count (0 for plain BT
            components).
        params: PyTree with key ``"ratings"`` plus one ``f"coeffs_{k}"`` entry
            per component.
    """

    def __init__(
        self,
        n_competitors: int,
        n_features_per_loss: list[int],
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        hessian_reg: float = 1e-5,
        reg: float = 1.0,
        max_iter: int = 1000,
        ftol: float = 1e-9,
        gtol: float = 1e-9,
        dtype: Any = jnp.float64,
        verbose: bool = False,
    ):
        self.n_competitors = n_competitors
        self.n_features_per_loss = list(n_features_per_loss)
        self.n_losses = len(self.n_features_per_loss)
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.hessian_reg = hessian_reg
        self.reg = reg
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
        self.dtype = dtype
        self.verbose = verbose
        self.fitted = False
        self.alpha = scale / math.log(base)
        # Only feature-bearing components carry a coefficient leaf. Empty
        # (shape ``(0,)``) leaves would break L-BFGS's gradient-norm reduction.
        self.params: PyTree = {"ratings": jnp.zeros(n_competitors, dtype=dtype)}
        for k, f_k in enumerate(self.n_features_per_loss):
            if f_k > 0:
                self.params[f"coeffs_{k}"] = jnp.zeros(f_k, dtype=dtype)

    @staticmethod
    @partial(jit, static_argnames=["n_losses", "feature_dims"])
    def loss_function(
        params: PyTree,
        data: PyTree,
        n_losses: int = 2,
        feature_dims: tuple[int, ...] = (),
        reg: float = 0.0,
    ) -> jnp.ndarray:
        """Returns the composite contextual BT NLL plus the feature penalty.

        Args:
            params: Dict with key ``"ratings"`` and one ``f"coeffs_{k}"`` entry
                per feature-bearing component.
            data: Dict produced by :func:`arena_rank.preprocessing.build_composite_contextual_data`.
            n_losses: Number of loss components (static for JIT).
            feature_dims: Per-component feature count (static for JIT). A count
                of 0 means the component contributes a plain BT term.
            reg: L2 penalty on the feature coefficients of every component.
        """
        ratings = params["ratings"]
        loss_weights = data["loss_weights"]

        total = jnp.float64(0.0)
        coeff_penalty = jnp.float64(0.0)
        for k in range(n_losses):
            matchups = data[f"pairs_{k}"]
            opt_weights = data[f"opt_weights_{k}"]
            outcomes = data[f"outcomes_{k}"]

            logits = ratings[matchups[:, 0]] - ratings[matchups[:, 1]]
            if feature_dims[k] > 0:
                coeffs = params[f"coeffs_{k}"]
                logits = logits + data[f"features_{k}"] @ coeffs
                coeff_penalty = coeff_penalty + jnp.sum(coeffs**2)

            nll_k = -jnp.sum(opt_weights * (outcomes * logits - nn.softplus(logits)))
            total = total + loss_weights[k] * nll_k / jnp.sum(opt_weights)

        return total + 0.5 * reg * coeff_penalty

    @staticmethod
    @partial(jit, static_argnames=["n_competitors", "n_losses", "feature_dims"])
    def compute_hessian_and_covariance(
        params: PyTree,
        data: PyTree,
        hessian_reg: float,
        reg: float,
        n_competitors: int,
        n_losses: int,
        feature_dims: tuple[int, ...],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the joint Hessian and gradient covariance for the sandwich estimator.

        The parameter vector is laid out as ``[ratings, coeffs_0, coeffs_1, ...]``
        of total dimension ``P = n_competitors + sum(feature_dims)``. Each
        component contributes to the shared rating block, its own coefficient
        block, and the rating/coefficient cross blocks::

            H = sum_k (w_k / N_k) * H_k + hessian_reg * I + reg * I_coeffs
            B = sum_k (w_k^2 / N_k) * Sigma_k

        where ``Sigma_k`` is the centered empirical score covariance and
        ``N_k = sum(opt_weights_k)``. ``hessian_reg`` stabilizes the full
        diagonal for inversion; ``reg`` adds the curvature of the in-loss
        coefficient penalty on the coefficient block (the ratings carry no
        penalty in the fit). Components are assumed independent, so
        cross-component covariance terms are zero.

        Args:
            params: Fitted parameter PyTree.
            data: Dict produced by :func:`arena_rank.preprocessing.build_composite_contextual_data`.
            hessian_reg: Diagonal regularizer added to the full Hessian for
                numerical stability.
            reg: L2 penalty on feature coefficients (curvature added to the
                coefficient block).
            n_competitors: Total number of competitors (static for JIT).
            n_losses: Number of loss components (static for JIT).
            feature_dims: Per-component feature count (static for JIT).
        """
        ratings = params["ratings"]
        loss_weights = data["loss_weights"]
        dtype = ratings.dtype

        n_params = n_competitors + sum(feature_dims)
        hessian = jnp.zeros((n_params, n_params), dtype=dtype)
        grad_cov = jnp.zeros((n_params, n_params), dtype=dtype)

        # Starting column of each component's coefficient block.
        offsets = []
        off = n_competitors
        for f_k in feature_dims:
            offsets.append(off)
            off += f_k

        for k in range(n_losses):
            matchups = data[f"pairs_{k}"]
            counts = data[f"counts_{k}"]
            opt_weights = data[f"opt_weights_{k}"]
            outcomes = data[f"outcomes_{k}"]
            features = data[f"features_{k}"]

            f_k = feature_dims[k]
            o = offsets[k]
            n_k = jnp.sum(opt_weights)
            idx_a = matchups[:, 0]
            idx_b = matchups[:, 1]

            logits = ratings[idx_a] - ratings[idx_b]
            if f_k > 0:
                logits = logits + features @ params[f"coeffs_{k}"]
            probs = jax.nn.sigmoid(logits)

            # ---- Hessian contribution (scaled by w_k / N_k) ----
            hess_vals = probs * (1.0 - probs) * opt_weights
            h_rr = assemble_parwise_matrix(
                jnp.zeros((n_competitors, n_competitors), dtype=dtype), hess_vals, idx_a, idx_b
            )
            h_scale = loss_weights[k] / n_k
            hessian = hessian.at[:n_competitors, :n_competitors].add(h_scale * h_rr)

            if f_k > 0:
                hess_weighted_feats = hess_vals[:, None] * features  # (n_obs, f_k)
                h_cc = features.T @ hess_weighted_feats  # (f_k, f_k)
                h_rc = jnp.zeros((n_competitors, f_k), dtype=dtype)
                h_rc = h_rc.at[idx_a, :].add(hess_weighted_feats)
                h_rc = h_rc.at[idx_b, :].add(-hess_weighted_feats)
                hessian = hessian.at[:n_competitors, o : o + f_k].add(h_scale * h_rc)
                hessian = hessian.at[o : o + f_k, :n_competitors].add(h_scale * h_rc.T)
                hessian = hessian.at[o : o + f_k, o : o + f_k].add(h_scale * h_cc)

            # ---- Gradient covariance contribution (scaled by w_k^2 / N_k) ----
            # Score covariance divides by counts (not opt_weights) because each
            # bucket of c_r identical observations shares the same score value.
            grad_vals = (probs - outcomes) * opt_weights
            var_grad_vals = (grad_vals**2) / counts

            mean_score = jnp.zeros(n_params, dtype=dtype)
            mean_score = mean_score.at[idx_a].add(grad_vals / n_k)
            mean_score = mean_score.at[idx_b].add(-grad_vals / n_k)

            second_moment = jnp.zeros((n_params, n_params), dtype=dtype)
            sm_rr = assemble_parwise_matrix(
                jnp.zeros((n_competitors, n_competitors), dtype=dtype), var_grad_vals, idx_a, idx_b
            )
            second_moment = second_moment.at[:n_competitors, :n_competitors].add(sm_rr)

            if f_k > 0:
                var_weighted_feats = var_grad_vals[:, None] * features  # (n_obs, f_k)
                sm_cc = features.T @ var_weighted_feats  # (f_k, f_k)
                sm_rc = jnp.zeros((n_competitors, f_k), dtype=dtype)
                sm_rc = sm_rc.at[idx_a, :].add(var_weighted_feats)
                sm_rc = sm_rc.at[idx_b, :].add(-var_weighted_feats)
                second_moment = second_moment.at[:n_competitors, o : o + f_k].add(sm_rc)
                second_moment = second_moment.at[o : o + f_k, :n_competitors].add(sm_rc.T)
                second_moment = second_moment.at[o : o + f_k, o : o + f_k].add(sm_cc)

                mean_score = mean_score.at[o : o + f_k].add((features * grad_vals[:, None]).sum(axis=0) / n_k)

            second_moment = second_moment / n_k
            sigma_k = second_moment - jnp.outer(mean_score, mean_score)
            grad_cov = grad_cov + (loss_weights[k] ** 2 / n_k) * sigma_k

        # hessian_reg stabilizes the full diagonal; reg adds the in-loss
        # coefficient-penalty curvature on the coefficient block only.
        reg_diag = jnp.concatenate(
            [
                jnp.full(n_competitors, hessian_reg, dtype=dtype),
                jnp.full(sum(feature_dims), hessian_reg + reg, dtype=dtype),
            ]
        )
        hessian = hessian + jnp.diag(reg_diag)
        return hessian, grad_cov

    def fit(self, data_dict: dict[str, jnp.ndarray]) -> "CompositeContextualBradleyTerry":
        """Fits the model via L-BFGS.

        Args:
            data_dict: Dict produced by :func:`arena_rank.preprocessing.build_composite_contextual_data`.

        Returns:
            ``self`` (for chaining).
        """
        opt_fn = partial(
            self.loss_function,
            data=data_dict,
            n_losses=self.n_losses,
            feature_dims=tuple(self.n_features_per_loss),
            reg=self.reg,
        )
        self.params, _ = lbfgs_minimize(
            function=opt_fn,
            initial_params=self.params,
            max_iter=self.max_iter,
            gtol=self.gtol,
            ftol=self.ftol,
            verbose=self.verbose,
        )
        self.fitted = True
        return self

    def compute_ratings_and_cis(
        self,
        data_dict: dict[str, jnp.ndarray],
        competitors: list[str],
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        """Fits the model (if needed) and returns ratings, coeffs, and sandwich CIs.

        Uses the sandwich estimator ``Var(theta_hat) = H^{-1} B H^{-1}`` over the
        joint ``[ratings, coeffs_0, ...]`` parameter vector, then extracts the
        rating block.

        Args:
            data_dict: Dict produced by :func:`arena_rank.preprocessing.build_composite_contextual_data`.
            competitors: Sorted list of model names (index mapping).
            significance_level: Two-sided significance level (default 0.05
                gives 95% CIs).

        Returns:
            Dict with keys ``"competitors"``, ``"ratings"``, ``"rating_lower"``,
            ``"rating_upper"``, ``"variances"`` (all in the scaled Elo-like
            space), and ``"coeffs"`` (per-component list of feature coefficients,
            in unscaled logit space).
        """
        if not self.fitted:
            self.fit(data_dict)

        ratings = self.params["ratings"]
        coeffs = [
            self.params[f"coeffs_{k}"] if f_k > 0 else jnp.zeros(0, dtype=ratings.dtype)
            for k, f_k in enumerate(self.n_features_per_loss)
        ]

        hessian, grad_cov = self.compute_hessian_and_covariance(
            params=self.params,
            data=data_dict,
            hessian_reg=self.hessian_reg,
            reg=self.reg,
            n_competitors=self.n_competitors,
            n_losses=self.n_losses,
            feature_dims=tuple(self.n_features_per_loss),
        )

        hessian_inv = jnp.linalg.inv(hessian)
        covariance = hessian_inv @ grad_cov @ hessian_inv
        variances = jnp.diag(covariance)[: self.n_competitors]

        std_errs = jnp.sqrt(jnp.maximum(variances, 0.0))
        z = jax.scipy.stats.norm.ppf(1.0 - significance_level / 2.0)
        widths = z * std_errs

        alpha = self.alpha
        offset = self.init_rating

        def scale(x: jnp.ndarray) -> jnp.ndarray:
            return x * alpha + offset

        return {
            "competitors": competitors,
            "ratings": scale(ratings),
            "rating_lower": scale(ratings - widths),
            "rating_upper": scale(ratings + widths),
            "variances": variances * (alpha**2),
            "coeffs": coeffs,
        }
