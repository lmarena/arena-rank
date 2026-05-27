"""Composite Bradley-Terry rating system implementation in JAX.

Extends the standard Bradley-Terry model to support multiple loss components:

    L(theta) = sum_k w_k * L_k(theta) + 0.5 * ridge_lambda * ||theta||^2

Each L_k is a standard BT negative log-likelihood evaluated on a (potentially
different) set of pairwise outcomes, all sharing the same latent rating vector
theta. Per-component NLL is normalized by its own total weight mass so that
loss_weights[k] directly controls relative influence regardless of sample size.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jaxtyping import PyTree

from arena_rank.utils.math_utils import assemble_parwise_matrix, lbfgs_minimize

jax.config.update("jax_enable_x64", True)

DEFAULT_OUTCOME_MAP: dict[str, float] = {
    "model_a": 1.0,
    "model_b": 0.0,
    "tie": 0.5,
    "both_bad": 0.5,
}


def build_composite_data(
    dfs: list[pd.DataFrame],
    outcome_cols: list[str],
    loss_weights: list[float],
    outcome_maps: list[Callable | None] | None = None,
    weight_cols: list[str | None] | None = None,
) -> tuple[dict[str, jnp.ndarray], list[str]]:
    """Prepares the data dict consumed by :class:`CompositeBradleyTerry`.

    Each element of *dfs* supplies one loss component. Every DataFrame must
    contain ``model_a`` and ``model_b`` columns plus the column named in the
    corresponding entry of *outcome_cols*.

    Args:
        dfs: One DataFrame per loss component.
        outcome_cols: Column holding the outcome value in each DataFrame.
        loss_weights: Scalar weight w_k for each loss component.
        outcome_maps: Per-component mapping applied to outcome values
            (e.g. a dict ``get`` for string-to-float conversion). ``None``
            entries leave outcomes as-is.
        weight_cols: Per-component column name holding per-observation weights
            (e.g. inverse-propensity weights). ``None`` entries default to
            uniform weights of 1. After aggregation, ``opt_weights = counts *
            weights`` (arena-rank convention).

    Returns:
        A tuple ``(data_dict, competitors)`` where *data_dict* is a JAX-friendly
        dict and *competitors* is the sorted list of all model names (shared
        index mapping).
    """
    if outcome_maps is None:
        outcome_maps = [None] * len(dfs)
    if weight_cols is None:
        weight_cols = [None] * len(dfs)

    all_models: set[str] = set()
    for df in dfs:
        all_models.update(df["model_a"])
        all_models.update(df["model_b"])
    competitors = sorted(all_models)
    comp_to_idx = {m: i for i, m in enumerate(competitors)}

    data_dict: dict[str, jnp.ndarray] = {
        "loss_weights": jnp.array(loss_weights, dtype=jnp.float64),
    }

    for k, (df, col, omap, wcol) in enumerate(zip(dfs, outcome_cols, outcome_maps, weight_cols)):
        idx_a = np.array([comp_to_idx[m] for m in df["model_a"]], dtype=np.int32)
        idx_b = np.array([comp_to_idx[m] for m in df["model_b"]], dtype=np.int32)
        matchups = jnp.column_stack([jnp.array(idx_a), jnp.array(idx_b)])

        if omap is not None:
            outcomes = jnp.array(df[col].map(omap).values, dtype=jnp.float64)
        else:
            outcomes = jnp.array(df[col].values.astype(float), dtype=jnp.float64)

        if wcol is not None:
            weights = jnp.array(df[wcol].values.astype(float), dtype=jnp.float64)
        else:
            weights = jnp.ones(len(df), dtype=jnp.float64)

        rows = jnp.column_stack([matchups.astype(jnp.float64), outcomes, weights])
        unique_rows, counts = jnp.unique(rows, return_counts=True, axis=0)

        u_matchups = unique_rows[:, :2].astype(jnp.int32)
        u_outcomes = unique_rows[:, 2]
        u_weights = unique_rows[:, 3]
        u_counts = counts.astype(jnp.float64)

        data_dict[f"pairs_{k}"] = u_matchups
        data_dict[f"outcomes_{k}"] = u_outcomes
        data_dict[f"counts_{k}"] = u_counts
        data_dict[f"weights_{k}"] = u_weights
        data_dict[f"opt_weights_{k}"] = u_counts * u_weights

    return data_dict, competitors


class CompositeBradleyTerry:
    """Bradley-Terry model with composite loss and optional ridge penalty.

    The objective is::

        L(theta) = sum_k w_k * L_k(theta) + 0.5 * ridge_lambda * ||theta||^2

    Each component L_k is a standard BT log-likelihood on its own set of
    pairwise comparisons. All components share the same rating vector theta,
    so the optimum balances the different quality axes according to the
    user-supplied *loss_weights*.
    """

    def __init__(
        self,
        n_competitors: int,
        n_losses: int = 2,
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        ridge_lambda: float = 1e-5,
        max_iter: int = 1000,
        ftol: float = 1e-9,
        gtol: float = 1e-9,
        dtype: Any = jnp.float64,
        verbose: bool = False,
    ):
        self.n_competitors = n_competitors
        self.n_losses = n_losses
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.ridge_lambda = ridge_lambda
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
        self.dtype = dtype
        self.verbose = verbose
        self.fitted = False
        self.alpha = scale / math.log(base)
        self.params: PyTree = {"ratings": jnp.zeros(n_competitors, dtype=dtype)}

    @staticmethod
    @partial(jit, static_argnames=["n_losses"])
    def loss_function(
        params: PyTree,
        data: PyTree,
        n_losses: int = 2,
        ridge_lambda: float = 0.0,
    ) -> jnp.ndarray:
        """Returns the composite BT negative log-likelihood plus ridge penalty.

        Args:
            params: Dict with key ``"ratings"`` of shape ``(n_competitors,)``.
            data: Dict produced by :func:`build_composite_data`.
            n_losses: Number of loss components (static for JIT).
            ridge_lambda: L2 penalty strength.
        """
        ratings = params["ratings"]
        loss_weights = data["loss_weights"]

        total = jnp.float64(0.0)
        for k in range(n_losses):
            matchups = data[f"pairs_{k}"]
            opt_weights = data[f"opt_weights_{k}"]
            outcomes = data[f"outcomes_{k}"]

            logits = ratings[matchups[:, 0]] - ratings[matchups[:, 1]]
            nll_k = -jnp.sum(opt_weights * (outcomes * logits - nn.softplus(logits)))
            total = total + loss_weights[k] * nll_k / jnp.sum(opt_weights)

        return total + 0.5 * ridge_lambda * jnp.sum(ratings**2)

    @staticmethod
    @partial(jit, static_argnames=["n_competitors", "n_losses"])
    def compute_hessian_and_covariance(
        ratings: jnp.ndarray,
        data: PyTree,
        ridge_lambda: float,
        n_competitors: int,
        n_losses: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the Hessian and gradient covariance for the sandwich estimator.

        Computes::

            H = sum_k (w_k / N_k) * H_k + ridge_lambda * I
            B = sum_k (w_k^2 / N_k) * Sigma_k

        where ``H_k`` is the per-component Hessian (unnormalised), ``Sigma_k``
        is the centered empirical score covariance, and ``N_k =
        sum(opt_weights_k)``. Components are assumed independent so
        cross-component covariance terms are zero.

        Args:
            ratings: Fitted rating vector, shape ``(n_competitors,)``.
            data: Dict produced by :func:`build_composite_data`.
            ridge_lambda: Ridge penalty (enters the Hessian, not the gradient
                covariance).
            n_competitors: Total number of competitors (static for JIT).
            n_losses: Number of loss components (static for JIT).
        """
        loss_weights = data["loss_weights"]
        hessian = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
        grad_cov = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)

        for k in range(n_losses):
            matchups = data[f"pairs_{k}"]
            counts = data[f"counts_{k}"]
            opt_weights = data[f"opt_weights_{k}"]
            outcomes = data[f"outcomes_{k}"]

            n_k = jnp.sum(opt_weights)
            idx_a = matchups[:, 0]
            idx_b = matchups[:, 1]

            logits = ratings[idx_a] - ratings[idx_b]
            probs = jax.nn.sigmoid(logits)

            hess_vals = probs * (1.0 - probs) * opt_weights
            h_k = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
            h_k = assemble_parwise_matrix(h_k, hess_vals, idx_a, idx_b)
            hessian = hessian + (loss_weights[k] / n_k) * h_k

            # Score covariance divides by counts (not opt_weights) because each
            # bucket of c_r identical observations shares the same score value.
            grad_vals = (probs - outcomes) * opt_weights
            var_grad_vals = (grad_vals**2) / counts

            mean_score = jnp.zeros(n_competitors, dtype=ratings.dtype)
            mean_score = mean_score.at[idx_a].add(grad_vals / n_k)
            mean_score = mean_score.at[idx_b].add(-grad_vals / n_k)

            second_moment = jnp.zeros((n_competitors, n_competitors), dtype=ratings.dtype)
            second_moment = assemble_parwise_matrix(second_moment, var_grad_vals, idx_a, idx_b)
            second_moment = second_moment / n_k

            sigma_k = second_moment - jnp.outer(mean_score, mean_score)
            grad_cov = grad_cov + (loss_weights[k] ** 2 / n_k) * sigma_k

        hessian = hessian + ridge_lambda * jnp.eye(n_competitors, dtype=ratings.dtype)
        return hessian, grad_cov

    def fit(self, data_dict: dict[str, jnp.ndarray]) -> "CompositeBradleyTerry":
        """Fits the model via L-BFGS.

        Args:
            data_dict: Dict produced by :func:`build_composite_data`.

        Returns:
            ``self`` (for chaining).
        """
        opt_fn = partial(
            self.loss_function,
            data=data_dict,
            n_losses=self.n_losses,
            ridge_lambda=self.ridge_lambda,
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
        """Fits the model (if needed) and returns ratings with sandwich CIs.

        Uses the sandwich estimator ``Var(theta_hat) = H^{-1} B H^{-1}`` where
        *H* is the composite Hessian (including the ridge term) and *B* is the
        composite gradient covariance.

        Args:
            data_dict: Dict produced by :func:`build_composite_data`.
            competitors: Sorted list of model names (index mapping).
            significance_level: Two-sided significance level (default 0.05
                gives 95% CIs).

        Returns:
            Dict with keys ``"competitors"``, ``"ratings"``, ``"rating_lower"``,
            ``"rating_upper"``, ``"variances"`` (all in the scaled Elo-like
            space).
        """
        if not self.fitted:
            self.fit(data_dict)

        ratings = self.params["ratings"]

        hessian, grad_cov = self.compute_hessian_and_covariance(
            ratings=ratings,
            data=data_dict,
            ridge_lambda=self.ridge_lambda,
            n_competitors=self.n_competitors,
            n_losses=self.n_losses,
        )

        hessian_inv = jnp.linalg.inv(hessian)
        covariance = hessian_inv @ grad_cov @ hessian_inv
        variances = jnp.diag(covariance)

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
        }
