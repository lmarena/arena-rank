"""Soft-target Bradley-Terry rating for factuality scores.

Differs from the existing arena_rank.BradleyTerry: uses scipy.optimize
(L-BFGS-B) over numpy arrays, with a soft cross-entropy target derived from
sigmoid((score_a - score_b) / temperature). This is the right tool for
small (hundreds of models) factuality fits and avoids pulling JAX into
this code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize


@dataclass(frozen=True, slots=True)
class FactualityBradleyTerryResults:
    ratings_df: pd.DataFrame  # columns: model_name, rating, rating_lower, rating_upper, variance, theta
    # columns: model_name, expected_wins, expected_losses, model_a_count, model_b_count, battle_count
    expected_outcomes: pd.DataFrame
    fit_success: bool
    fit_message: str
    fit_iterations: int
    fit_loss: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _impute_max_observed(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    if not finite.any():
        return np.where(np.isfinite(scores), scores, 0.0)
    fill = np.nanmax(scores[finite])
    out = np.where(finite, scores, fill)
    return out


def _binary_cross_entropy(
    theta: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    target: np.ndarray,
    opt_weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[float, np.ndarray]:
    """Weighted BT cross-entropy loss + ridge penalty, normalized by sum(opt_weights).

    Algebraically identical to the softplus form
    ``(1/N) · Σ opt_weights · (softplus(d) − y·d) + 0.5·λ·‖θ‖²``
    used in composite_bt.py, with ``d = θ[idx_a] − θ[idx_b]`` and
    ``N = sum(opt_weights)``.
    """
    diff = theta[idx_a] - theta[idx_b]
    p = _sigmoid(diff)
    eps = 1e-12
    w_sum = float(opt_weights.sum())
    if w_sum <= 0:
        raise ValueError("opt_weights.sum() must be positive")
    per_row = opt_weights * (target * np.log(p + eps) + (1.0 - target) * np.log(1.0 - p + eps))
    loss = -float(per_row.sum() / w_sum) + 0.5 * ridge_lambda * float(np.dot(theta, theta))
    contrib = opt_weights * (p - target) / w_sum
    grad = np.zeros_like(theta)
    np.add.at(grad, idx_a, contrib)
    np.add.at(grad, idx_b, -contrib)
    grad = grad + ridge_lambda * theta
    return loss, grad


def _fit_thetas(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    target: np.ndarray,
    opt_weights: np.ndarray,
    n_models: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, OptimizeResult]:
    theta0 = np.zeros(n_models)
    result = minimize(
        _binary_cross_entropy,
        theta0,
        args=(idx_a, idx_b, target, opt_weights, ridge_lambda),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9},
    )
    return result.x, result


_RATING_SCALE = 400.0 / np.log(10.0)
_RATING_OFFSET = 1000.0


def _to_rating(theta: np.ndarray, anchor_index: int | None = None, anchor_rating: float = _RATING_OFFSET) -> np.ndarray:
    rating = theta * _RATING_SCALE + _RATING_OFFSET
    if anchor_index is not None:
        rating = rating - rating[anchor_index] + anchor_rating
    return rating


_NORMAL_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def _hessian_at_optimum(
    theta: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    opt_weights: np.ndarray,
    n_models: int,
    ridge_lambda: float,
) -> np.ndarray:
    """Hessian of (1/N)·Σ opt_weights·(softplus(d) − y·d) + 0.5·λ·‖θ‖² at theta.

    The ridge term makes H positive-definite so callers can invert it
    directly with np.linalg.inv instead of falling back to a pseudo-inverse.
    """
    diff = theta[idx_a] - theta[idx_b]
    p = _sigmoid(diff)
    w_sum = float(opt_weights.sum())
    s = opt_weights * p * (1.0 - p) / w_sum
    H = np.zeros((n_models, n_models))
    np.add.at(H, (idx_a, idx_a), s)
    np.add.at(H, (idx_b, idx_b), s)
    np.add.at(H, (idx_a, idx_b), -s)
    np.add.at(H, (idx_b, idx_a), -s)
    if ridge_lambda > 0:
        H = H + ridge_lambda * np.eye(n_models)
    return H


def _closed_form_variance_rating(
    theta: np.ndarray,
    target: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    opt_weights: np.ndarray,
    counts: np.ndarray,
    n_models: int,
    ridge_lambda: float,
) -> np.ndarray:
    """Sandwich-estimator variance of ``rating`` per model.

    Matches composite_bt.py: ``Cov(θ) = H⁻¹ B H⁻¹`` where ``H`` is the
    (normalized, ridge-augmented) Hessian and ``B`` is the centered score
    covariance. The ``var_grad_vals / counts`` factor de-aggregates the
    contribution of buckets that represent multiple identical observations.

    The ridge term in ``H`` makes it positive-definite, so we invert
    directly with ``np.linalg.inv`` rather than a pseudo-inverse.
    """
    diff = theta[idx_a] - theta[idx_b]
    p = _sigmoid(diff)
    w_sum = float(opt_weights.sum())

    H = _hessian_at_optimum(theta, idx_a, idx_b, opt_weights, n_models, ridge_lambda)
    H_inv = np.linalg.inv(H)

    grad_vals = (p - target) * opt_weights
    var_grad_vals = (grad_vals**2) / counts

    mean_score = np.zeros(n_models)
    np.add.at(mean_score, idx_a, grad_vals / w_sum)
    np.add.at(mean_score, idx_b, -grad_vals / w_sum)

    second_moment = np.zeros((n_models, n_models))
    np.add.at(second_moment, (idx_a, idx_a), var_grad_vals)
    np.add.at(second_moment, (idx_b, idx_b), var_grad_vals)
    np.add.at(second_moment, (idx_a, idx_b), -var_grad_vals)
    np.add.at(second_moment, (idx_b, idx_a), -var_grad_vals)
    second_moment = second_moment / w_sum

    # Match composite_bt.py: sigma_k = second_moment - outer(mean_score, mean_score),
    # then B = sigma_k / N. The extra /N comes from the (w_k / N) coefficient on
    # each loss component's grad_cov contribution in compute_hessian_and_covariance.
    sigma = second_moment - np.outer(mean_score, mean_score)
    B = sigma / w_sum
    cov_theta = H_inv @ B @ H_inv
    variance_theta = np.clip(np.diag(cov_theta), a_min=0.0, a_max=None)
    return variance_theta * (_RATING_SCALE**2)


class FactualityBradleyTerry:
    def __init__(
        self,
        temperature: float,
        *,
        bootstrap_iterations: int = 100,
        bootstrap_seed: int | None = None,
        imputation: Literal["max_observed", "drop"] = "max_observed",
        anchor_model: str | None = None,
        anchor_rating: float = _RATING_OFFSET,
        ridge_lambda: float = 1e-5,
        method: Literal["closed_form", "bootstrap"] = "closed_form",
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if method not in ("closed_form", "bootstrap"):
            raise ValueError(f"method must be 'closed_form' or 'bootstrap', got {method!r}")
        self.temperature = float(temperature)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.bootstrap_seed = bootstrap_seed
        self.imputation = imputation
        self.anchor_model = anchor_model
        self.anchor_rating = float(anchor_rating)
        self.ridge_lambda = float(ridge_lambda)
        self.method = method

    def fit(self, battles: pd.DataFrame) -> FactualityBradleyTerryResults:
        df = battles.copy()
        if self.imputation == "drop":
            df = df.dropna(subset=["score_a", "score_b"])
            if df.empty:
                raise ValueError("No battles remain after dropping NULL scores")
        else:  # max_observed
            all_scores = np.concatenate([df["score_a"].to_numpy(dtype=float), df["score_b"].to_numpy(dtype=float)])
            imputed = _impute_max_observed(all_scores)
            df["score_a"] = imputed[: len(df)]
            df["score_b"] = imputed[len(df) :]

        models = pd.Index(sorted(set(df["model_a_name"]).union(df["model_b_name"])), name="model_name")
        model_to_idx = {m: i for i, m in enumerate(models)}
        idx_a = df["model_a_name"].map(model_to_idx).to_numpy()
        idx_b = df["model_b_name"].map(model_to_idx).to_numpy()
        score_a = df["score_a"].to_numpy(dtype=float)
        score_b = df["score_b"].to_numpy(dtype=float)
        target = _sigmoid((score_a - score_b) / self.temperature)

        if "sample_weight" in df.columns:
            sample_weight = df["sample_weight"].to_numpy(dtype=float)
        else:
            sample_weight = np.ones(len(df), dtype=float)

        # Aggregate identical (idx_a, idx_b, target, sample_weight) rows. The np.unique
        # step matches composite_bt.py: identical observations collapse into one row
        # with their count tracked separately, and opt_weights = counts × weights enters
        # both the loss normalization and the sandwich CI formula.
        rows = np.column_stack(
            [
                idx_a.astype(np.int64),
                idx_b.astype(np.int64),
                target,
                sample_weight,
            ]
        )
        unique_rows, counts = np.unique(rows, axis=0, return_counts=True)
        idx_a_agg = unique_rows[:, 0].astype(np.int64)
        idx_b_agg = unique_rows[:, 1].astype(np.int64)
        target_agg = unique_rows[:, 2].astype(float)
        weights_agg = unique_rows[:, 3].astype(float)
        counts_agg = counts.astype(float)
        opt_weights = counts_agg * weights_agg

        theta, scipy_result = _fit_thetas(idx_a_agg, idx_b_agg, target_agg, opt_weights, len(models), self.ridge_lambda)
        fit_success = bool(scipy_result.success)
        fit_message = str(scipy_result.message)
        fit_iterations = int(scipy_result.nit)
        fit_loss = float(scipy_result.fun)

        anchor_idx = model_to_idx.get(self.anchor_model) if self.anchor_model else None
        rating = _to_rating(theta, anchor_idx, self.anchor_rating)

        if self.method == "closed_form":
            variance = _closed_form_variance_rating(
                theta,
                target_agg,
                idx_a_agg,
                idx_b_agg,
                opt_weights,
                counts_agg,
                len(models),
                self.ridge_lambda,
            )
            half_width = _NORMAL_975 * np.sqrt(variance)
            rating_lower = rating - half_width
            rating_upper = rating + half_width
        else:
            # Bootstrap resamples FROM THE ORIGINAL non-aggregated arrays, not from
            # the aggregated ones — resampling aggregated rows would double-count.
            rng = np.random.default_rng(self.bootstrap_seed)
            n = len(df)
            if self.bootstrap_iterations <= 0:
                raise ValueError("bootstrap_iterations must be > 0 when method='bootstrap'")
            boot_ratings = np.empty((self.bootstrap_iterations, len(models)))
            for b in range(self.bootstrap_iterations):
                sample = rng.integers(0, n, size=n)
                theta_b, _ = _fit_thetas(
                    idx_a[sample],
                    idx_b[sample],
                    target[sample],
                    sample_weight[sample],
                    len(models),
                    self.ridge_lambda,
                )
                boot_ratings[b] = _to_rating(theta_b, anchor_idx, self.anchor_rating)
            rating_lower = np.quantile(boot_ratings, 0.025, axis=0)
            rating_upper = np.quantile(boot_ratings, 0.975, axis=0)
            variance = np.var(boot_ratings, axis=0)

        ratings_df = pd.DataFrame(
            {
                "model_name": models,
                "rating": rating,
                "rating_lower": rating_lower,
                "rating_upper": rating_upper,
                "variance": variance,
                "theta": theta,
            }
        )

        # Expected outcomes use weighted observed targets on aggregated buckets;
        # battle_count uses RAW (pre-aggregation) row counts.
        expected_wins = np.zeros(len(models))
        expected_losses = np.zeros(len(models))
        np.add.at(expected_wins, idx_a_agg, opt_weights * target_agg)
        np.add.at(expected_wins, idx_b_agg, opt_weights * (1.0 - target_agg))
        np.add.at(expected_losses, idx_a_agg, opt_weights * (1.0 - target_agg))
        np.add.at(expected_losses, idx_b_agg, opt_weights * target_agg)

        model_a_count = np.bincount(idx_a, minlength=len(models))
        model_b_count = np.bincount(idx_b, minlength=len(models))
        battle_count = model_a_count + model_b_count

        expected_outcomes = pd.DataFrame(
            {
                "model_name": models,
                "expected_wins": expected_wins,
                "expected_losses": expected_losses,
                "model_a_count": model_a_count,
                "model_b_count": model_b_count,
                "battle_count": battle_count,
            }
        )

        return FactualityBradleyTerryResults(
            ratings_df=ratings_df,
            expected_outcomes=expected_outcomes,
            fit_success=fit_success,
            fit_message=fit_message,
            fit_iterations=fit_iterations,
            fit_loss=fit_loss,
        )
