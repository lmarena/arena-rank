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
    theta: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray]:
    diff = theta[idx_a] - theta[idx_b]
    p = _sigmoid(diff)
    eps = 1e-12
    loss = -np.mean(target * np.log(p + eps) + (1.0 - target) * np.log(1.0 - p + eps))
    grad = np.zeros_like(theta)
    contrib = (p - target) / len(target)
    np.add.at(grad, idx_a, contrib)
    np.add.at(grad, idx_b, -contrib)
    return loss, grad


def _fit_thetas(
    idx_a: np.ndarray, idx_b: np.ndarray, target: np.ndarray, n_models: int
) -> tuple[np.ndarray, OptimizeResult]:
    theta0 = np.zeros(n_models)
    result = minimize(
        _binary_cross_entropy,
        theta0,
        args=(idx_a, idx_b, target),
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
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.bootstrap_seed = bootstrap_seed
        self.imputation = imputation
        self.anchor_model = anchor_model
        self.anchor_rating = float(anchor_rating)

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

        theta, scipy_result = _fit_thetas(idx_a, idx_b, target, len(models))
        fit_success = bool(scipy_result.success)
        fit_message = str(scipy_result.message)
        fit_iterations = int(scipy_result.nit)
        fit_loss = float(scipy_result.fun)

        anchor_idx = model_to_idx.get(self.anchor_model) if self.anchor_model else None
        rating = _to_rating(theta, anchor_idx, self.anchor_rating)

        # Bootstrap CIs
        rng = np.random.default_rng(self.bootstrap_seed)
        n = len(df)
        boot_ratings = np.empty((self.bootstrap_iterations, len(models)))
        for b in range(self.bootstrap_iterations):
            sample = rng.integers(0, n, size=n)
            theta_b, _ = _fit_thetas(idx_a[sample], idx_b[sample], target[sample], len(models))
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

        # Per-model counts and expected outcomes
        diff = theta[idx_a] - theta[idx_b]
        p_a_wins = _sigmoid(diff)
        expected_wins = np.zeros(len(models))
        expected_losses = np.zeros(len(models))
        np.add.at(expected_wins, idx_a, p_a_wins)
        np.add.at(expected_wins, idx_b, 1.0 - p_a_wins)
        np.add.at(expected_losses, idx_a, 1.0 - p_a_wins)
        np.add.at(expected_losses, idx_b, p_a_wins)
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
