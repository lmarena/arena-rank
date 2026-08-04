from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

from arena_rank.preprocessing.common import (
    OutcomeMap,
    default_outcome_map,
)


def get_outcomes(
    df: pd.DataFrame,
    outcome_col: str,
    outcome_map: OutcomeMap = default_outcome_map,
) -> np.ndarray:
    outcomes = df[outcome_col]
    if pd.api.types.is_numeric_dtype(outcomes.dtype):
        return outcomes.to_numpy(dtype=np.float64)
    return outcomes.map(outcome_map).to_numpy(dtype=np.float64)


def get_matchups_and_competitors(
    df: pd.DataFrame,
    competitor_cols: Sequence[str] = ("model_a", "model_b"),
) -> tuple[np.ndarray, list[str]]:
    n_rows = len(df)
    indices, competitors = pd.factorize(
        pd.concat([df[competitor_cols[0]], df[competitor_cols[1]]]),
        sort=True,
    )
    return np.column_stack([indices[:n_rows], indices[n_rows:]]).astype(np.int32), competitors.tolist()


def build_composite_data(
    dfs: Sequence[pd.DataFrame],
    outcome_cols: Sequence[str],
    loss_weights: Sequence[float],
    outcome_maps: Sequence[OutcomeMap | None] | None = None,
    weight_cols: Sequence[str | None] | None = None,
) -> tuple[dict[str, jnp.ndarray], list[str]]:
    outcome_maps = outcome_maps or [None] * len(dfs)
    weight_cols = weight_cols or [None] * len(dfs)
    competitors = sorted(pd.concat([df[col] for df in dfs for col in ("model_a", "model_b")]).unique().tolist())
    competitor_indices = {competitor: index for index, competitor in enumerate(competitors)}
    data = {"loss_weights": jnp.asarray(loss_weights, dtype=jnp.float64)}

    for index, (df, outcome_col, outcome_map, weight_col) in enumerate(
        zip(dfs, outcome_cols, outcome_maps, weight_cols)
    ):
        pairs = np.column_stack(
            [
                df["model_a"].map(competitor_indices).to_numpy(dtype=np.int32),
                df["model_b"].map(competitor_indices).to_numpy(dtype=np.int32),
            ]
        )
        outcomes = (
            get_outcomes(df, outcome_col, outcome_map)
            if outcome_map is not None
            else df[outcome_col].to_numpy(dtype=np.float64)
        )
        weights = (
            df[weight_col].to_numpy(dtype=np.float64) if weight_col is not None else np.ones(len(df), dtype=np.float64)
        )
        rows, counts = np.unique(
            np.column_stack([pairs, outcomes, weights]),
            return_counts=True,
            axis=0,
        )
        counts = counts.astype(np.float64)
        data[f"pairs_{index}"] = jnp.asarray(rows[:, :2], dtype=jnp.int32)
        data[f"outcomes_{index}"] = jnp.asarray(rows[:, 2], dtype=jnp.float64)
        data[f"counts_{index}"] = jnp.asarray(counts, dtype=jnp.float64)
        data[f"weights_{index}"] = jnp.asarray(rows[:, 3], dtype=jnp.float64)
        data[f"opt_weights_{index}"] = jnp.asarray(counts * rows[:, 3], dtype=jnp.float64)

    return data, competitors


def build_composite_contextual_data(
    dfs: Sequence[pd.DataFrame],
    outcome_cols: Sequence[str],
    loss_weights: Sequence[float],
    feature_cols_per_loss: Sequence[Sequence[str] | None] | None = None,
    outcome_maps: Sequence[OutcomeMap | None] | None = None,
    weight_cols: Sequence[str | None] | None = None,
    normalize_features: bool = True,
) -> tuple[dict[str, jnp.ndarray], list[str], list[int]]:
    feature_cols_per_loss = feature_cols_per_loss or [None] * len(dfs)
    outcome_maps = outcome_maps or [None] * len(dfs)
    weight_cols = weight_cols or [None] * len(dfs)
    competitors = sorted(pd.concat([df[col] for df in dfs for col in ("model_a", "model_b")]).unique().tolist())
    competitor_indices = {competitor: index for index, competitor in enumerate(competitors)}
    data = {"loss_weights": jnp.asarray(loss_weights, dtype=jnp.float64)}
    feature_dims: list[int] = []

    for index, (df, outcome_col, feature_cols, outcome_map, weight_col) in enumerate(
        zip(dfs, outcome_cols, feature_cols_per_loss, outcome_maps, weight_cols)
    ):
        pairs = np.column_stack(
            [
                df["model_a"].map(competitor_indices).to_numpy(dtype=np.int32),
                df["model_b"].map(competitor_indices).to_numpy(dtype=np.int32),
            ]
        )
        outcomes = (
            get_outcomes(df, outcome_col, outcome_map)
            if outcome_map is not None
            else df[outcome_col].to_numpy(dtype=np.float64)
        )
        weights = (
            df[weight_col].to_numpy(dtype=np.float64) if weight_col is not None else np.ones(len(df), dtype=np.float64)
        )
        if feature_cols:
            features = df[list(feature_cols)].to_numpy(dtype=np.float64)
            if normalize_features:
                std = features.std(axis=0)
                features = (features - features.mean(axis=0)) / np.where(std == 0, 1.0, std)
            feature_dims.append(len(feature_cols))
        else:
            features = np.zeros((len(df), 0), dtype=np.float64)
            feature_dims.append(0)

        data[f"pairs_{index}"] = jnp.asarray(pairs, dtype=jnp.int32)
        data[f"outcomes_{index}"] = jnp.asarray(outcomes, dtype=jnp.float64)
        data[f"counts_{index}"] = jnp.ones(len(df), dtype=jnp.float64)
        data[f"weights_{index}"] = jnp.asarray(weights, dtype=jnp.float64)
        data[f"opt_weights_{index}"] = jnp.asarray(weights, dtype=jnp.float64)
        data[f"features_{index}"] = jnp.asarray(features, dtype=jnp.float64)

    return data, competitors, feature_dims
