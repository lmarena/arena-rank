from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import polars as pl

from arena_rank.preprocessing.common import (
    OutcomeMap,
)


def _outcome_expr(
    df: pl.DataFrame,
    outcome_col: str,
    outcome_map: OutcomeMap | None,
) -> pl.Expr:
    if df.schema[outcome_col].is_numeric():
        return pl.col(outcome_col).cast(pl.Float64)
    if isinstance(outcome_map, Mapping):
        return pl.col(outcome_col).replace_strict(outcome_map, default=None, return_dtype=pl.Float64)
    if outcome_map is not None:
        return pl.col(outcome_col).map_elements(outcome_map, return_dtype=pl.Float64)
    return pl.col(outcome_col).cast(pl.Float64)


def _competitors(dfs: Sequence[pl.DataFrame]) -> list[str]:
    return (
        pl.concat([df.select(pl.col(col).alias("competitor")) for df in dfs for col in ("model_a", "model_b")])
        .get_column("competitor")
        .unique()
        .sort()
        .to_list()
    )


def _competitor_indices(
    df: pl.DataFrame,
    competitor_cols: Sequence[str],
) -> tuple[list[str], dict[str, int]]:
    competitors = (
        pl.concat([df.select(pl.col(column).alias("competitor")) for column in competitor_cols])
        .get_column("competitor")
        .unique()
        .sort()
        .to_list()
    )
    return competitors, {competitor: index for index, competitor in enumerate(competitors)}


def _pair_exprs(competitor_indices: Mapping[str, int]) -> list[pl.Expr]:
    return [
        pl.col(column).replace_strict(competitor_indices, return_dtype=pl.Int32).alias(alias)
        for column, alias in (("model_a", "_a"), ("model_b", "_b"))
    ]


def _feature_exprs(
    feature_cols: Sequence[str],
    normalize_features: bool,
) -> list[pl.Expr]:
    if not normalize_features:
        return [pl.col(column).cast(pl.Float64) for column in feature_cols]
    expressions = []
    for column in feature_cols:
        values = pl.col(column).cast(pl.Float64)
        std = values.std(ddof=0)
        expressions.append(((values - values.mean()) / pl.when(std == 0).then(1.0).otherwise(std)).alias(column))
    return expressions


def build_composite_data_from_polars(
    dfs: Sequence[pl.DataFrame],
    outcome_cols: Sequence[str],
    loss_weights: Sequence[float],
    outcome_maps: Sequence[OutcomeMap | None] | None = None,
    weight_cols: Sequence[str | None] | None = None,
) -> tuple[dict[str, jnp.ndarray], list[str]]:
    outcome_maps = outcome_maps or [None] * len(dfs)
    weight_cols = weight_cols or [None] * len(dfs)
    competitors = _competitors(dfs)
    competitor_indices = {competitor: index for index, competitor in enumerate(competitors)}
    data = {"loss_weights": jnp.asarray(loss_weights, dtype=jnp.float64)}

    for index, (df, outcome_col, outcome_map, weight_col) in enumerate(
        zip(dfs, outcome_cols, outcome_maps, weight_cols)
    ):
        weight = pl.col(weight_col).cast(pl.Float64) if weight_col is not None else pl.lit(1.0, dtype=pl.Float64)
        grouped = (
            df.select(
                _pair_exprs(competitor_indices)
                + [
                    _outcome_expr(df, outcome_col, outcome_map).alias("_outcome"),
                    weight.alias("_weight"),
                ]
            )
            .group_by("_a", "_b", "_outcome", "_weight")
            .len("_count")
            .with_columns(
                pl.col("_count").cast(pl.Float64),
                (pl.col("_count") * pl.col("_weight")).alias("_opt_weight"),
            )
            .sort("_a", "_b", "_outcome", "_weight")
        )
        data[f"pairs_{index}"] = jnp.asarray(grouped.select("_a", "_b").to_numpy(), dtype=jnp.int32)
        data[f"outcomes_{index}"] = jnp.asarray(
            grouped.get_column("_outcome").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"counts_{index}"] = jnp.asarray(
            grouped.get_column("_count").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"weights_{index}"] = jnp.asarray(
            grouped.get_column("_weight").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"opt_weights_{index}"] = jnp.asarray(
            grouped.get_column("_opt_weight").to_numpy(),
            dtype=jnp.float64,
        )

    return data, competitors


def build_composite_contextual_data_from_polars(
    dfs: Sequence[pl.DataFrame],
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
    competitors = _competitors(dfs)
    competitor_indices = {competitor: index for index, competitor in enumerate(competitors)}
    data = {"loss_weights": jnp.asarray(loss_weights, dtype=jnp.float64)}
    feature_dims: list[int] = []

    for index, (df, outcome_col, feature_cols, outcome_map, weight_col) in enumerate(
        zip(dfs, outcome_cols, feature_cols_per_loss, outcome_maps, weight_cols)
    ):
        weight = pl.col(weight_col).cast(pl.Float64) if weight_col is not None else pl.lit(1.0, dtype=pl.Float64)
        if feature_cols:
            features = _feature_exprs(feature_cols, normalize_features)
            feature_dims.append(len(feature_cols))
        else:
            features = []
            feature_dims.append(0)
        grouped = (
            df.select(
                _pair_exprs(competitor_indices)
                + [
                    _outcome_expr(df, outcome_col, outcome_map).alias("_outcome"),
                    weight.alias("_weight"),
                    *features,
                ]
            )
            .group_by("_a", "_b", "_outcome", "_weight", *(feature_cols or []))
            .len("_count")
            .with_columns(
                pl.col("_count").cast(pl.Float64),
                (pl.col("_count") * pl.col("_weight")).alias("_opt_weight"),
            )
            .sort("_a", "_b", "_outcome", "_weight", *(feature_cols or []))
        )

        data[f"pairs_{index}"] = jnp.asarray(grouped.select("_a", "_b").to_numpy(), dtype=jnp.int32)
        data[f"outcomes_{index}"] = jnp.asarray(
            grouped.get_column("_outcome").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"counts_{index}"] = jnp.asarray(
            grouped.get_column("_count").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"weights_{index}"] = jnp.asarray(
            grouped.get_column("_weight").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"opt_weights_{index}"] = jnp.asarray(
            grouped.get_column("_opt_weight").to_numpy(),
            dtype=jnp.float64,
        )
        data[f"features_{index}"] = (
            jnp.asarray(grouped.select(feature_cols).to_numpy(), dtype=jnp.float64)
            if feature_cols
            else jnp.zeros((grouped.height, 0), dtype=jnp.float64)
        )

    return data, competitors, feature_dims
