from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import pandas as pd
import polars as pl

from arena_rank.preprocessing.common import (
    DEFAULT_OUTCOME_MAP,
    OutcomeMap,
    default_outcome_map,
)
from arena_rank.preprocessing.pandas import get_matchups_and_competitors, get_outcomes
from arena_rank.preprocessing.polars import (
    _competitor_indices,
    _outcome_expr,
)


class BasePairDataset:
    def __init__(
        self,
        competitors: list[str],
        pairs: jnp.ndarray,
        outcomes: jnp.ndarray,
        weights: jnp.ndarray,
    ):
        self.n_competitors = len(competitors)
        self.competitors = competitors
        self.competitor_to_index = {competitor: index for index, competitor in enumerate(competitors)}
        self.n_pairs = len(outcomes)
        self.pairs = pairs
        self.outcomes = outcomes
        self.weights = weights

    def as_dict(self) -> dict[str, jnp.ndarray]:
        raise NotImplementedError


class PairDataset(BasePairDataset):
    def __init__(
        self,
        competitors: list[str],
        pairs: jnp.ndarray,
        outcomes: jnp.ndarray,
        counts: jnp.ndarray,
        weights: jnp.ndarray,
        opt_weights: jnp.ndarray,
        ess_counts: jnp.ndarray | None = None,
    ):
        super().__init__(competitors, pairs, outcomes, weights)
        self.counts = counts
        self.opt_weights = opt_weights
        self.ess_counts = counts if ess_counts is None else ess_counts

    def as_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "pairs": self.pairs,
            "outcomes": self.outcomes,
            "opt_weights": self.opt_weights,
        }

    @staticmethod
    def from_pandas(
        df: pd.DataFrame,
        competitor_cols: Sequence[str] = ("model_a", "model_b"),
        outcome_col: str = "winner",
        outcome_map: OutcomeMap = default_outcome_map,
        reweighted: bool = False,
        min_pair_count: int = 50,
        row_weight_col: str | None = None,
    ) -> PairDataset:
        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        matchups = jnp.asarray(matchups, dtype=jnp.int32)
        outcomes = jnp.asarray(get_outcomes(df, outcome_col, outcome_map), dtype=jnp.float64)
        row_weights = (
            jnp.asarray(df[row_weight_col].values, dtype=jnp.float64)
            if row_weight_col is not None
            else jnp.ones(len(df), dtype=jnp.float64)
        )

        rows = jnp.column_stack([matchups.astype(jnp.float64), outcomes])
        unique_rows, row_indices, counts = jnp.unique(
            rows,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
        counts = counts.astype(jnp.float64)
        sum_weights = jnp.zeros(unique_rows.shape[0], dtype=jnp.float64).at[row_indices].add(row_weights)
        sum_squared_weights = jnp.zeros(unique_rows.shape[0], dtype=jnp.float64).at[row_indices].add(row_weights**2)
        nonzero_weights = sum_squared_weights > 0
        ess_counts = jnp.where(
            nonzero_weights,
            sum_weights**2 / jnp.where(nonzero_weights, sum_squared_weights, 1.0),
            counts,
        )

        pairs = unique_rows[:, :2].astype(jnp.int32)
        unique_pairs, pair_indices = jnp.unique(jnp.sort(pairs, axis=1), return_inverse=True, axis=0)
        pair_sums = jnp.zeros(unique_pairs.shape[0], dtype=jnp.float64).at[pair_indices].add(sum_weights)[pair_indices]
        pair_weights = (
            1.0 / jnp.maximum(pair_sums, min_pair_count) if reweighted else jnp.ones_like(pair_sums, dtype=jnp.float64)
        )

        return PairDataset(
            competitors=competitors,
            pairs=pairs,
            outcomes=unique_rows[:, 2],
            counts=counts,
            weights=(sum_weights / counts) * pair_weights,
            opt_weights=sum_weights * pair_weights,
            ess_counts=ess_counts,
        )

    @staticmethod
    def from_polars(
        df: pl.DataFrame,
        competitor_cols: Sequence[str] = ("model_a", "model_b"),
        outcome_col: str = "winner",
        outcome_map: OutcomeMap = DEFAULT_OUTCOME_MAP,
        reweighted: bool = False,
        min_pair_count: int = 50,
        row_weight_col: str | None = None,
    ) -> PairDataset:
        competitors, competitor_indices = _competitor_indices(df, competitor_cols)
        matchups = jnp.asarray(
            df.select(
                pl.col(competitor_cols[0]).replace_strict(competitor_indices, return_dtype=pl.Int32),
                pl.col(competitor_cols[1]).replace_strict(competitor_indices, return_dtype=pl.Int32),
            ).to_numpy(),
            dtype=jnp.int32,
        )
        outcomes = jnp.asarray(
            df.select(_outcome_expr(df, outcome_col, outcome_map)).to_series().to_numpy(),
            dtype=jnp.float64,
        )
        row_weights = (
            jnp.asarray(df.get_column(row_weight_col).to_numpy(), dtype=jnp.float64)
            if row_weight_col is not None
            else jnp.ones(len(df), dtype=jnp.float64)
        )

        rows = jnp.column_stack([matchups.astype(jnp.float64), outcomes])
        unique_rows, row_indices, counts = jnp.unique(
            rows,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
        counts = counts.astype(jnp.float64)
        sum_weights = jnp.zeros(unique_rows.shape[0], dtype=jnp.float64).at[row_indices].add(row_weights)
        sum_squared_weights = jnp.zeros(unique_rows.shape[0], dtype=jnp.float64).at[row_indices].add(row_weights**2)
        nonzero_weights = sum_squared_weights > 0
        ess_counts = jnp.where(
            nonzero_weights,
            sum_weights**2 / jnp.where(nonzero_weights, sum_squared_weights, 1.0),
            counts,
        )

        pairs = unique_rows[:, :2].astype(jnp.int32)
        unique_pairs, pair_indices = jnp.unique(jnp.sort(pairs, axis=1), return_inverse=True, axis=0)
        pair_sums = jnp.zeros(unique_pairs.shape[0], dtype=jnp.float64).at[pair_indices].add(sum_weights)[pair_indices]
        pair_weights = (
            1.0 / jnp.maximum(pair_sums, min_pair_count) if reweighted else jnp.ones_like(pair_sums, dtype=jnp.float64)
        )

        return PairDataset(
            competitors=competitors,
            pairs=pairs,
            outcomes=unique_rows[:, 2],
            counts=counts,
            weights=(sum_weights / counts) * pair_weights,
            opt_weights=sum_weights * pair_weights,
            ess_counts=ess_counts,
        )


class ContextualPairDataset(BasePairDataset):
    def __init__(
        self,
        competitors: list[str],
        pairs: jnp.ndarray,
        outcomes: jnp.ndarray,
        features: jnp.ndarray,
        weights: jnp.ndarray,
    ):
        super().__init__(competitors, pairs, outcomes, weights)
        self.features = features
        self.n_features = features.shape[1]

    def as_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "pairs": self.pairs,
            "outcomes": self.outcomes,
            "features": self.features,
            "weights": self.weights,
        }

    @staticmethod
    def from_pandas(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        competitor_cols: Sequence[str] = ("model_a", "model_b"),
        outcome_col: str = "winner",
        outcome_map: OutcomeMap = default_outcome_map,
        reweighted: bool = True,
        min_pair_count: int = 50,
        normalize_features: bool = True,
        row_weight_col: str | None = None,
    ) -> ContextualPairDataset:
        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        matchups = jnp.asarray(matchups, dtype=jnp.int32)
        outcomes = jnp.asarray(get_outcomes(df, outcome_col, outcome_map), dtype=jnp.float64)
        features = jnp.asarray(df[list(feature_cols)].values.astype(float), dtype=jnp.float64)
        if normalize_features:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            features = (features - mean) / jnp.where(std == 0, 1.0, std)

        row_weights = (
            jnp.asarray(df[row_weight_col].values, dtype=jnp.float64)
            if row_weight_col is not None
            else jnp.ones(len(outcomes), dtype=jnp.float64)
        )
        if reweighted:
            _, pair_indices = jnp.unique(jnp.sort(matchups, axis=1), return_inverse=True, axis=0)
            pair_sums = jnp.zeros(int(pair_indices.max()) + 1, dtype=jnp.float64).at[pair_indices].add(row_weights)
            pair_weights = 1.0 / jnp.maximum(pair_sums[pair_indices], min_pair_count)
            weights = row_weights * pair_weights
            weights = weights / jnp.mean(weights)
        elif row_weight_col is not None:
            weights = row_weights / jnp.mean(row_weights)
        else:
            weights = jnp.ones(len(outcomes), dtype=jnp.float64)

        return ContextualPairDataset(
            competitors=competitors,
            pairs=matchups,
            outcomes=outcomes,
            features=features,
            weights=weights,
        )

    @staticmethod
    def from_polars(
        df: pl.DataFrame,
        feature_cols: Sequence[str],
        competitor_cols: Sequence[str] = ("model_a", "model_b"),
        outcome_col: str = "winner",
        outcome_map: OutcomeMap = DEFAULT_OUTCOME_MAP,
        reweighted: bool = True,
        min_pair_count: int = 50,
        normalize_features: bool = True,
        row_weight_col: str | None = None,
    ) -> ContextualPairDataset:
        competitors, competitor_indices = _competitor_indices(df, competitor_cols)
        matchups = jnp.asarray(
            df.select(
                pl.col(competitor_cols[0]).replace_strict(competitor_indices, return_dtype=pl.Int32),
                pl.col(competitor_cols[1]).replace_strict(competitor_indices, return_dtype=pl.Int32),
            ).to_numpy(),
            dtype=jnp.int32,
        )
        outcomes = jnp.asarray(
            df.select(_outcome_expr(df, outcome_col, outcome_map)).to_series().to_numpy(),
            dtype=jnp.float64,
        )
        features = jnp.asarray(df.select(feature_cols).to_numpy().astype(float), dtype=jnp.float64)
        if normalize_features:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            features = (features - mean) / jnp.where(std == 0, 1.0, std)

        row_weights = (
            jnp.asarray(df.get_column(row_weight_col).to_numpy(), dtype=jnp.float64)
            if row_weight_col is not None
            else jnp.ones(len(outcomes), dtype=jnp.float64)
        )
        if reweighted:
            _, pair_indices = jnp.unique(jnp.sort(matchups, axis=1), return_inverse=True, axis=0)
            pair_sums = jnp.zeros(int(pair_indices.max()) + 1, dtype=jnp.float64).at[pair_indices].add(row_weights)
            pair_weights = 1.0 / jnp.maximum(pair_sums[pair_indices], min_pair_count)
            weights = row_weights * pair_weights
            weights = weights / jnp.mean(weights)
        elif row_weight_col is not None:
            weights = row_weights / jnp.mean(row_weights)
        else:
            weights = jnp.ones(len(outcomes), dtype=jnp.float64)

        return ContextualPairDataset(
            competitors=competitors,
            pairs=matchups,
            outcomes=outcomes,
            features=features,
            weights=weights,
        )
