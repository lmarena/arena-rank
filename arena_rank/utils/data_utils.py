from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import jax.numpy as jnp
import pandas as pd


def default_outcome_map(outcome: str) -> float:
    """Default outcome mapping function for standard arena results."""
    mapping = {
        "model_a": 1.0,
        "model_b": 0.0,
        "tie": 0.5,
        "both_bad": 0.5,
    }
    return mapping.get(outcome, 0.5)


def get_outcomes(df, outcome_col, outcome_map: Callable[[str], float], dtype=jnp.float64) -> jnp.ndarray:
    """Maps the str winner column used in lmarena datasets to float outcomes like 1.0, 0.0, 0.5"""
    outcomes = jnp.array(df[outcome_col].map(outcome_map).values, dtype=dtype)
    return outcomes


def get_matchups_and_competitors(df, competitor_cols: list = ["model_a", "model_b"]) -> Tuple[jnp.ndarray, List[str]]:
    """maps the str model_a, model_b columns used in lmarena datasets to integer indices and returns the list of unique competitors"""
    n_rows = len(df)
    competitor_indices, competitors = pd.factorize(
        pd.concat([df[competitor_cols[0]], df[competitor_cols[1]]]), sort=True
    )
    competitor_indices = jnp.array(competitor_indices, dtype=jnp.int32)
    matchups = jnp.column_stack([competitor_indices[:n_rows], competitor_indices[n_rows:]])
    return matchups, competitors.tolist()


class BasePairDataset(ABC):
    """Abstract base class for pairwise comparison datasets."""

    n_pairs: int
    n_competitors: int
    pairs: jnp.ndarray
    outcomes: jnp.ndarray
    weights: jnp.ndarray
    competitors: List[str]
    competitor_to_index: Dict[str, int]

    def __init__(
        self,
        competitors: List[str],
        pairs: jnp.ndarray,
        outcomes: jnp.ndarray,
        weights: jnp.ndarray,
    ):
        self.n_competitors = len(competitors)
        self.competitors = competitors
        self.competitor_to_index = {comp: idx for idx, comp in enumerate(competitors)}
        self.n_pairs = len(outcomes)
        self.pairs = pairs
        self.outcomes = outcomes
        self.weights = weights

    @abstractmethod
    def as_dict(self) -> Dict:
        """Return dataset as dictionary for model consumption."""
        pass

    @staticmethod
    @abstractmethod
    def from_pandas(df, **kwargs):
        """Create dataset from pandas DataFrame."""
        pass


class PairDataset(BasePairDataset):
    """Dataset class for standard Bradley-Terry model. Aggregates rows with the same (A,B,outcome) triplet for efficiency."""

    counts: jnp.ndarray  # shape (n_pairs,)
    opt_weights: jnp.ndarray  # shape (n_pairs,)
    ess_counts: jnp.ndarray  # shape (n_pairs,)

    def __init__(
        self,
        competitors: List[str],
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
        # Kish effective count (sum w)^2 / (sum w^2); equals `counts` when row
        # weights are constant within the aggregate.
        self.ess_counts = counts if ess_counts is None else ess_counts

    def as_dict(self) -> Dict:
        return {
            "pairs": self.pairs,
            "outcomes": self.outcomes,
            "opt_weights": self.opt_weights,
        }

    @staticmethod
    def from_pandas(
        df,
        competitor_cols: list = ["model_a", "model_b"],
        outcome_col: str = "winner",
        outcome_map: Callable[[str], float] = default_outcome_map,
        reweighted: bool = False,
        min_pair_count: int = 50,
        row_weight_col: str | None = None,
    ) -> "PairDataset":
        """Builds a PairDataset from a pandas dataframe.

        Args:
            row_weight_col: Optional column with per-row weights in [0, 1] expressing
                each row's information value. When provided, the effective pair count
                used by the pair-IPW becomes the sum of row weights per pair (not the
                raw row count), so the two reweighting layers compose in the same
                "effective sample size" currency.
        """
        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        outcomes = get_outcomes(df, outcome_col, outcome_map)

        if row_weight_col is not None:
            row_weights = jnp.array(df[row_weight_col].values, dtype=jnp.float64)
        else:
            row_weights = jnp.ones(len(df), dtype=jnp.float64)

        # Aggregate identical (a, b, outcome) triplets, tracking raw counts (for
        # bootstrap), summed weights (loss / IPW), summed squared weights (meat).
        rows = jnp.column_stack([matchups.astype(jnp.float64), outcomes])
        unique_rows, row_to_unique_idx, unique_row_counts = jnp.unique(
            rows, return_inverse=True, return_counts=True, axis=0
        )
        unique_row_counts = unique_row_counts.astype(jnp.float64)
        sum_row_weights = jnp.zeros(unique_rows.shape[0], dtype=jnp.float64).at[row_to_unique_idx].add(row_weights)
        sum_sq_row_weights = (
            jnp.zeros(unique_rows.shape[0], dtype=jnp.float64).at[row_to_unique_idx].add(row_weights**2)
        )
        nonzero_weight_buckets = sum_sq_row_weights > 0
        safe_sum_sq_row_weights = jnp.where(nonzero_weight_buckets, sum_sq_row_weights, 1.0)
        ess_counts = jnp.where(
            nonzero_weight_buckets,
            sum_row_weights**2 / safe_sum_sq_row_weights,
            unique_row_counts,
        )

        unique_matchups = unique_rows[:, :2].astype(jnp.int32)
        unique_outcomes = unique_rows[:, 2]

        # Effective pair count = sum of row weights for the (sorted) pair.
        sorted_unique_matchups = jnp.sort(unique_matchups, axis=1)
        unique_pairs, unique_pair_indices = jnp.unique(sorted_unique_matchups, axis=0, return_inverse=True)
        eff_pair_sums = jnp.zeros(unique_pairs.shape[0], dtype=jnp.float64)
        eff_pair_sums = eff_pair_sums.at[unique_pair_indices].add(sum_row_weights)
        eff_pair_counts = eff_pair_sums[unique_pair_indices]

        if reweighted:
            pair_weights = 1.0 / jnp.maximum(eff_pair_counts, min_pair_count)
        else:
            pair_weights = jnp.ones_like(eff_pair_counts, dtype=jnp.float64)

        return PairDataset(
            competitors=competitors,
            pairs=unique_matchups,
            outcomes=unique_outcomes,
            counts=unique_row_counts,
            weights=(sum_row_weights / unique_row_counts) * pair_weights,
            opt_weights=sum_row_weights * pair_weights,
            ess_counts=ess_counts,
        )


class ContextualPairDataset(BasePairDataset):
    """
    Dataset container for Contextual Bradley-Terry.
    Unlike PairDataset, this does NOT aggregate rows by matchup, because features vary per specific battle.
    """

    features: jnp.ndarray  # shape (n_rows, n_features)

    def __init__(
        self,
        competitors: List[str],
        pairs: jnp.ndarray,
        outcomes: jnp.ndarray,
        features: jnp.ndarray,
        weights: jnp.ndarray,
    ):
        super().__init__(competitors, pairs, outcomes, weights)
        self.features = features
        self.n_features = features.shape[1]

    def as_dict(self) -> Dict:
        return {
            "pairs": self.pairs,
            "outcomes": self.outcomes,
            "features": self.features,
            "weights": self.weights,
        }

    @staticmethod
    def from_pandas(
        df,
        feature_cols: List[str],
        competitor_cols: list = ["model_a", "model_b"],
        outcome_col: str = "winner",
        outcome_map: Callable[[str], float] = default_outcome_map,
        reweighted: bool = True,
        min_pair_count: int = 50,
        normalize_features: bool = True,
        row_weight_col: str | None = None,
    ) -> "ContextualPairDataset":
        """Builds a ContextualPairDataset from a pandas dataframe.

        Args:
            row_weight_col: Optional column with per-row weights in [0, 1] expressing
                each row's information value. The effective pair count used by the
                pair-IPW becomes the sum of row weights per pair, so the two
                reweighting layers compose in the same "effective sample size"
                currency.
        """
        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        outcomes = get_outcomes(df, outcome_col, outcome_map)

        features = jnp.array(df[feature_cols].values.astype(float))
        if normalize_features:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std = jnp.where(std == 0, 1.0, std)
            features = (features - mean) / std

        if row_weight_col is not None:
            row_weights = jnp.array(df[row_weight_col].values, dtype=jnp.float64)
        else:
            row_weights = jnp.ones(len(outcomes), dtype=jnp.float64)

        if reweighted:
            sorted_matchups = jnp.sort(matchups, axis=1)
            _, pair_idx = jnp.unique(sorted_matchups, axis=0, return_inverse=True)
            n_pairs = int(pair_idx.max()) + 1

            # Effective pair count = sum of row weights for the pair.
            eff_pair_sums = jnp.zeros(n_pairs, dtype=jnp.float64).at[pair_idx].add(row_weights)
            eff_pair_counts_per_row = eff_pair_sums[pair_idx]
            pair_weights = 1.0 / jnp.maximum(eff_pair_counts_per_row, min_pair_count)
            weights = row_weights * pair_weights
            weights = weights / jnp.mean(weights)
        elif row_weight_col is not None:
            # Without pair-IPW but with explicit row weights, preserve relative weighting
            # but normalize mean to 1 so the L2 reg term scale is comparable.
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
