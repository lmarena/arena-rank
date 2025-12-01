import pandas as pd
import jax.numpy as jnp
from typing import Dict, Tuple, List


def get_outcomes(df, outcome_col, outcome_map, dtype=jnp.float64) -> jnp.ndarray:
    """maps the str winner column used in lmarena datasets to float outcomes like 1.0, 0.0, 0.5"""
    outcomes = jnp.empty(len(df), dtype=dtype)
    for outcome_str, outcome_val in outcome_map.items():
        cond = jnp.array(df[outcome_col] == outcome_str)
        outcomes = jnp.where(cond, outcome_val, outcomes)
    return outcomes


def get_matchups_and_competitors(df, cometator_cols: list = ["model_a", "model_b"]) -> Tuple[jnp.ndarray, List[str]]:
    """maps the str model_a, model_b columns used in lmarena datasets to integer indices and returns the list of unique models"""
    n_rows = len(df)
    competitor_indices, competitors = pd.factorize(pd.concat([df[cometator_cols[0]], df[cometator_cols[1]]]))
    competitor_indices = jnp.array(competitor_indices, dtype=jnp.int32)
    matchups = jnp.column_stack([competitor_indices[:n_rows], competitor_indices[n_rows:]])
    return matchups, competitors.tolist()


class PairDataset:
    n_pairs: int
    n_competitors: int
    pairs: jnp.ndarray  # shape (n_pairs, 2) integer indices of competitors
    weights: jnp.ndarray
    outcomes: jnp.ndarray  # shape (n_pairs,)
    counts: jnp.ndarray  # shape (n_pairs,)
    opt_weights: jnp.ndarray  # shape (n_pairs,)

    def __init__(
        self,
        competitors: List[str],
        pairs: jnp.ndarray,
        outcomes: jnp.ndarray,
        counts: jnp.ndarray,
        weights: jnp.ndarray,
        opt_weights: jnp.ndarray,
    ):
        self.n_competitors = len(competitors)
        self.competitors = competitors
        self.pairs = pairs
        self.outcomes = outcomes
        self.counts = counts
        self.weights = weights
        self.opt_weights = opt_weights

    @staticmethod
    def from_pandas(
        df,
        competitor_cols: list = ["model_a", "model_b"],
        outcome_col: str = "winner",
        outcome_map: Dict[str, float] = {
            "model_a": 1.0,
            "model_b": 0.0,
            "tie": 0.5,
            "both_bad": 0.5,
        },
        reweighted: bool = False,
        min_pair_count: int = 50,
    ) -> "PairDataset":
        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        outcomes = get_outcomes(df, outcome_col, outcome_map)

        rows = jnp.column_stack([matchups.astype(jnp.float64), outcomes])
        unique_rows, unique_row_counts = jnp.unique(rows, return_counts=True, axis=0)

        unique_matchups = unique_rows[:, :2].astype(jnp.int32)
        unique_outcomes = unique_rows[:, 2]
        unique_row_counts = unique_row_counts.astype(jnp.float64)

        sorted_unique_matchups = jnp.sort(
            unique_matchups, axis=1
        )  # sort each row so (a,b) and (b,a) are treated the same
        unique_pairs, unique_pair_indices = jnp.unique(sorted_unique_matchups, axis=0, return_inverse=True)

        unique_pair_sums = jnp.zeros(unique_pairs.shape[0], dtype=jnp.float64)
        unique_pair_sums = unique_pair_sums.at[unique_pair_indices].add(unique_row_counts)
        pair_counts = unique_pair_sums[unique_pair_indices]

        if reweighted:
            weights = 1.0 / jnp.maximum(pair_counts, min_pair_count)
        else:
            weights = jnp.ones_like(pair_counts, dtype=jnp.float64)

        return PairDataset(
            competitors=competitors,
            pairs=unique_matchups,
            outcomes=unique_outcomes,
            counts=unique_row_counts,
            weights=weights,
            opt_weights=weights * unique_row_counts,
        )


class ContextualPairDataset(PairDataset):
    """
    Dataset container for Contextual Bradley-Terry.
    Unlike PairDataset, this does NOT aggregate rows by matchup,
    because features vary per specific battle.
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
        self.n_competitors = len(competitors)
        self.competitors = competitors
        self.n_pairs = len(outcomes)
        self.pairs = pairs
        self.outcomes = outcomes
        self.n_features = features.shape[1]
        self.features = features
        self.weights = weights

    @staticmethod
    def from_pandas(
        df,
        feature_cols: List[str],
        competitor_cols: list = ["model_a", "model_b"],
        outcome_col: str = "winner",
        outcome_map: Dict[str, float] = {
            "model_a": 1.0,
            "model_b": 0.0,
            "tie": 0.5,
            "both_bad": 0.5,
        },
        reweighted: bool = True,
        min_pair_count: int = 50,
    ) -> "ContextualPairDataset":
        # 1. Get Matchups and Outcomes (Raw, un-aggregated)
        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        outcomes = get_outcomes(df, outcome_col, outcome_map)

        # 2. Process Features (Normalize)
        # We use pandas/numpy for the initial stats calculation to avoid loading huge raw data to GPU memory immediately
        features_raw = df[feature_cols].values.astype(float)
        mean = features_raw.mean(axis=0)
        std = features_raw.std(axis=0)
        # Avoid division by zero
        std = jnp.where(std == 0, 1.0, std)
        features_norm = (features_raw - mean) / std
        features = jnp.array(features_norm, dtype=jnp.float64)

        # 3. Calculate Weights (The tricky part)
        # Even though we don't aggregate rows, we need to know how common a pair is
        # to apply the inverse-frequency weighting (if reweighted=True).

        if reweighted:
            # Sort matchups so (A,B) and (B,A) are identified as the same pair
            sorted_matchups = jnp.sort(matchups, axis=1)

            # Find unique pairs and the inverse indices mapping rows to those pairs
            _, pair_idx, pair_counts = jnp.unique(sorted_matchups, axis=0, return_inverse=True, return_counts=True)

            # Map the total count of the pair back to the individual row
            # If row 0 is A vs B, and A vs B happens 100 times total, row_counts[0] = 100
            row_pair_counts = pair_counts[pair_idx].astype(jnp.float64)

            # Apply weight formula: 1 / max(count, min_threshold)
            weights = 1.0 / jnp.maximum(row_pair_counts, min_pair_count)

            # Normalize weights so the mean weight is 1.0 (maintains loss scale)
            weights = weights / jnp.mean(weights)
        else:
            weights = jnp.ones(len(outcomes), dtype=jnp.float64)

        return ContextualPairDataset(
            competitors=competitors,
            pairs=matchups,
            outcomes=outcomes,
            features=features,
            weights=weights,
        )
