"""Compatibility imports for the preprocessing API.

New code should import these names from :mod:`arena_rank.preprocessing`.
"""

from arena_rank.preprocessing import (
    BasePairDataset,
    ContextualPairDataset,
    PairDataset,
    default_outcome_map,
)
from arena_rank.preprocessing.pandas import get_matchups_and_competitors, get_outcomes

__all__ = [
    "BasePairDataset",
    "ContextualPairDataset",
    "PairDataset",
    "default_outcome_map",
    "get_matchups_and_competitors",
    "get_outcomes",
]
