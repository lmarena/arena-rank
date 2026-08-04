from collections.abc import Callable, Mapping
from typing import Any

OutcomeMap = Callable[[Any], float] | Mapping[Any, float]

DEFAULT_OUTCOME_MAP: dict[str, float] = {
    "model_a": 1.0,
    "model_b": 0.0,
    "tie": 0.5,
    "both_bad": 0.5,
}


def default_outcome_map(outcome: str) -> float:
    return DEFAULT_OUTCOME_MAP.get(outcome, 0.5)
