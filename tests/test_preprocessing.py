import numpy as np
import pandas as pd
import polars as pl

from arena_rank.preprocessing import ContextualPairDataset, PairDataset


def test_pandas_and_polars_pair_builders_match():
    pdf = pd.DataFrame(
        {
            "model_a": ["B", "A", "A", "C"],
            "model_b": ["A", "B", "C", "A"],
            "winner": ["model_a", "tie", "model_b", "both_bad"],
            "row_weight": [1.0, 0.5, 0.25, 1.0],
            "feature_0": [0.0, 1.0, 0.0, 1.0],
            "feature_1": [1.0, 0.0, -1.0, 0.0],
        }
    )
    pldf = pl.from_pandas(pdf)

    pandas_pair = PairDataset.from_pandas(pdf, row_weight_col="row_weight", reweighted=True)
    polars_pair = PairDataset.from_polars(pldf, row_weight_col="row_weight", reweighted=True)
    assert pandas_pair.competitors == polars_pair.competitors
    for name in (
        "pairs",
        "outcomes",
        "counts",
        "weights",
        "opt_weights",
        "ess_counts",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(pandas_pair, name)),
            np.asarray(getattr(polars_pair, name)),
        )

    pandas_contextual = ContextualPairDataset.from_pandas(
        pdf,
        feature_cols=["feature_0", "feature_1"],
        row_weight_col="row_weight",
    )
    polars_contextual = ContextualPairDataset.from_polars(
        pldf,
        feature_cols=["feature_0", "feature_1"],
        row_weight_col="row_weight",
    )
    assert pandas_contextual.competitors == polars_contextual.competitors
    for name in ("pairs", "outcomes", "features", "weights"):
        np.testing.assert_array_equal(
            np.asarray(getattr(pandas_contextual, name)),
            np.asarray(getattr(polars_contextual, name)),
        )
