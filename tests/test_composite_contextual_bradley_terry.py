import numpy as np
import pandas as pd
import polars as pl

from arena_rank import (
    DEFAULT_OUTCOME_MAP,
    CompositeBradleyTerry,
    CompositeContextualBradleyTerry,
    build_composite_contextual_data,
    build_composite_contextual_data_from_polars,
    build_composite_data,
)
from arena_rank.models.contextual_bradley_terry import ContextualBradleyTerry
from arena_rank.preprocessing import ContextualPairDataset


def _make_contextual_df(seed, n_competitors=20, n_matches=4000, n_features=3):
    """Builds a battle DataFrame whose outcomes follow r_a - r_b + f^T beta."""
    rng = np.random.default_rng(seed)
    strengths = rng.normal(size=n_competitors)
    coeffs = rng.normal(size=n_features)

    idx_a = rng.integers(0, n_competitors, size=n_matches)
    offsets = rng.integers(1, n_competitors, size=n_matches)
    idx_b = (idx_a + offsets) % n_competitors
    features = rng.normal(size=(n_matches, n_features))

    logits = strengths[idx_a] - strengths[idx_b] + features @ coeffs
    a_wins = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    df = pd.DataFrame(
        {
            "model_a": idx_a.astype(str),
            "model_b": idx_b.astype(str),
            "winner": np.where(a_wins == 1, "model_a", "model_b"),
            **{f"feature_{i}": features[:, i] for i in range(n_features)},
        }
    )
    return df, strengths, coeffs


def _make_plain_df(seed, models, n_per_pair=120):
    """Builds a plain (feature-less) battle DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for _ in range(n_per_pair):
                # earlier-in-list models are stronger
                winner = "model_a" if rng.random() < 0.7 else "model_b"
                rows.append({"model_a": a, "model_b": b, "winner": winner})
    return pd.DataFrame(rows)


def test_polars_contextual_builder_matches_pandas_builder():
    pdf = pd.DataFrame(
        {
            "model_a": ["A", "A", "B"],
            "model_b": ["B", "C", "C"],
            "winner": ["model_a", "tie", "model_b"],
            "sample_weight": [1.0, 0.5, 0.25],
            "feature_0": [1.0, 0.0, -1.0],
        }
    )
    kwargs = {
        "outcome_cols": ["winner"],
        "loss_weights": [1.0],
        "feature_cols_per_loss": [["feature_0"]],
        "outcome_maps": [DEFAULT_OUTCOME_MAP],
        "weight_cols": ["sample_weight"],
    }

    pandas_data, pandas_competitors, pandas_dims = build_composite_contextual_data(dfs=[pdf], **kwargs)
    polars_data, polars_competitors, polars_dims = build_composite_contextual_data_from_polars(
        dfs=[pl.from_pandas(pdf)], **kwargs
    )

    assert pandas_competitors == polars_competitors
    assert pandas_dims == polars_dims
    for key in pandas_data:
        np.testing.assert_allclose(pandas_data[key], polars_data[key])


def test_polars_contextual_builder_aggregates_exact_duplicates():
    df = pl.DataFrame(
        {
            "model_a": ["A", "A", "A", "B"],
            "model_b": ["B", "B", "B", "A"],
            "outcome": [1.0, 1.0, 1.0, 1.0],
            "sample_weight": [0.5, 0.5, 0.5, 1.0],
            "feature": [1.0, 1.0, 1.0, -1.0],
        }
    )
    data, _, _ = build_composite_contextual_data_from_polars(
        dfs=[df],
        outcome_cols=["outcome"],
        loss_weights=[1.0],
        feature_cols_per_loss=[["feature"]],
        weight_cols=["sample_weight"],
        normalize_features=False,
    )

    assert data["pairs_0"].shape == (2, 2)
    np.testing.assert_array_equal(data["counts_0"], [3.0, 1.0])
    np.testing.assert_array_equal(data["weights_0"], [0.5, 1.0])
    np.testing.assert_array_equal(data["opt_weights_0"], [1.5, 1.0])


def test_single_contextual_component_matches_standalone_contextual():
    """A 1-component (unweighted) contextual composite fit must match a
    standalone ContextualBradleyTerry fit on the same data."""
    df, _, _ = _make_contextual_df(seed=0)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    data, competitors, feature_dims = build_composite_contextual_data(
        dfs=[df],
        outcome_cols=["winner"],
        loss_weights=[1.0],
        feature_cols_per_loss=[feature_cols],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get],
    )
    composite = CompositeContextualBradleyTerry(
        n_competitors=len(competitors), n_features_per_loss=feature_dims, reg=1.0
    )
    composite_res = composite.compute_ratings_and_cis(data, competitors)
    composite_ratings = dict(zip(competitors, np.asarray(composite_res["ratings"])))

    ds = ContextualPairDataset.from_pandas(
        df, feature_cols=feature_cols, outcome_map=DEFAULT_OUTCOME_MAP.get, reweighted=False, normalize_features=True
    )
    standalone = ContextualBradleyTerry(n_competitors=len(ds.competitors), n_features=len(feature_cols), reg=1.0)
    standalone_res = standalone.compute_ratings_and_cis(ds)
    standalone_ratings = dict(zip(ds.competitors, np.asarray(standalone_res["ratings"])))

    for m in competitors:
        assert abs(composite_ratings[m] - standalone_ratings[m]) < 1.0, m
    np.testing.assert_allclose(np.asarray(composite_res["coeffs"][0]), np.asarray(standalone_res["coeffs"]), atol=1e-2)


def test_recovers_known_coeffs_and_ordering():
    """The single-component fit recovers the planted feature coefficients and
    relative rating ordering."""
    df, strengths, true_coeffs = _make_contextual_df(seed=3)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    data, competitors, feature_dims = build_composite_contextual_data(
        dfs=[df],
        outcome_cols=["winner"],
        loss_weights=[1.0],
        feature_cols_per_loss=[feature_cols],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get],
    )
    model = CompositeContextualBradleyTerry(
        n_competitors=len(competitors),
        n_features_per_loss=feature_dims,
        reg=1e-3,
        init_rating=0.0,
        scale=1.0,
        base=np.e,
    )
    res = model.compute_ratings_and_cis(data, competitors)

    # features were not normalized in the generative model (std ~1), so coeffs
    # should land close to the planted values.
    np.testing.assert_allclose(np.asarray(res["coeffs"][0]), true_coeffs, atol=0.15)

    # competitors are sorted as strings, so realign fitted ratings to the
    # numeric index of the generative strengths before correlating.
    fitted = np.asarray(res["ratings"])
    aligned_strengths = np.array([strengths[int(m)] for m in competitors])
    assert np.corrcoef(fitted, aligned_strengths)[0, 1] > 0.95


def test_contextual_plus_plain_component():
    """A contextual component can be jointly fit with a feature-less plain BT
    component sharing the same ratings."""
    ctx_df, _, _ = _make_contextual_df(seed=5, n_competitors=8, n_matches=3000)
    feature_cols = [c for c in ctx_df.columns if c.startswith("feature_")]
    plain_df = _make_plain_df(seed=5, models=[str(i) for i in range(8)])

    data, competitors, feature_dims = build_composite_contextual_data(
        dfs=[ctx_df, plain_df],
        outcome_cols=["winner", "winner"],
        loss_weights=[0.5, 0.5],
        feature_cols_per_loss=[feature_cols, None],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, DEFAULT_OUTCOME_MAP.get],
    )

    assert feature_dims == [len(feature_cols), 0]

    model = CompositeContextualBradleyTerry(n_competitors=len(competitors), n_features_per_loss=feature_dims)
    res = model.compute_ratings_and_cis(data, competitors)

    assert len(res["coeffs"][0]) == len(feature_cols)
    assert len(res["coeffs"][1]) == 0
    assert np.isfinite(np.asarray(res["ratings"])).all()
    assert np.isfinite(np.asarray(res["rating_lower"])).all()
    assert (np.asarray(res["rating_upper"]) >= np.asarray(res["rating_lower"])).all()


def test_all_plain_components_match_composite_bradley_terry():
    """With no features anywhere, the composite-contextual point estimates must
    match CompositeBradleyTerry (which aggregates) on the same data."""
    models = ["A", "B", "C", "D"]
    human_df = _make_plain_df(seed=1, models=models)
    fact_df = _make_plain_df(seed=2, models=models)

    cc_data, cc_comp, cc_dims = build_composite_contextual_data(
        dfs=[human_df, fact_df],
        outcome_cols=["winner", "winner"],
        loss_weights=[0.5, 0.5],
        feature_cols_per_loss=[None, None],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, DEFAULT_OUTCOME_MAP.get],
    )
    cc = CompositeContextualBradleyTerry(n_competitors=len(cc_comp), n_features_per_loss=cc_dims)
    cc_ratings = dict(zip(cc_comp, np.asarray(cc.compute_ratings_and_cis(cc_data, cc_comp)["ratings"])))

    c_data, c_comp = build_composite_data(
        dfs=[human_df, fact_df],
        outcome_cols=["winner", "winner"],
        loss_weights=[0.5, 0.5],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, DEFAULT_OUTCOME_MAP.get],
    )
    c = CompositeBradleyTerry(n_competitors=len(c_comp), n_losses=2)
    c_ratings = dict(zip(c_comp, np.asarray(c.compute_ratings_and_cis(c_data, c_comp)["ratings"])))

    for m in models:
        assert abs(cc_ratings[m] - c_ratings[m]) < 1e-2, m


def test_endpoint_weight_zero_drops_component():
    """At loss_weights=[1.0, 0.0], the 2-component fit must match a 1-component
    fit on the first (contextual) component alone."""
    ctx_df, _, _ = _make_contextual_df(seed=7, n_competitors=6, n_matches=2500)
    feature_cols = [c for c in ctx_df.columns if c.startswith("feature_")]
    plain_df = _make_plain_df(seed=7, models=[str(i) for i in range(6)])

    single_data, single_comp, single_dims = build_composite_contextual_data(
        dfs=[ctx_df],
        outcome_cols=["winner"],
        loss_weights=[1.0],
        feature_cols_per_loss=[feature_cols],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get],
    )
    single = CompositeContextualBradleyTerry(n_competitors=len(single_comp), n_features_per_loss=single_dims)
    single_ratings = dict(
        zip(single_comp, np.asarray(single.compute_ratings_and_cis(single_data, single_comp)["ratings"]))
    )

    both_data, both_comp, both_dims = build_composite_contextual_data(
        dfs=[ctx_df, plain_df],
        outcome_cols=["winner", "winner"],
        loss_weights=[1.0, 0.0],
        feature_cols_per_loss=[feature_cols, None],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, DEFAULT_OUTCOME_MAP.get],
    )
    both = CompositeContextualBradleyTerry(n_competitors=len(both_comp), n_features_per_loss=both_dims)
    both_ratings = dict(zip(both_comp, np.asarray(both.compute_ratings_and_cis(both_data, both_comp)["ratings"])))

    for m in single_comp:
        assert abs(single_ratings[m] - both_ratings[m]) < 1e-2, m
