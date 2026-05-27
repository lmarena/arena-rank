import numpy as np
import pandas as pd

from arena_rank import (
    DEFAULT_OUTCOME_MAP,
    CompositeBradleyTerry,
    build_composite_data,
)


def _make_human_pref_df(rng, models, n_per_pair=80):
    """Builds a small human-preference DataFrame with deterministic winners."""
    rows = []
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    for a, b in pairs:
        for _ in range(n_per_pair):
            # First model in alphabetical order is the stronger one.
            winner = "model_a" if rng.random() < 0.75 else "model_b"
            rows.append({"model_a": a, "model_b": b, "winner": winner})
    return pd.DataFrame(rows)


def _make_fact_df(rng, models, n_per_pair=80):
    """Builds a small factuality DataFrame with sigmoid-style outcomes."""
    rows = []
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    for a, b in pairs:
        for _ in range(n_per_pair):
            # First model has higher factuality probability.
            outcome = float(rng.uniform(0.6, 0.8))
            rows.append({"model_a": a, "model_b": b, "outcome": outcome})
    return pd.DataFrame(rows)


def test_two_model_dominance():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "model_a": ["A"] * 200,
            "model_b": ["B"] * 200,
            "winner": rng.choice(["model_a", "model_b"], size=200, p=[0.85, 0.15]),
        }
    )

    data_dict, competitors = build_composite_data(
        dfs=[df],
        outcome_cols=["winner"],
        loss_weights=[1.0],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get],
    )
    fit = CompositeBradleyTerry(n_competitors=len(competitors), n_losses=1).fit(data_dict)
    result = fit.compute_ratings_and_cis(data_dict, competitors)

    ratings = dict(zip(result["competitors"], np.asarray(result["ratings"])))
    assert ratings["A"] - ratings["B"] > 200


def test_endpoint_w0_equals_human_only_fit():
    """At w=[1.0, 0.0], a 2-component fit must match a single-component human fit."""
    rng = np.random.default_rng(42)
    models = ["A", "B", "C"]
    human_df = _make_human_pref_df(rng, models)
    fact_df = _make_fact_df(rng, models)

    # Single-component human-only fit
    data1, comp1 = build_composite_data(
        dfs=[human_df],
        outcome_cols=["winner"],
        loss_weights=[1.0],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get],
    )
    single = CompositeBradleyTerry(n_competitors=len(comp1), n_losses=1).fit(data1)
    single_ratings = dict(zip(comp1, np.asarray(single.compute_ratings_and_cis(data1, comp1)["ratings"])))

    # 2-component fit with all weight on human
    data2, comp2 = build_composite_data(
        dfs=[human_df, fact_df],
        outcome_cols=["winner", "outcome"],
        loss_weights=[1.0, 0.0],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, None],
    )
    composite = CompositeBradleyTerry(n_competitors=len(comp2), n_losses=2).fit(data2)
    composite_ratings = dict(zip(comp2, np.asarray(composite.compute_ratings_and_cis(data2, comp2)["ratings"])))

    for m in comp1:
        assert abs(single_ratings[m] - composite_ratings[m]) < 1e-2, m


def test_endpoint_w1_equals_factuality_only_fit():
    """At w=[0.0, 1.0], a 2-component fit must match a single-component factuality fit."""
    rng = np.random.default_rng(7)
    models = ["A", "B", "C"]
    human_df = _make_human_pref_df(rng, models)
    fact_df = _make_fact_df(rng, models)

    data1, comp1 = build_composite_data(
        dfs=[fact_df],
        outcome_cols=["outcome"],
        loss_weights=[1.0],
    )
    single = CompositeBradleyTerry(n_competitors=len(comp1), n_losses=1).fit(data1)
    single_ratings = dict(zip(comp1, np.asarray(single.compute_ratings_and_cis(data1, comp1)["ratings"])))

    data2, comp2 = build_composite_data(
        dfs=[human_df, fact_df],
        outcome_cols=["winner", "outcome"],
        loss_weights=[0.0, 1.0],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, None],
    )
    composite = CompositeBradleyTerry(n_competitors=len(comp2), n_losses=2).fit(data2)
    composite_ratings = dict(zip(comp2, np.asarray(composite.compute_ratings_and_cis(data2, comp2)["ratings"])))

    for m in comp1:
        assert abs(single_ratings[m] - composite_ratings[m]) < 1e-2, m


def test_compute_ratings_and_cis_returns_finite_cis():
    rng = np.random.default_rng(3)
    models = ["A", "B", "C"]
    human_df = _make_human_pref_df(rng, models)
    fact_df = _make_fact_df(rng, models)

    data, comp = build_composite_data(
        dfs=[human_df, fact_df],
        outcome_cols=["winner", "outcome"],
        loss_weights=[0.5, 0.5],
        outcome_maps=[DEFAULT_OUTCOME_MAP.get, None],
    )
    fit = CompositeBradleyTerry(n_competitors=len(comp), n_losses=2).fit(data)
    result = fit.compute_ratings_and_cis(data, comp)

    assert np.isfinite(np.asarray(result["ratings"])).all()
    assert np.isfinite(np.asarray(result["rating_lower"])).all()
    assert np.isfinite(np.asarray(result["rating_upper"])).all()
    lower = np.asarray(result["rating_lower"])
    upper = np.asarray(result["rating_upper"])
    assert (upper >= lower).all()
