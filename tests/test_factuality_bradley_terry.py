import numpy as np
import pandas as pd

from arena_rank import FactualityBradleyTerry


def test_two_model_dominance():
    rng = np.random.default_rng(0)
    battles = pd.DataFrame(
        {
            "model_a_name": ["A"] * 100 + ["B"] * 100,
            "model_b_name": ["B"] * 100 + ["A"] * 100,
            "score_a": np.concatenate([rng.uniform(0.85, 1.0, 100), rng.uniform(0.0, 0.2, 100)]),
            "score_b": np.concatenate([rng.uniform(0.0, 0.2, 100), rng.uniform(0.85, 1.0, 100)]),
        }
    )

    fit = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=50, bootstrap_seed=42).fit(battles)

    ratings = fit.ratings_df.set_index("model_name")
    assert ratings.loc["A", "rating"] > ratings.loc["B", "rating"]
    assert ratings.loc["A", "rating"] - ratings.loc["B", "rating"] > 200  # well separated
    assert ratings.loc["A", "rating_upper"] >= ratings.loc["A", "rating"]
    assert ratings.loc["A", "rating_lower"] <= ratings.loc["A", "rating"]


def test_three_model_transitive_ordering():
    rng = np.random.default_rng(0)

    def battle_rows(model_a, model_b, lean):
        a = rng.uniform(lean, lean + 0.1, 60)
        b = rng.uniform(1.0 - lean - 0.1, 1.0 - lean, 60)
        return pd.DataFrame(
            {
                "model_a_name": [model_a] * 60,
                "model_b_name": [model_b] * 60,
                "score_a": a,
                "score_b": b,
            }
        )

    battles = pd.concat(
        [
            battle_rows("A", "B", 0.8),  # A beats B
            battle_rows("B", "C", 0.7),  # B beats C
            battle_rows("A", "C", 0.85),  # A beats C
        ],
        ignore_index=True,
    )

    fit = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=20, bootstrap_seed=1).fit(battles)

    ratings = fit.ratings_df.set_index("model_name")
    assert ratings.loc["A", "rating"] > ratings.loc["B", "rating"] > ratings.loc["C", "rating"]


def test_temperature_scaling_widens_separation():
    rng = np.random.default_rng(0)
    battles = pd.DataFrame(
        {
            "model_a_name": ["A"] * 200,
            "model_b_name": ["B"] * 200,
            "score_a": rng.uniform(0.6, 0.7, 200),
            "score_b": rng.uniform(0.4, 0.5, 200),
        }
    )

    fit_hot = (
        FactualityBradleyTerry(temperature=0.5, bootstrap_iterations=20, bootstrap_seed=1)
        .fit(battles)
        .ratings_df.set_index("model_name")
    )
    fit_cold = (
        FactualityBradleyTerry(temperature=0.05, bootstrap_iterations=20, bootstrap_seed=1)
        .fit(battles)
        .ratings_df.set_index("model_name")
    )

    sep_hot = fit_hot.loc["A", "rating"] - fit_hot.loc["B", "rating"]
    sep_cold = fit_cold.loc["A", "rating"] - fit_cold.loc["B", "rating"]
    assert sep_cold > sep_hot * 1.5


def test_null_imputation_does_not_crash():
    battles = pd.DataFrame(
        {
            "model_a_name": ["A", "A", "B"],
            "model_b_name": ["B", "B", "A"],
            "score_a": [0.9, np.nan, 0.1],
            "score_b": [0.2, 0.3, np.nan],
        }
    )

    fit = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=10, bootstrap_seed=0).fit(battles)

    assert set(fit.ratings_df["model_name"]) == {"A", "B"}
    assert fit.ratings_df["rating"].notna().all()


def test_drop_imputation_removes_null_battles():
    battles = pd.DataFrame(
        {
            "model_a_name": ["A", "A", "A"],
            "model_b_name": ["B", "B", "B"],
            "score_a": [0.9, np.nan, 0.85],
            "score_b": [0.2, 0.3, 0.15],
        }
    )

    results = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=5, imputation="drop", bootstrap_seed=0).fit(
        battles
    )

    assert results.expected_outcomes.set_index("model_name").loc["A", "battle_count"] == 2


def test_fit_exposes_scipy_metadata():
    battles = pd.DataFrame(
        {
            "model_a_name": ["m1", "m2", "m1"],
            "model_b_name": ["m2", "m3", "m3"],
            "score_a": [0.8, 0.7, 0.9],
            "score_b": [0.2, 0.3, 0.5],
        }
    )
    result = FactualityBradleyTerry(temperature=0.1).fit(battles)

    assert isinstance(result.fit_success, bool)
    assert isinstance(result.fit_message, str)
    assert isinstance(result.fit_iterations, int) and result.fit_iterations > 0
    assert isinstance(result.fit_loss, float) and result.fit_loss >= 0.0


def test_doubling_weight_matches_duplicating_row():
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "model_a_name": ["A"] * 50 + ["B"] * 50,
            "model_b_name": ["B"] * 50 + ["A"] * 50,
            "score_a": np.concatenate([rng.uniform(0.7, 0.9, 50), rng.uniform(0.1, 0.3, 50)]),
            "score_b": np.concatenate([rng.uniform(0.1, 0.3, 50), rng.uniform(0.7, 0.9, 50)]),
        }
    )

    weighted = base.copy()
    weighted["sample_weight"] = 1.0
    weighted.iloc[0, weighted.columns.get_loc("sample_weight")] = 2.0

    duplicated = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    duplicated["sample_weight"] = 1.0

    fit_w = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=0).fit(weighted)
    fit_d = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=0).fit(duplicated)

    r_w = fit_w.ratings_df.set_index("model_name")["rating"]
    r_d = fit_d.ratings_df.set_index("model_name")["rating"]
    assert abs(r_w["A"] - r_d["A"]) < 1e-4
    assert abs(r_w["B"] - r_d["B"]) < 1e-4


def test_ridge_shrinks_theta():
    rng = np.random.default_rng(0)
    battles = pd.DataFrame(
        {
            "model_a_name": ["A"] * 100 + ["B"] * 100,
            "model_b_name": ["B"] * 100 + ["A"] * 100,
            "score_a": np.concatenate([rng.uniform(0.85, 1.0, 100), rng.uniform(0.0, 0.2, 100)]),
            "score_b": np.concatenate([rng.uniform(0.0, 0.2, 100), rng.uniform(0.85, 1.0, 100)]),
        }
    )

    no_ridge = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=0, ridge_lambda=0.0).fit(battles)
    with_ridge = FactualityBradleyTerry(temperature=0.1, bootstrap_iterations=0, ridge_lambda=1.0).fit(battles)

    theta_norm_no_ridge = float(np.linalg.norm(no_ridge.ratings_df["theta"].to_numpy()))
    theta_norm_with_ridge = float(np.linalg.norm(with_ridge.ratings_df["theta"].to_numpy()))
    assert theta_norm_with_ridge < theta_norm_no_ridge
    # half-lambda convention: ridge=1.0 still shrinks, just less aggressively


def test_closed_form_ci_matches_bootstrap_within_tolerance():
    rng = np.random.default_rng(0)
    n = 600
    a_models = rng.choice(["A", "B", "C", "D", "E"], size=n)
    b_models = rng.choice(["A", "B", "C", "D", "E"], size=n)
    keep = a_models != b_models
    a_models = a_models[keep]
    b_models = b_models[keep]
    score_a = rng.uniform(0.0, 1.0, size=len(a_models))
    score_b = rng.uniform(0.0, 1.0, size=len(b_models))
    battles = pd.DataFrame({"model_a_name": a_models, "model_b_name": b_models, "score_a": score_a, "score_b": score_b})

    boot = FactualityBradleyTerry(
        temperature=0.1, bootstrap_iterations=2000, bootstrap_seed=1, ridge_lambda=1e-5, method="bootstrap"
    ).fit(battles)
    closed = FactualityBradleyTerry(
        temperature=0.1, bootstrap_iterations=0, ridge_lambda=1e-5, method="closed_form"
    ).fit(battles)

    pd.testing.assert_series_equal(
        boot.ratings_df.set_index("model_name")["rating"],
        closed.ratings_df.set_index("model_name")["rating"],
        check_exact=False,
        atol=1.0,
        check_names=False,
    )

    boot_var = boot.ratings_df.set_index("model_name")["variance"]
    closed_var = closed.ratings_df.set_index("model_name")["variance"]
    # Sandwich CIs and bootstrap are statistically equivalent; allow a 50% band
    # to absorb finite-sample noise in 2000-iter bootstrap.
    ratio = closed_var / boot_var
    assert (ratio.between(0.5, 1.5)).all(), f"variance ratio out of range: {ratio.to_dict()}"


def test_aggregation_compresses_duplicates():
    # 100 identical battles aggregate to 1 row with count=100. battle_count uses
    # the raw (pre-aggregation) row counts, so each model still shows 100 battles.
    battles = pd.DataFrame(
        {
            "model_a_name": ["A"] * 100,
            "model_b_name": ["B"] * 100,
            "score_a": [0.9] * 100,
            "score_b": [0.1] * 100,
        }
    )

    fit_agg = FactualityBradleyTerry(temperature=0.1, ridge_lambda=1e-5).fit(battles)

    assert set(fit_agg.ratings_df["model_name"]) == {"A", "B"}
    assert np.isfinite(fit_agg.ratings_df["rating"]).all()
    outcomes = fit_agg.expected_outcomes.set_index("model_name")
    assert outcomes.loc["A", "battle_count"] == 100
    assert outcomes.loc["B", "battle_count"] == 100


def test_anchor_offset_preserves_variance():
    rng = np.random.default_rng(0)
    battles = pd.DataFrame(
        {
            "model_a_name": ["A"] * 50 + ["B"] * 50 + ["A"] * 50,
            "model_b_name": ["B"] * 50 + ["C"] * 50 + ["C"] * 50,
            "score_a": rng.uniform(0.6, 0.9, 150),
            "score_b": rng.uniform(0.1, 0.4, 150),
        }
    )

    fit = FactualityBradleyTerry(
        temperature=0.1,
        ridge_lambda=1e-5,
        method="closed_form",
        anchor_model="B",
        anchor_rating=1500.0,
    ).fit(battles)

    ratings = fit.ratings_df.set_index("model_name")

    # Anchor's rating equals anchor_rating exactly (post-fit additive offset).
    assert abs(ratings.loc["B", "rating"] - 1500.0) < 1e-6

    # Anchor's variance is non-zero — sandwich-derived, not gauge-fixed.
    assert ratings.loc["B", "variance"] > 0.0

    # CI bracket around anchor is non-degenerate.
    assert ratings.loc["B", "rating_lower"] < 1500.0 < ratings.loc["B", "rating_upper"]
