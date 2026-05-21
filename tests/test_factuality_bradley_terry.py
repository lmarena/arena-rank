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
