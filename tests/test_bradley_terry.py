import random
import numpy as np
import pandas as pd
from arena.models.bradley_terry import BradleyTerry
from arena.models.contextual_bradley_terry import ContextualBradleyTerry
from arena.utils.data_utils import PairDataset, ContextualPairDataset


def test_wikipedia_example():
    """
    Recreates the 'Worked example of solution procedure' from the Wikipedia article.
    https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

    Data (22 games total):
    - A vs B: A wins 2, B wins 3
    - A vs D: A wins 1, D wins 4
    - B vs C: B wins 5, C wins 3
    - C vs D: C wins 1, D wins 3
    - (A vs C and B vs D are not played)

    Ground Truth from Wikipedia (normalized such that geometric mean is 1):
    p_A = 0.640
    p_B = 1.043
    p_C = 0.660
    p_D = 2.270
    """
    random.seed(67)

    # construct the dataset of 22 games
    games = []

    def add_games(winner, loser, count):
        for _ in range(count):
            # randomly assign positions
            is_winner_model_a = random.choice([True, False])
            games.append(
                {
                    "team_1": winner if is_winner_model_a else loser,
                    "team_2": loser if is_winner_model_a else winner,
                    "winner": "team_1" if is_winner_model_a else "team_2",
                }
            )

    add_games("A", "B", 2)  # A beats B 2 times
    add_games("B", "A", 3)  # B beats A 3 times
    add_games("A", "D", 1)  # A beats D 1 time
    add_games("D", "A", 4)  # D beats A 4 times
    add_games("B", "C", 5)  # B beats C 5 times
    add_games("C", "B", 3)  # C beats B 3 times
    add_games("C", "D", 1)  # C beats D 1 time
    add_games("D", "C", 3)  # D beats C 3 times

    df = pd.DataFrame(games)
    # initialize dataset
    dataset = PairDataset.from_pandas(
        df, competitor_cols=["team_1", "team_2"], outcome_map={"team_1": 1.0, "team_2": 0.0}, reweighted=False
    )
    # initialize model
    model = BradleyTerry(
        n_competitors=len(dataset.competitors),
        scale=1.0,
        base=np.e,
        init_rating=0.0,
    )
    model.fit(dataset)

    results = model.compute_ratings_and_cis(dataset, significance_level=0.05)
    names = results["competitors"]
    ratings = results["ratings"]

    # convert log-ratings back to 'power' scores
    power_scores = np.exp(ratings)
    score_map = {name: float(score) for name, score in zip(names, power_scores)}

    expected = {"A": 0.640, "B": 1.043, "C": 0.660, "D": 2.270}

    calculated_scores = np.array([score_map[team] for team in ["A", "B", "C", "D"]])
    expected_scores = np.array([expected[team] for team in ["A", "B", "C", "D"]])
    np.testing.assert_allclose(calculated_scores, expected_scores, rtol=1e-3)


def test_contextual_bradley():
    """
    End-to-end test for ContextualBradleyTerry model with reweighting
    """

    # generate a dataset of 10000 matches between 100 competitors with 10 features
    np.random.seed(67)
    n_matches = 10000
    n_competitors = 100
    n_features = 10

    strengths = np.random.randn(n_competitors)
    idxs_a = np.random.randint(0, n_competitors, size=n_matches)
    offsets = np.random.randint(1, n_competitors, size=n_matches)
    idxs_b = (idxs_a + offsets) % n_competitors
    strengths_a = strengths[idxs_a]
    strengths_b = strengths[idxs_b]
    strength_logits = strengths_a - strengths_b
    feature_coeffs = np.random.randn(n_features)
    features = np.random.randn(n_matches, n_features)
    feature_logits = features @ feature_coeffs
    logits = strength_logits + feature_logits
    probs_a_wins = 1 / (1 + np.exp(-logits))
    outcomes = np.random.binomial(1, probs_a_wins)

    competitor_names_a = idxs_a.astype(str)
    competitor_names_b = idxs_b.astype(str)
    df = pd.DataFrame(
        {
            "model_a": competitor_names_a,
            "model_b": competitor_names_b,
            "winner": np.where(outcomes == 1, "model_a", "model_b"),
            **{f"feature_{i}": features[:, i] for i in range(n_features)},
        }
    )

    ds = ContextualPairDataset.from_pandas(
        df, feature_cols=[f"feature_{i}" for i in range(n_features)], reweighted=True
    )
    model = ContextualBradleyTerry(
        n_competitors=n_competitors,
        n_features=n_features,
        init_rating=0.0,
    )
    result = model.compute_ratings_and_cis(ds, significance_level=0.05)
    assert "ratings" in result
    assert "coeffs" in result
    assert len(result["ratings"]) == n_competitors
    assert len(result["coeffs"]) == n_features
