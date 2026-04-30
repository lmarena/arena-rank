import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
from arena_rank.utils.data_utils import get_matchups_and_competitors, PairDataset


def test_get_matchups_and_competitors() -> None:
    col_a = ["D", "A", "B", "C"]
    col_b = ["B", "C", "D", "A"]
    df = pd.DataFrame({"col_a": col_a, "col_b": col_b})
    matchups, competitors = get_matchups_and_competitors(df, competitor_cols=["col_a", "col_b"])
    assert competitors == ["A", "B", "C", "D"]
    expected_matchups = np.array([[3, 1], [0, 2], [1, 3], [2, 0]])
    np.testing.assert_array_equal(matchups, expected_matchups)


def test_pair_dataset_aggregation_logic():
    data = [
        ("comp_0", "comp_1", 1.0, 2),  # comp_0 beats comp_1 twice
        ("comp_0", "comp_1", 0.0, 1),  # comp_1 beats comp_0 once
        ("comp_0", "comp_1", 0.5, 1),  # comp_0 ties comp_1 once
        ("comp_1", "comp_0", 1.0, 1),  # comp_1 beats comp_0 once (reverse order)
        ("comp_2", "comp_0", 0.0, 1),  # comp_0 beats comp_2 once (reverse order)
        ("comp_2", "comp_0", 1.0, 3),  # comp_2 beats comp_0 three times (reverse order)
    ]

    rev = {1.0: "comp_a", 0.0: "comp_b", 0.5: "tie"}
    raw = [(a, b, rev[o]) for (a, b, o, c) in data for _ in range(c)]
    df = pd.DataFrame(raw, columns=["comp_a", "comp_b", "winner"])

    dataset = PairDataset.from_pandas(
        df,
        competitor_cols=["comp_a", "comp_b"],
        outcome_col="winner",
        outcome_map={"comp_a": 1.0, "comp_b": 0.0, "tie": 0.5},
        reweighted=False,
    )

    # expected outputs
    exp = jnp.array([(dataset.competitor_to_index[a], dataset.competitor_to_index[b], o, c) for (a, b, o, c) in data])
    match = (
        (exp[:, 0, None] == dataset.pairs[:, 0])
        & (exp[:, 1, None] == dataset.pairs[:, 1])
        & jnp.isclose(exp[:, 2, None], dataset.outcomes)
    )
    assert jnp.all(jnp.sum(match, axis=1) == 1)
    assert jnp.allclose(jnp.sum(match * dataset.counts, axis=1), exp[:, 3])
    assert jnp.sum(dataset.counts) == len(df)


@pytest.mark.parametrize("n_competitors", [5])
@pytest.mark.parametrize("n_matches_per_pair", [10])
def test_pair_dataset_properties_randomized(n_competitors, n_matches_per_pair):
    """
    Randomized property-based test to ensure consistency on larger data
    and verify weight calculations.
    """
    np.random.seed(42)

    competitors = np.char.add("c_", np.arange(n_competitors).astype(str))

    # Generate all pairs (including reverse orders to test robustness)
    # We intentionally use a list of indices then map to names to control ground truth
    idx_a = np.random.randint(0, n_competitors, n_matches_per_pair * n_competitors**2)
    idx_b = np.random.randint(0, n_competitors, n_matches_per_pair * n_competitors**2)

    # Filter out self-play
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]

    model_a_col = competitors[idx_a]
    model_b_col = competitors[idx_b]

    # Random winners
    outcomes = np.random.choice(["model_a", "model_b", "tie"], size=len(idx_a))

    df = pd.DataFrame({"model_a": model_a_col, "model_b": model_b_col, "winner": outcomes})

    # 2. Execution (With Reweighting)
    dataset = PairDataset.from_pandas(
        df,
        reweighted=True,
        min_pair_count=1,  # Set low to ensure math is easy to check
    )
    # number of unique (A,B, outcome) triplets must equal number of rows in dataset
    unique_triplets = df.groupby(["model_a", "model_b", "winner"]).size().reset_index()
    assert dataset.n_pairs == len(unique_triplets)

    # sum of triplet counts must equal total rows in original df
    assert np.isclose(jnp.sum(dataset.counts), len(df))

    random_idx = np.random.randint(0, dataset.n_pairs)
    p1, p2 = dataset.pairs[random_idx]

    pair_mask = ((dataset.pairs[:, 0] == p1) & (dataset.pairs[:, 1] == p2)) | (
        (dataset.pairs[:, 0] == p2) & (dataset.pairs[:, 1] == p1)
    )

    total_pair_observations = jnp.sum(dataset.counts[pair_mask])
    expected_weight_val = 1.0 / max(float(total_pair_observations), float(1))
    assert np.isclose(dataset.weights[random_idx], expected_weight_val)
