import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from arena_rank.utils.data_utils import ContextualPairDataset, PairDataset, get_matchups_and_competitors


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


def _viral_prompt_df():
    """Constructs a dataset with: pair (A,B) with 5 votes on unique prompts,
    pair (A,C) with 600 votes all from one viral prompt (heavily down-weighted)."""
    rows = []
    for _ in range(5):
        rows.append(("A", "B", "model_a", 1.0))  # unsaturated, weight 1.0
    for _ in range(600):
        rows.append(("A", "C", "model_a", 50.0 / 600.0))  # viral prompt, weight = cutoff/hash_count
    return pd.DataFrame(rows, columns=["model_a", "model_b", "winner", "dedup_weight"])


def test_row_weight_col_none_is_bit_identical():
    df = pd.DataFrame(
        [
            ("A", "B", "model_a"),
            ("A", "B", "model_b"),
            ("A", "B", "model_a"),
            ("B", "C", "tie"),
            ("B", "C", "model_a"),
        ],
        columns=["model_a", "model_b", "winner"],
    )

    ds_none = PairDataset.from_pandas(df, reweighted=True, min_pair_count=2)
    df_w = df.assign(row_weight=1.0)
    ds_explicit_ones = PairDataset.from_pandas(df_w, reweighted=True, min_pair_count=2, row_weight_col="row_weight")

    np.testing.assert_allclose(ds_none.counts, ds_explicit_ones.counts)
    np.testing.assert_allclose(ds_none.weights, ds_explicit_ones.weights)
    np.testing.assert_allclose(ds_none.opt_weights, ds_explicit_ones.opt_weights)


def test_row_weight_col_viral_prompt_pair_weight_uses_effective_count():
    df = _viral_prompt_df()
    ds = PairDataset.from_pandas(df, reweighted=True, min_pair_count=50, row_weight_col="dedup_weight")

    a_idx = ds.competitor_to_index["A"]
    b_idx = ds.competitor_to_index["B"]
    c_idx = ds.competitor_to_index["C"]

    # Pair (A,B): eff_pair_count = 5 -> below floor -> pair_weight = 1/50.
    ab_mask = ((ds.pairs[:, 0] == a_idx) & (ds.pairs[:, 1] == b_idx)) | (
        (ds.pairs[:, 0] == b_idx) & (ds.pairs[:, 1] == a_idx)
    )
    # Pair (A,C): eff_pair_count = 600 * (50/600) = 50 -> at floor -> pair_weight = 1/50.
    ac_mask = ((ds.pairs[:, 0] == a_idx) & (ds.pairs[:, 1] == c_idx)) | (
        (ds.pairs[:, 0] == c_idx) & (ds.pairs[:, 1] == a_idx)
    )

    # Sum of opt_weights over (A,B): 5 votes each contributing 1 * (1/50) = 0.1
    np.testing.assert_allclose(float(jnp.sum(ds.opt_weights[ab_mask])), 5.0 / 50.0)
    # Sum of opt_weights over (A,C): 600 votes each at dedup_weight=50/600, pair_weight=1/50.
    # Per-aggregated-row contribution: sum_row_weights * pair_weight = 50 * (1/50) = 1.
    np.testing.assert_allclose(float(jnp.sum(ds.opt_weights[ac_mask])), 1.0)

    # counts must be raw row counts (for bootstrap resampling).
    np.testing.assert_allclose(float(jnp.sum(ds.counts[ab_mask])), 5)
    np.testing.assert_allclose(float(jnp.sum(ds.counts[ac_mask])), 600)


def test_row_weight_col_above_floor_pair_contributes_one():
    # Pair with enough effective volume to exceed the floor -> sum of opt_weights == 1.
    rows = [("X", "Y", "model_a", 0.5)] * 200  # 200 rows, each weight 0.5, eff_pair_count = 100
    df = pd.DataFrame(rows, columns=["model_a", "model_b", "winner", "rw"])

    ds = PairDataset.from_pandas(df, reweighted=True, min_pair_count=50, row_weight_col="rw")
    np.testing.assert_allclose(float(jnp.sum(ds.opt_weights)), 1.0)


def test_row_weight_col_contextual_none_equivalence():
    np.random.seed(7)
    n = 60
    df = pd.DataFrame(
        {
            "model_a": np.random.choice(["A", "B", "C"], n),
            "model_b": np.random.choice(["A", "B", "C"], n),
            "winner": np.random.choice(["model_a", "model_b"], n),
            "feature_0": np.random.randn(n),
            "feature_1": np.random.randn(n),
        }
    )
    df = df[df["model_a"] != df["model_b"]].reset_index(drop=True)

    ds_none = ContextualPairDataset.from_pandas(df, feature_cols=["feature_0", "feature_1"], reweighted=True)
    df_w = df.assign(rw=1.0)
    ds_ones = ContextualPairDataset.from_pandas(
        df_w, feature_cols=["feature_0", "feature_1"], reweighted=True, row_weight_col="rw"
    )

    np.testing.assert_allclose(ds_none.weights, ds_ones.weights)


def test_row_weight_col_contextual_viral_prompt():
    df = _viral_prompt_df()
    df["feat"] = np.linspace(-1.0, 1.0, len(df))

    ds = ContextualPairDataset.from_pandas(
        df,
        feature_cols=["feat"],
        reweighted=True,
        min_pair_count=50,
        row_weight_col="dedup_weight",
    )

    # All rows present (no aggregation).
    assert ds.pairs.shape[0] == len(df)
    # Weights are mean-normalized (mean ~ 1).
    np.testing.assert_allclose(float(jnp.mean(ds.weights)), 1.0, rtol=1e-10)


def test_ess_counts_kish_within_bucket():
    df = pd.DataFrame(
        [("A", "B", "model_a", 1.0), ("A", "B", "model_a", 0.2), ("A", "B", "model_b", 1.0)],
        columns=["model_a", "model_b", "winner", "w"],
    )

    ds = PairDataset.from_pandas(df, reweighted=False, row_weight_col="w")

    # Mixed-weight bucket: ess = (1 + 0.2)^2 / (1 + 0.04); constant-weight bucket: ess = count.
    np.testing.assert_allclose(sorted(np.asarray(ds.ess_counts)), [1.0, 1.44 / 1.04])
    ds_unweighted = PairDataset.from_pandas(df, reweighted=False)
    np.testing.assert_allclose(ds_unweighted.ess_counts, ds_unweighted.counts)


def test_ess_counts_all_zero_weight_bucket_uses_raw_count():
    df = pd.DataFrame(
        [("A", "B", "model_a", 0.0), ("A", "B", "model_a", 0.0), ("A", "B", "model_b", 1.0)],
        columns=["model_a", "model_b", "winner", "w"],
    )

    ds = PairDataset.from_pandas(df, reweighted=False, row_weight_col="w")

    np.testing.assert_allclose(ds.ess_counts, ds.counts)
    assert np.isfinite(np.asarray(ds.ess_counts)).all()


def test_sandwich_meat_matches_per_row_scores():
    from arena_rank.models.bradley_terry import BradleyTerry

    df = pd.DataFrame(
        [
            ("A", "B", "model_a", 1.0),
            ("A", "B", "model_a", 0.2),
            ("A", "B", "model_b", 0.5),
            ("B", "C", "model_a", 1.0),
        ],
        columns=["model_a", "model_b", "winner", "w"],
    )
    ds = PairDataset.from_pandas(df, reweighted=False, row_weight_col="w")
    ratings = jnp.array([0.3, 0.0, -0.2])

    hess, cov = BradleyTerry.compute_hessian_and_covariance(
        ratings, ds.pairs, ds.outcomes, ds.ess_counts, ds.opt_weights, 0.0, 3
    )

    # Per-row reference: every vote is its own aggregate with count 1.
    matchups, _ = get_matchups_and_competitors(df, ["model_a", "model_b"])
    hess_ref, cov_ref = BradleyTerry.compute_hessian_and_covariance(
        ratings,
        matchups,
        jnp.array([1.0, 1.0, 0.0, 1.0]),
        jnp.ones(4),
        jnp.array(df["w"].values),
        0.0,
        3,
    )

    np.testing.assert_allclose(hess, hess_ref)
    np.testing.assert_allclose(cov, cov_ref)
