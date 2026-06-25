"""README minimal example (locally generated data).

The README example downloads the `lmarena-ai/arena-human-preference-140k`
dataset from HuggingFace, which requires network access. This sandbox has no
network connectivity, so we instead generate a synthetic pairwise-preference
DataFrame on the fly that follows the same schema (`model_a`, `model_b`,
`winner`) and feed it through the exact same `PairDataset` + `BradleyTerry`
pipeline described in the README. The result is a comparable top-10
leaderboard with 95% confidence intervals.
"""
import numpy as np
import pandas as pd

from arena_rank.utils.data_utils import PairDataset
from arena_rank.models.bradley_terry import BradleyTerry


# 12 "models" with known latent strengths. Larger value = stronger.
MODELS = {
    "alpha":    3.0,
    "bravo":    2.6,
    "charlie":  2.2,
    "delta":    1.8,
    "echo":     1.4,
    "foxtrot":  1.0,
    "golf":     0.6,
    "hotel":    0.2,
    "india":   -0.2,
    "juliet":  -0.6,
    "kilo":    -1.0,
    "lima":    -1.5,
}
names = list(MODELS.keys())
strengths = np.array([MODELS[m] for m in names])

rng = np.random.default_rng(0)
rows = []
n_per_pair = 200  # comparisons per unordered pair

for i, a in enumerate(names):
    for b in names[i + 1:]:
        # Bradley-Terry probability that `a` beats `b`
        p_a = 1.0 / (1.0 + np.exp(-(strengths[i] - strengths[names.index(b)])))
        for _ in range(n_per_pair):
            winner = "model_a" if rng.random() < p_a else "model_b"
            rows.append({"model_a": a, "model_b": b, "winner": winner})

df = pd.DataFrame(rows)
print(f"Generated {len(df):,} pairwise comparisons across "
      f"{len(names)} models ({n_per_pair} per unordered pair).\n")

dataset = PairDataset.from_pandas(df)
model = BradleyTerry(n_competitors=len(dataset.competitors))

# Compute ratings and 95% confidence intervals.
results = model.compute_ratings_and_cis(dataset, significance_level=0.05)

leaderboard = (
    pd.DataFrame(results)
    .sort_values("ratings", ascending=False)
    .reset_index(drop=True)
    .head(10)
)
print("Top 10 competitors by Bradley-Terry rating (95% CI):")
print(leaderboard.to_markdown(index=False))
