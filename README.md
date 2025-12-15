# arena-ai
Open source rating systems in jax powering the LMArena leaderboard.

## Installation
From pip:
`pip install arena-ai`

From source:
```
git clone https://github.com/lmarena/arena-ai && cd arena-ai
uv sync
```

## Examples
The minimal example of using `arena` to produce a leaderboard on LMArena data can be run in only a few lines:
```python
# Minimal example of using arena to produce a leaderboard using LMArena data
import numpy as np
import datasets
from arena.utils.data_utils import PairDataset
from arena.models.bradley_terry import BradleyTerry

df = datasets.load_dataset(
    "lmarena-ai/arena-human-preference-140k",
    columns=["model_a", "model_b", "winner"]
)["train"].to_pandas()

dataset = PairDataset.from_pandas(df)
model = BradleyTerry(n_competitors=len(dataset.competitors))

# compute ratings and 95% confidence intervals
results = model.compute_ratings_and_cis(dataset, significance_level=0.05)

# print top 10 competitors with ratings and confidence intervals
for idx in np.argsort(-results["ratings"])[:10]:
    competitor = dataset.competitors[idx]
    rating = results["ratings"][idx]
    ci_lower, ci_upper = results["rating_lower"][idx], results["rating_upper"][idx]
    print(f"{competitor:36s}: {rating:7.2f} "f"({ci_lower:7.2f}, {ci_upper:7.2f})")
```

```text
gemini-2.5-pro                      : 1124.07 (1117.61, 1130.53)
gemini-2.5-pro-preview-03-25        : 1097.88 (1082.00, 1113.77)
grok-4-0709                         : 1093.34 (1078.44, 1108.25)
o3-2025-04-16                       : 1079.39 (1072.86, 1085.92)
chatgpt-4o-latest-20250326          : 1078.14 (1071.33, 1084.94)
gemini-2.5-pro-preview-05-06        : 1074.80 (1064.55, 1085.05)
deepseek-r1-0528                    : 1074.48 (1067.19, 1081.78)
grok-3-preview-02-24                : 1071.28 (1063.70, 1078.85)
llama-4-maverick-03-26-experimental : 1067.21 (1059.38, 1075.04)
gemini-2.5-flash                    : 1061.26 (1055.31, 1067.22)
```
(Note that this ranking is with a subset of publicly released data from July, and without style control so it's not reflective of the live leaderboard.)

There are more advanced example notebooks in the [examples](examples/) folder covering techniques such as the style control leaderboard on [LMArena](examples/lmarena.ipynb), analysis of voter patterns on the [PRISM](examples/prism.ipynb) dataset, and analysis of sports and video game competitions using the general Bradley-Terry methodology in [nba.ipynb](examples/nba.ipynb) and [melee.ipynb](examples/melee.ipynb).

## Contributing
We welcome and encourage contributions. To develop `arena` make sure to install the development dependencies and the git pre-commit hooks.

```
uv sync --group dev
pre-commit install
```