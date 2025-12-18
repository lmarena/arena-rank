<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lmarena/arena-rank/refs/heads/main/arena_rank/assets/logo-dark.png">
    <img alt="LMArena logo" src="https://raw.githubusercontent.com/lmarena/arena-rank/refs/heads/main/arena_rank/assets/logo.png" width=30%>
  </picture>
</p>

<h3 align="center">
Arena-Rank: The ranking methodology powering the LMArena leaderboard.
</h3>

<p align="center">
| <a href="https://lmarena.ai"><b>LMArena</b></a> | <a href="https://news.lmarena.ai"><b>Blog</b></a> | <a href="https://x.com/arena"><b>X</b></a> | <a href="https://discord.com/invite/LMArena"><b>Discord</b></a> | <a href="https://www.linkedin.com/company/lmarena"><b>LinkedIn</b></a> |
</p>


## Installation
From pip:
`pip install arena-rank`

From source:
```
git clone https://github.com/lmarena/arena-rank && cd arena-rank
uv sync
```

## Examples
Below is a minimal example using Arena-Rank to produce a leaderboard on LMArena data:
```python
import pandas as pd
import datasets
from arena_rank.utils.data_utils import PairDataset
from arena_rank.models.bradley_terry import BradleyTerry

df = datasets.load_dataset(
    "lmarena-ai/arena-human-preference-140k",
    columns=["model_a", "model_b", "winner"]
)["train"].to_pandas()

dataset = PairDataset.from_pandas(df)
model = BradleyTerry(n_competitors=len(dataset.competitors))

# compute ratings and 95% confidence intervals
results = model.compute_ratings_and_cis(dataset, significance_level=0.05)
```
We visualize the top 10 models on the leaderboard:
```python
# print top 10 competitors with ratings and confidence intervals
leaderboard = pd.DataFrame(results).sort_values("ratings", ascending=False).head(10)
print(leaderboard.to_markdown(index=False))
```

```text
| competitors                         |   ratings |   rating_lower |   rating_upper |   variances |
|:------------------------------------|----------:|---------------:|---------------:|------------:|
| gemini-2.5-pro                      |   1124.07 |        1117.61 |        1130.53 |    10.8542  |
| gemini-2.5-pro-preview-03-25        |   1097.88 |        1082    |        1113.77 |    65.6717  |
| grok-4-0709                         |   1093.34 |        1078.44 |        1108.25 |    57.8409  |
| o3-2025-04-16                       |   1079.39 |        1072.86 |        1085.92 |    11.0919  |
| chatgpt-4o-latest-20250326          |   1078.14 |        1071.33 |        1084.94 |    12.0447  |
| gemini-2.5-pro-preview-05-06        |   1074.8  |        1064.55 |        1085.05 |    27.3722  |
| deepseek-r1-0528                    |   1074.48 |        1067.19 |        1081.78 |    13.8388  |
| grok-3-preview-02-24                |   1071.28 |        1063.7  |        1078.85 |    14.9286  |
| llama-4-maverick-03-26-experimental |   1067.21 |        1059.38 |        1075.04 |    15.953   |
| gemini-2.5-flash                    |   1061.26 |        1055.31 |        1067.22 |    9.21695  |
```

See the [examples](examples/) folder for notebooks with more advanced examples, covering techniques such as the style-controlled leaderboard on [LMArena](examples/lmarena.ipynb), analysis of voter patterns on the [PRISM](examples/prism.ipynb) dataset, and analysis of [sports](examples/nba.ipynb) and [video game competitions](examples/melee.ipynb) using the Bradley-Terry methodology.

## Contributing
We welcome and encourage contributions. To develop Arena-Rank, make sure to install the development dependencies and the git pre-commit hooks.

```
uv sync --group dev
pre-commit install
```
