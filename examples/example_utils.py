"""Util functions for the example notebooks"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tiktoken


### LMarena
def extract_metadata_features(df):
    """Extracts nested metadata dictionaries into top-level columns."""
    feature_map = {
        "sum_assistant_a_tokens": "sum_assistant_tokens_a",
        "sum_assistant_b_tokens": "sum_assistant_tokens_b",
    }
    # for dictionary columns (like header_count), sum the values
    for key in ["header_count", "bold_count", "list_count"]:
        for suffix in ["_a", "_b"]:
            col_name = f"{key}{suffix}"
            df[col_name] = df["conv_metadata"].apply(
                lambda x: sum(x[col_name].values()) if isinstance(x[col_name], dict) else 0
            )

    for old, new in feature_map.items():
        df[new] = df["conv_metadata"].apply(lambda x: x[old])
    return df


def add_style_feature_cols(df, feature_names):
    """computes normalized relative differences for the style features."""
    df = extract_metadata_features(df)

    for feature in feature_names:
        col_a, col_b = f"{feature}_a", f"{feature}_b"
        diff = df[col_a] - df[col_b]
        total = df[col_a] + df[col_b]
        total = total.replace(0, 1)
        df[feature + "_raw"] = diff / total
        df[feature] = (df[feature + "_raw"] - df[feature + "_raw"].mean()) / df[feature + "_raw"].std()

    df = df[["model_a", "model_b", "winner"] + feature_names]
    return df


def plot_leaderboard(
    results,
    top_n=20,
    item_name="Model",
    rating_name="Arena Score",
    title="Style-Control Leaderboard",
    label=None,
    results_v2=None,
    label_v2="Secondary",
):
    """
    Plots the leaderboard. If results_v2 is provided, it plots a comparison
    with the second set of results in orange.
    """
    leaderboard_df = pd.DataFrame(
        {
            item_name: results["competitors"],
            "Rating": results["ratings"],
            "Lower": results["rating_lower"],
            "Upper": results["rating_upper"],
        }
    )
    leaderboard_df = leaderboard_df.sort_values(by="Rating", ascending=False).reset_index(drop=True)
    leaderboard_df["error_lower"] = leaderboard_df["Rating"] - leaderboard_df["Lower"]
    leaderboard_df["error_upper"] = leaderboard_df["Upper"] - leaderboard_df["Rating"]
    plot_df = leaderboard_df.head(top_n)

    plt.figure(figsize=(14, 8))
    plt.errorbar(
        x=plot_df[item_name],
        y=plot_df["Rating"],
        yerr=[plot_df["error_lower"], plot_df["error_upper"]],
        fmt="o",
        color="royalblue",
        alpha=0.8 if results_v2 is not None else 1.0,
        linewidth=2.5,
        capsize=4,
        markersize=8,
        label=label,
    )

    # handle optional secondary data
    if results_v2 is not None:
        df2 = pd.DataFrame(
            {
                item_name: results_v2["competitors"],
                "Rating": results_v2["ratings"],
                "Lower": results_v2["rating_lower"],
                "Upper": results_v2["rating_upper"],
            }
        )

        # calculate error bars
        df2["error_lower"] = df2["Rating"] - df2["Lower"]
        df2["error_upper"] = df2["Upper"] - df2["Rating"]

        # merge plot_df's names with df2 to ensure correct X-axis alignment
        plot_df_v2 = plot_df[[item_name]].merge(df2, on=item_name, how="left")
        plt.errorbar(
            x=plot_df_v2[item_name],
            y=plot_df_v2["Rating"],
            yerr=[plot_df_v2["error_lower"], plot_df_v2["error_upper"]],
            fmt="o",
            color="orange",
            alpha=0.6,
            linewidth=2.5,
            capsize=4,
            markersize=8,
            label=label_v2,
        )
        plt.legend(fontsize=12)

    plt.title(f"{title} (Top {top_n})", fontsize=16)
    plt.ylabel(rating_name, fontsize=12)
    plt.xlabel(f"{item_name} Name", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.show()


### Melee


def plot_melee_bump_chart(df_all_years, message_type="custom"):
    PLAYER_COLOR_MAP = {
        "Mang0": "#D55E00",
        "Armada": "#0072B2",
        "Hungrybox": "#E69F00",
        "Mew2King": "#009E73",
        "Ken": "#CC79A7",
        "Zain": "#F0E442",
        "PPMD": "#56B4E9",
        "Cody Schwab": "#000000",
        "Leffen": "#0000FF",
        "Azen": "#00FF00",
        "ChuDat": "#FF00E4",
        "PC Chris": "#1A0078",
        "Moky": "#BFC98A",
        "Plup": "#FF005D",
        "AMSa": "#505000",
        "KoreanDJ": "#5D0000",
        "Jmook": "#AE78FF",
        "Isai": "#93FF86",
        "Cort": "#500D43",
        "Axe": "#00FFFF",
        "DaShizWiz": "#FFA186",
        "Sastopher": "#6B6B6B",
        "Wizzrobe": "#358600",
    }

    df_all_years = df_all_years.copy()
    df_all_years["Rank"] = df_all_years.groupby("Year")["Rating"].rank(ascending=False, method="first")
    df_top = df_all_years[df_all_years["Rank"] <= 5].copy()

    # Calculate sort order only (colors are now static)
    df_top["Score"] = 1 / df_top["Rank"]
    sorted_players = df_top.groupby("Competitor")["Score"].sum().sort_values(ascending=False).index.tolist()

    plt.style.use("default")
    plt.figure(figsize=(24, 8))
    unique_years = sorted(df_top["Year"].unique())

    plt.axvspan(2020 - 0.5, 2021 + 0.5, color="#d4d4d4", alpha=0.6, zorder=1)
    text = (
        "Due to Covid the data is mostly from online tournaments and is of lower quality"
        if message_type == "custom"
        else "Due to Covid no official ranks were released"
    )
    plt.text(
        (2020 + 2021) / 2,
        5.8,
        text,
        fontsize=9,
        color="black",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="grey", linewidth=0.5),
        zorder=10,
    )

    for player in sorted_players:
        player_data = df_top[df_top["Competitor"] == player].sort_values("Year")
        if player_data.empty:
            continue

        years, ranks = player_data["Year"].values, player_data["Rank"].values
        segments = []
        if len(years) > 0:
            cx, cy = [years[0]], [ranks[0]]
            for i in range(1, len(years)):
                if years[i] == years[i - 1] + 1:
                    cx.append(years[i])
                    cy.append(ranks[i])
                else:
                    segments.append((cx, cy))
                    cx, cy = [years[i]], [ranks[i]]
            segments.append((cx, cy))

        # Use static color map, default to black if player not in map
        color = PLAYER_COLOR_MAP.get(player, "#000000")

        for x_seg, y_seg in segments:
            plt.plot(x_seg, y_seg, marker="o", linewidth=5, color=color, zorder=5)
            last_x, last_y = x_seg[-1], y_seg[-1]
            offset = 0.1
            if len(y_seg) >= 2:
                final_y = last_y - offset if last_y <= y_seg[-2] else last_y + offset
            else:
                final_y = last_y - offset
            plt.text(
                last_x, final_y, player, color=color, fontweight="bold", fontsize=11, ha="center", va="center", zorder=6
            )

    plt.gca().invert_yaxis()
    plt.yticks(range(1, 6))
    xtick_labels = [str(y) + ("*" if y in [2020, 2021] else "") for y in unique_years]
    plt.xticks(unique_years, xtick_labels, fontsize=12)
    plt.ylabel("Rank (1=Highest)")
    plt.xlabel("Year")
    extra = "(SSBMRank/RetroSSBMRank)" if message_type == "official" else "(Bradley-Terry Rankings)"
    plt.title(f"Top 5 Melee Players Over Time {extra}", fontsize=14, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    if unique_years:
        plt.xlim(min(unique_years) - 0.5, max(unique_years) + 0.5)
    plt.ylim(5.5, 0.7)
    plt.tight_layout()
    plt.show()


def filter_melee_matches(df):
    # reduces noise by applying two filters:
    # 1. Remove players with few unique opponents, this helps prevent issues due to disconnected components
    # 2. Remove players who have only wins or only losses, these cases would cause BT ratings to to to infinity or -infinity
    opponents_1 = df.groupby("competitor_1")["competitor_2"].nunique()
    opponents_2 = df.groupby("competitor_2")["competitor_1"].nunique()
    total_opponents = opponents_1.add(opponents_2, fill_value=0)
    total_opponent_thresh = total_opponents.quantile(0.70)
    total_opponent_thresh = max(total_opponent_thresh, 3)
    players_with_min_opponents = total_opponents[total_opponents >= total_opponent_thresh].index
    df = df[df["competitor_1"].isin(players_with_min_opponents) & df["competitor_2"].isin(players_with_min_opponents)]

    # iteratively remove players with only wins or only losses
    # apply several times since removing one player can cause another to have only wins or losses
    for _ in range(5):
        player_wins = (
            df.groupby("competitor_1")["outcome"]
            .sum()
            .add(df.groupby("competitor_2")["outcome"].apply(lambda x: (x == 0).sum()), fill_value=0)
        )
        player_losses = (
            df.groupby("competitor_1")["outcome"]
            .apply(lambda x: (x == 0).sum())
            .add(df.groupby("competitor_2")["outcome"].sum(), fill_value=0)
        )
        players_with_both = player_wins[(player_wins >= 1) & (player_losses >= 1)].index
        df = df[(df["competitor_1"].isin(players_with_both)) & (df["competitor_2"].isin(players_with_both))]
    return df


def filter_melee_leaderboard(leaderboard_df, match_df):
    # even after filtering matches, some players may still have extreme records, or very few matches and should be removed from the leaderboard
    # filter based on number of unique opponents in the match_df and on total matches played
    # heuristics are applied based on how many total matches are in the match_df
    opponents_1 = match_df.groupby("competitor_1")["competitor_2"].nunique()
    opponents_2 = match_df.groupby("competitor_2")["competitor_1"].nunique()
    total_opponents = opponents_1.add(opponents_2, fill_value=0)
    players_with_min_opponents = set(total_opponents[total_opponents >= 10].index)

    counts = leaderboard_df.head(100).set_index("Competitor")["Matches Played"]
    # heuristics for minimum matches played based on total matches that year
    total_matches = len(match_df)
    if total_matches < 1000:
        lower_bound = 10
    elif total_matches < 4000:
        lower_bound = 20
    else:
        lower_bound = 30
    # only keep players who have played at least the 60th percentile of matches played among the top 100 players
    threshold = min(max(counts.quantile(0.6), lower_bound), 35)
    players_with_min_matches = set(leaderboard_df[leaderboard_df["Matches Played"] >= threshold]["Competitor"].tolist())
    leaderboard_players = players_with_min_opponents.intersection(players_with_min_matches)
    leaderboard_players.discard("Zion")  # Zion is a banned player and ineligible for ranking
    leaderboard_df = leaderboard_df[leaderboard_df["Competitor"].isin(leaderboard_players)].reset_index(drop=True)
    leaderboard_df["Competitor"] = leaderboard_df["Competitor"].apply(lambda x: x.replace(" (Melee player)", ""))
    return leaderboard_df


def convert_ssbmrank_dict_to_plotting_df(ranking_dict):
    data = []
    for year, players in ranking_dict.items():
        if players == ["No Official Ranking"]:
            data.append({"Competitor": f"Placeholder-{year}", "Year": year, "Rating": 0})
            continue
        for i, player in enumerate(players):
            if not player:
                print(f"Empty player name for year {year}, rank {i + 1}")
            rank = i + 1
            rating = 6 - rank
            if rank <= 5:
                data.append({"Competitor": player, "Year": year, "Rating": rating})
    df = pd.DataFrame(data)
    min_year = min(ranking_dict.keys())
    max_year = max(ranking_dict.keys())
    all_years_range = range(min_year, max_year + 1)
    all_years_df = pd.DataFrame({"Year": list(all_years_range)})
    df_all_years = all_years_df.merge(df, on="Year", how="left")
    return df_all_years


### PRISM
def extract_model_message(history):
    """
    Filters conversation history to return a list of turn 0 model messages.
    Calculates token counts for the content of the message.
    """
    tokenizer = tiktoken.get_encoding("o200k_base")

    def get_token_count(content):
        text = str(content) if content is not None else ""
        return len(tokenizer.encode(text))

    return [
        {
            "model_name": msg.get("model_name"),
            "score": msg.get("score"),
            "token_count": get_token_count(msg.get("content")),
        }
        for msg in history
        if msg.get("role") == "model" and msg.get("turn") == 0
    ]


def preprocess_prism(df):
    """
    Reads prism data, filters for turn 0 model responses,
    creates unique pairs, and includes token counts for both models.
    """

    df["filtered_messages"] = df["conversation_history"].apply(extract_model_message)

    # explode to have one row per turn/vote rather than per conversation
    message_df = df[["conversation_id", "user_id", "filtered_messages"]].explode("filtered_messages")

    # add token counts to the exploded dataframe
    message_df = pd.concat(
        [message_df.drop("filtered_messages", axis=1), message_df["filtered_messages"].apply(pd.Series)], axis=1
    ).dropna(subset=["model_name"])

    # self-join to create pairs
    pair_df = message_df.merge(message_df, on=["conversation_id", "user_id"], suffixes=("_a", "_b"), how="inner")

    # filter duplicate pairs model_a < model_b
    pair_df = pair_df[pair_df["model_name_a"] < pair_df["model_name_b"]].copy()

    conditions = [pair_df["score_a"] > pair_df["score_b"], pair_df["score_a"] < pair_df["score_b"]]
    choices = [1.0, 0.0]
    pair_df["outcome"] = np.select(conditions, choices, default=0.5)

    pair_df = pair_df.rename(
        columns={
            "model_name_a": "model_a",
            "model_name_b": "model_b",
            "token_count_a": "tokens_a",
            "token_count_b": "tokens_b",
        }
    )
    return pair_df


def add_per_model_feature_cols(
    df: pd.DataFrame,
    col_model_a: str = "model_a",
    col_model_b: str = "model_b",
    col_context: str = "lm_familiarity",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds contextual skill features per competitor to a DataFrame for each unique value of the context column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_model_a (str): Name of the column containing the first compared item.
        col_model_b (str): Name of the column containing the second compared item.
        col_context (str): Name of the column containing the context.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            1. The DataFrame with new features added.
            2. A list of the names of the added feature columns.
    """

    # identify all unique models and context levels
    all_models = np.unique(pd.concat([df[col_model_a], df[col_model_b]]).astype(str))
    context_levels = df[col_context].unique()

    new_col_names = []
    new_columns_data = {}

    # calculate masks for performance for each context feature value
    context_masks = {ctx: (df[col_context] == ctx).values for ctx in context_levels}

    model_a_values = df[col_model_a].values
    model_b_values = df[col_model_b].values

    for model in all_models:
        # Pre-calculate where this model appears
        is_model_a = model_a_values == model
        is_model_b = model_b_values == model

        for context_level in context_levels:
            col_name = f"{model}_{context_level}"
            new_col_names.append(col_name)

            is_context = context_masks[context_level]
            col_data = np.zeros(len(df), dtype=np.int8)

            # set +1 where the context feature matches and Model is in A
            col_data[is_context & is_model_a] = 1

            # set -1 where the context feature matches and Model is in B
            col_data[is_context & is_model_b] = -1
            new_columns_data[col_name] = col_data

    new_features_df = pd.DataFrame(new_columns_data, index=df.index)
    df_final = pd.concat([df, new_features_df], axis=1)
    return df_final, new_col_names


def add_token_diff_feature_cols(
    df: pd.DataFrame,
    col_context: str = "lm_familiarity",
    col_tokens_a: str = "tokens_a",
    col_tokens_b: str = "tokens_b",
    add_intercept: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds interaction of token difference with the values of a contextual categorical feature, and an optional intercept.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_context (str): Name of the categorical column (e.g., "gender").
        col_tokens_a (str): Column for Model A token counts.
        col_tokens_b (str): Column for Model B token counts.
        add_intercept (bool): If True, adds a column of 1s named "intercept".

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            1. DataFrame with new features.
            2. List of the new column names.
    """

    diff_values = df[col_tokens_a].values - df[col_tokens_b].values
    context_levels = df[col_context].unique()

    new_col_names = []
    new_columns_data = {}

    if add_intercept:
        new_col_names.append("intercept")
        new_columns_data["intercept"] = np.ones(len(df), dtype=np.int8)

    # create interaction columns (Difference * one-hot context)
    for ctx in context_levels:
        col_name = f"{ctx}_token_diff"
        new_col_names.append(col_name)
        # mask for current context
        mask = (df[col_context] == ctx).values
        new_columns_data[col_name] = diff_values * mask

    new_features_df = pd.DataFrame(new_columns_data, index=df.index)

    df_final = pd.concat([df, new_features_df], axis=1)
    df_final["token_diff"] = diff_values
    return df_final, new_col_names


def plot_prism_results(results, feature_cols, competitors, category_label="Category"):
    """Plots models ratings and contextual offsets from PRISM results."""

    elo_scale = 400 / np.log(10)  # ~173.71

    ratings_array = np.array(results["ratings"])
    coeffs_array = np.array(results["coeffs"])

    raw_base_ratings = dict(zip(competitors, ratings_array))
    coeff_map = dict(zip(feature_cols, coeffs_array))

    processed_data = []
    for model in competitors:
        raw_base_elo = raw_base_ratings[model]
        model_specific_offsets_logodds = {}
        found_contexts = []

        prefix = f"{model}_"
        for col, val in coeff_map.items():
            clean_col = col.replace("skill_", "").replace("adv_", "")

            if clean_col.startswith(prefix):
                ctx = clean_col[len(prefix) :]
                model_specific_offsets_logodds[ctx] = val
                found_contexts.append(ctx)

        effective_ratings_elo = []
        for ctx in found_contexts:
            offset_elo = model_specific_offsets_logodds[ctx] * elo_scale
            effective_ratings_elo.append(raw_base_elo + offset_elo)

        # there is an extra degree of freedom, so center around the average and compute per-context offsets
        true_average_elo = np.mean(effective_ratings_elo)
        for i, ctx in enumerate(found_contexts):
            eff_rating_elo = effective_ratings_elo[i]
            real_offset_elo = eff_rating_elo - true_average_elo
            processed_data.append(
                {
                    "Model": model,
                    "Category": ctx,
                    "Base_Rating": true_average_elo,
                    "Real_Offset": real_offset_elo,
                    "Visual_Offset": real_offset_elo,
                }
            )

    df_plot = pd.DataFrame(processed_data)

    df_base_unique = df_plot[["Model", "Base_Rating"]].drop_duplicates().sort_values("Base_Rating", ascending=False)
    order = df_base_unique["Model"].tolist()
    y_map = {m: i for i, m in enumerate(order)}

    pivot_df = df_plot.pivot(index="Model", columns="Category", values="Real_Offset")
    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.reindex(order)

    unique_categories = sorted(df_plot["Category"].unique())

    fig, ax = plt.subplots(figsize=(14, len(competitors) * 0.7))

    ax.scatter(
        df_base_unique["Base_Rating"],
        [y_map[m] for m in df_base_unique["Model"]],
        color="black",
        marker="D",
        s=40,
        zorder=5,
    )

    for _, row in df_plot.iterrows():
        model = row["Model"]
        if model not in y_map:
            continue

        y_pos = y_map[model]
        base = row["Base_Rating"]
        offset = row["Visual_Offset"]
        cat = row["Category"]
        x_pos = base + offset
        color = f"C{unique_categories.index(cat)}"

        ax.plot([base, x_pos], [y_pos, y_pos], color=color, alpha=0.5, linewidth=1.0, zorder=3)
        ax.scatter(x_pos, y_pos, color=color, marker="|", s=300, linewidth=2.5, zorder=10)

    legend_handles = [
        mlines.Line2D([], [], color="black", marker="D", linestyle="None", markersize=6, label="Average Rating")
    ]
    for i, cat in enumerate(unique_categories):
        legend_handles.append(
            mlines.Line2D(
                [], [], color=f"C{i}", marker="|", linestyle="None", markersize=12, markeredgewidth=2, label=cat
            )
        )

    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", title=category_label)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=11)
    ax.set_title(f"Contextual BT Ratings and Offsets by {category_label}", fontsize=16, pad=20)
    ax.set_xlabel("BT Rating (Elo Scale)", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    return plt
