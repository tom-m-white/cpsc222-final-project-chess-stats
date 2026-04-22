import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


RAW_DATA_FILENAME = "imsogarb69_all_2604212016.csv"
GAMES_FILENAME = "games_clean.csv"
CALENDAR_FILENAME = "calendar_table.csv"
DAILY_SUMMARY_FILENAME = "daily_summary.csv"
MODEL_FEATURES = ["timeClass", "userColor", "userRating", "opponentRating", "rating_diff", "opponent_strength_bucket", "weekday", "hour_bucket_et", "opening_family", "games_played_so_far_today"]
NUMERIC_FEATURES = ["userRating", "opponentRating", "rating_diff", "games_played_so_far_today"]
CATEGORICAL_FEATURES = ["timeClass","userColor","opponent_strength_bucket","weekday","hour_bucket_et","opening_family"]
TIME_CONTROLS = ("blitz", "rapid", "bullet")
DRAW_RESULTS = {"agreed", "stalemate", "insufficient", "repetition", "timevsinsufficient"}
DEFAULT_OPENING_MIN_GAMES = 8
DEFAULT_RANDOM_STATE = 222

def load_data():
    parse_date_columns = ["date", "week_start"]
    games = pd.read_csv(GAMES_FILENAME, parse_dates=parse_date_columns)
    calendar = pd.read_csv(CALENDAR_FILENAME, parse_dates=parse_date_columns)
    daily = pd.read_csv(DAILY_SUMMARY_FILENAME, parse_dates=parse_date_columns)

    return games, calendar, daily

def dataset_table_overview(games, calendar, daily):
    print("Dataset Table Overview:")
    table = (
            [
                {
                    "table": GAMES_FILENAME,
                    "rows": len(games),
                    "columns": games.shape[1],
                },
                {
                    "table": CALENDAR_FILENAME,
                    "rows": len(calendar),
                    "columns": calendar.shape[1],
                },
                {
                    "table": DAILY_SUMMARY_FILENAME,
                    "rows": len(daily),
                    "columns": daily.shape[1],
                },
            ]
        )
    print(pd.DataFrame(table))
    print()

def project_scope_summary_df(games, daily):
    summary_records = [
        {"metric": "Games analyzed", "value": len(games)},
        {"metric": "Active days", "value": daily["date"].nunique()},
        {"metric": "First date", "value": games["date"].min().date().isoformat()},
        {"metric": "Last date", "value": games["date"].max().date().isoformat()},
        {"metric": "Overall win rate", "value": round(float(games["is_win"].mean()), 4)},
        {"metric": "Average games per active day", "value": round(float(daily["games_played"].mean()), 2)},
    ]
    summary_df = pd.DataFrame(summary_records)
    print("Project Scope Summary:")
    print(summary_df)
    print()

def time_control_distribution(games):
    distribution = games["timeClass"].value_counts().rename_axis("timeClass").reset_index(name="games")
    distribution["share"] = distribution["games"] / distribution["games"].sum()
    print("Time Control Distribution:")
    print(distribution)
    print()

def clean_games_data(games):
    games = games.copy()
    string_columns = games.select_dtypes(include=["object","string"]).columns
    for col in string_columns:
        games[col] = games[col].str.strip()
    games = games.replace("", np.nan)
    games.rename(columns={"outcome": "move_count", "moveCount" : "terminal_outcome"})
    games = games[games["timeClass"].isin(TIME_CONTROLS)].copy()

    numeric_columns = [
        "userAccuracy",
        "opponentAccuracy",
        "userRating",
        "opponentRating",
        "move_count",
    ]

    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column])

    games["date"] = pd.to_datetime(games["date"], format="%Y.%m.%d")
    games["start_datetime_et"] = pd.to_datetime(
        games["date"].dt.strftime("%Y-%m-%d") + " " + games["startTime"],
        format="%Y-%m-%d %H:%M:%S",
    )

    games["end_datetime_et"] = pd.to_datetime(
        games["date"].dt.strftime("%Y-%m-%d") + " " + games["endTime"],
        format="%Y-%m-%d %H:%M:%S",
    )

    games["start_hour_et"] = games["start_datetime_et"].dt.hour
    games["end_hour_et"] = games["end_datetime_et"].dt.hour

    games = games.sort_values(["date", "startTime", "endTime", "gameId"]).reset_index(drop=True)

    games["is_win"] = (games["result"] == "win").astype(int)
    games["is_draw"] = games["result"].isin(DRAW_RESULTS).astype(int)
    games["is_loss"] = ((games["is_win"] == 0) & (games["is_draw"] == 0)).astype(int)
    games["result_group"] = np.select(
        [games["is_win"] == 1, games["is_draw"] == 1],
        ["win", "draw"],
        default="loss",
    )
    games["rating_diff"] = games["userRating"] - games["opponentRating"]
    games["opponent_strength_bucket"] = games["rating_diff"].apply(bucket_opponent_strength)
    games["hour_bucket_et"] = games["start_hour_et"].apply(bucket_hour)
    games["opening_family"] = (
        games["opening"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    games["games_played_so_far_today"] = games.groupby("date").cumcount()
    return games

def bucket_hour(hour: int) -> str:
    if hour < 6:
        return "late_night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def bucket_opponent_strength(rating_diff: int) -> str:
    if rating_diff <= -100:
        return "opponent_100_plus_higher"
    if rating_diff < 0:
        return "opponent_slightly_higher"
    if rating_diff >= 100:
        return "opponent_100_plus_lower"
    if rating_diff > 0:
        return "opponent_slightly_lower"
    return "same_rating"

def plot_rating_progression(games):

    rating_by_day = (
        games.groupby(["date", "timeClass"], as_index=False)["userRating"]
        .mean()
        .rename(columns={"userRating": "avg_user_rating"})
        .sort_values(["timeClass", "date"])
    )

    rating_by_day["rolling_rating"] = (
        rating_by_day
        .set_index("date")
        .groupby("timeClass")["avg_user_rating"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .values
    )

    _, ax = plt.subplots()
    sns.lineplot(
        data=rating_by_day,
        x="date",
        y="rolling_rating",
        hue="timeClass",
        linewidth=2,
        ax=ax,
    )

    ax.set_title("Rating Progression by Time Control")
    ax.set_xlabel("Date")
    ax.set_ylabel("7-Day Rolling Average Rating")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return ax

def plot_games_per_week(games):
    games_per_week = (
        games.groupby(["week_start", "timeClass"], as_index=False)
        .size()
        .rename(columns={"size": "games_played"})
    )

    _, ax = plt.subplots()
    sns.lineplot(
        data=games_per_week,
        x="week_start",
        y="games_played",
        hue="timeClass",
        linewidth=2,
        ax=ax,
    )

    ax.set_title("Games Played Per Week by Time Control")
    ax.set_xlabel("Week Starting")
    ax.set_ylabel("Games Played")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return ax

def weakest_openings(games, color):
    min_games = DEFAULT_OPENING_MIN_GAMES
    filtered = games[games["userColor"] == color].copy()
    opening_summary = (
        filtered.groupby("opening_family", dropna=False)
        .agg(
            games=("gameId", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            draws=("is_draw", "sum"),
            win_rate=("is_win", "mean"),
            avg_rating_diff=("rating_diff", "mean"),
        )
        .reset_index()
    )
    opening_summary = opening_summary[opening_summary["games"] >= min_games].copy()
    opening_summary = opening_summary.sort_values(["win_rate", "games"], ascending=[True, False]).reset_index(drop=True)
    return opening_summary

import pandas as pd
import numpy as np
from scipy.stats import norm

def run_hypothesis_test(df):
    if df.empty:
        return pd.DataFrame()

    subset = df[
        (df["timeClass"].isin(["blitz", "rapid", "bullet"])) &
        (df["is_win"] + df["is_loss"] == 1)
    ]

    if subset.empty:
        return pd.DataFrame()

    white = subset[subset["userColor"] == "white"]
    black = subset[subset["userColor"] == "black"]

    white_wins = white["is_win"].sum()
    black_wins = black["is_win"].sum()

    white_total = len(white)
    black_total = len(black)

    if white_total == 0 or black_total == 0:
        return pd.DataFrame([{"error": "Not enough data"}])

    p1 = white_wins / white_total
    p2 = black_wins / black_total

    p_pool = (white_wins + black_wins) / (white_total + black_total)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/white_total + 1/black_total))

    z = (p1 - p2) / se

    p_value = 2 * (1 - norm.cdf(abs(z)))

    return pd.DataFrame([{
        "white_win_rate": p1,
        "black_win_rate": p2,
        "z_stat": z,
        "p_value": p_value,
        "significant_at_0_05": p_value < 0.05
    }])