import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RAW_DATA_FILENAME = "imsogarb69_all_2604212016.csv"
GAMES_FILENAME = "games_clean.csv"
CALENDAR_FILENAME = "calendar_table.csv"
DAILY_SUMMARY_FILENAME = "daily_summary.csv"
NUMERIC_FEATURES = [
    "userRating",
    "opponentRating",
    "rating_diff",
    "games_played_so_far_today",
    "recent_win_rate_5",
]
TIME_CONTROLS = ("blitz", "rapid", "bullet")
DRAW_RESULTS = {"agreed", "stalemate", "insufficient", "repetition", "timevsinsufficient"}
TIMECLASS_COLOR = {"bullet": "red", "blitz": "blue", "rapid": "green"}
DEFAULT_OPENING_MIN_GAMES = 8
DEFAULT_RANDOM_STATE = 16


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
    active_days = int((daily["games_played"] > 0).sum())
    summary_records = [
        {"metric": "Games analyzed", "value": len(games)},
        {"metric": "Calendar days sampled", "value": daily["date"].nunique()},
        {"metric": "Active playing days", "value": active_days},
        {"metric": "First date", "value": games["date"].min().date().isoformat()},
        {"metric": "Last date", "value": games["date"].max().date().isoformat()},
        {"metric": "Overall win rate", "value": round(float(games["is_win"].mean()), 4)},
        {"metric": "Average games per sampled day", "value": round(float(daily["games_played"].mean()), 2)},
        {
            "metric": "Average games per active day",
            "value": round(float(daily.loc[daily["games_played"] > 0, "games_played"].mean()), 2),
        },
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
        games[col] = games[col].astype("string").str.strip()
    games = games.replace("", np.nan)
    games = games.rename(columns={"outcome": "move_count", "moveCount" : "terminal_outcome"})
    games = games[games["timeClass"].isin(TIME_CONTROLS)].copy()

    numeric_columns = [
        "userAccuracy",
        "opponentAccuracy",
        "userRating",
        "opponentRating",
        "move_count",
    ]

    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    games["date"] = pd.to_datetime(games["date"], format="%Y.%m.%d", errors="coerce")
    games = games.dropna(subset=["date", "startTime", "endTime", "userRating", "opponentRating"])
    games["start_datetime_et"] = pd.to_datetime(
        games["date"].dt.strftime("%Y-%m-%d") + " " + games["startTime"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    games["end_datetime_et"] = pd.to_datetime(
        games["date"].dt.strftime("%Y-%m-%d") + " " + games["endTime"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    crossed_midnight = games["end_datetime_et"] < games["start_datetime_et"]
    games.loc[crossed_midnight, "end_datetime_et"] = (
        games.loc[crossed_midnight, "end_datetime_et"] + pd.Timedelta(days=1)
    )
    games = games.dropna(subset=["start_datetime_et", "end_datetime_et"])

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
    games["recent_win_rate_5"] = (
        games["is_win"]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
        .fillna(0.5)
    )
    games = add_calendar_features(games)
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


def add_calendar_features(games):
    games = games.copy()
    games["weekday_num"] = games["date"].dt.weekday
    games["weekday"] = games["date"].dt.day_name()
    games["is_weekend"] = games["weekday_num"].isin([5, 6]).astype(int)
    games["month"] = games["date"].dt.month
    games["month_name"] = games["date"].dt.month_name()
    games["year"] = games["date"].dt.year
    games["week_start"] = games["date"] - pd.to_timedelta(games["weekday_num"], unit="D")
    return games


def build_calendar_table(games):
    date_range = pd.date_range(games["date"].min(), games["date"].max(), freq="D")
    calendar = pd.DataFrame({"date": date_range})
    calendar["weekday_num"] = calendar["date"].dt.weekday
    calendar["weekday"] = calendar["date"].dt.day_name()
    calendar["is_weekend"] = calendar["weekday_num"].isin([5, 6]).astype(int)
    calendar["month"] = calendar["date"].dt.month
    calendar["month_name"] = calendar["date"].dt.month_name()
    calendar["year"] = calendar["date"].dt.year
    calendar["week_start"] = calendar["date"] - pd.to_timedelta(calendar["weekday_num"], unit="D")
    return calendar


def build_daily_summary(games, calendar):
    active_summary = (
        games.groupby("date")
        .agg(
            games_played=("gameId", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            draws=("is_draw", "sum"),
            avg_user_rating=("userRating", "mean"),
            avg_opponent_rating=("opponentRating", "mean"),
            avg_rating_diff=("rating_diff", "mean"),
            first_game_hour_et=("start_hour_et", "min"),
            last_game_hour_et=("start_hour_et", "max"),
        )
        .reset_index()
    )
    time_counts = (
        games.pivot_table(index="date", columns="timeClass", values="gameId", aggfunc="count", fill_value=0)
        .rename(columns={"blitz": "blitz_games", "rapid": "rapid_games", "bullet": "bullet_games"})
        .reset_index()
    )
    daily = calendar.merge(active_summary, on="date", how="left").merge(time_counts, on="date", how="left")

    count_columns = [
        "games_played",
        "wins",
        "losses",
        "draws",
        "blitz_games",
        "rapid_games",
        "bullet_games",
    ]
    for column in count_columns:
        daily[column] = daily[column].fillna(0).astype(int)

    daily["active_game_day"] = (daily["games_played"] > 0).astype(int)
    daily["win_rate"] = np.where(
        daily["games_played"] > 0,
        daily["wins"] / daily["games_played"],
        0.0,
    )
    daily[["avg_user_rating", "avg_opponent_rating", "avg_rating_diff"]] = (
        daily[["avg_user_rating", "avg_opponent_rating", "avg_rating_diff"]].ffill().bfill()
    )
    daily[["first_game_hour_et", "last_game_hour_et"]] = (
        daily[["first_game_hour_et", "last_game_hour_et"]].fillna(-1).astype(int)
    )

    ordered_columns = [
        "date",
        "games_played",
        "active_game_day",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "blitz_games",
        "rapid_games",
        "bullet_games",
        "avg_user_rating",
        "avg_opponent_rating",
        "avg_rating_diff",
        "first_game_hour_et",
        "last_game_hour_et",
        "weekday",
        "is_weekend",
        "month",
        "month_name",
        "year",
        "week_start",
    ]
    return daily[ordered_columns]


def validate_join_integrity(games, calendar, daily):
    game_dates = set(games["date"])
    calendar_dates = set(calendar["date"])
    daily_dates = set(daily["date"])
    result = pd.DataFrame(
        [
            {
                "check": "game dates included in calendar",
                "passed": game_dates.issubset(calendar_dates),
                "details": len(game_dates - calendar_dates),
            },
            {
                "check": "daily dates match calendar dates",
                "passed": daily_dates == calendar_dates,
                "details": len(daily_dates.symmetric_difference(calendar_dates)),
            },
            {
                "check": "daily table has one row per calendar day",
                "passed": daily["date"].nunique() == len(calendar),
                "details": len(daily),
            },
        ]
    )
    print("Join Integrity Checks:")
    print(result)
    print()
    return result


def save_prepared_tables():
    raw_games = pd.read_csv(RAW_DATA_FILENAME)
    games = clean_games_data(raw_games)
    calendar = build_calendar_table(games)
    daily = build_daily_summary(games, calendar)
    games.to_csv(GAMES_FILENAME, index=False)
    calendar.to_csv(CALENDAR_FILENAME, index=False)
    daily.to_csv(DAILY_SUMMARY_FILENAME, index=False)
    return games, calendar, daily


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
        palette=TIMECLASS_COLOR,
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
        palette=TIMECLASS_COLOR,
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

def run_hypothesis_test(df):
    """Two-proportion z-test for White vs. Black win rate."""
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
        "test": "White vs Black decisive-game win rate",
        "white_wins": int(white_wins),
        "white_total": int(white_total),
        "black_wins": int(black_wins),
        "black_total": int(black_total),
        "white_win_rate": p1,
        "black_win_rate": p2,
        "z_stat": z,
        "p_value": p_value,
        "significant_at_0_05": p_value < 0.05
    }])


def run_time_control_chi_square_test(games):
    subset = games[games["result_group"].isin(["win", "loss"])].copy()
    observed = pd.crosstab(subset["timeClass"], subset["result_group"])
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    summary = pd.DataFrame(
        [
            {
                "test": "time control vs win/loss",
                "chi2_stat": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "significant_at_0_05": p_value < 0.05,
            }
        ]
    )
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    print("Observed Counts:")
    print(observed)
    print()
    print("Chi-Square Test Summary:")
    print(summary)
    print()
    return {"summary": summary, "observed": observed, "expected": expected_df}


def plot_win_rate_by_time_control(games):
    summary = games.groupby("timeClass")["is_win"].mean().reset_index()
    _, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="timeClass",
        y="is_win",
        ax=ax,
    )
    ax.set_title("Win Rate by Time Control")
    ax.set_xlabel("Time Control")
    ax.set_xticks(range(len(summary["timeClass"])))
    ax.set_xticklabels(summary["timeClass"], rotation=35, ha="right")
    ax.set_ylabel("Win Rate")
    plt.tight_layout()
    return ax

def plot_win_rate_by_color(games):
    summary = games.groupby("userColor")["is_win"].mean().reset_index()
    _, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="userColor",
        y="is_win",
        ax=ax,
    )
    ax.set_title("Win Rate by Color")
    ax.set_xlabel("Color")
    ax.set_xticks(range(len(summary["userColor"])))
    ax.set_xticklabels(summary["userColor"], rotation=35, ha="right")
    ax.set_ylabel("Win Rate")
    plt.tight_layout()
    return ax


def plot_win_rate_by_opponent_strength(games):
    summary = games.groupby("opponent_strength_bucket")["is_win"].mean().reset_index()
    order = [
        "opponent_100_plus_higher",
        "opponent_slightly_higher",
        "same_rating",
        "opponent_slightly_lower",
        "opponent_100_plus_lower",
    ]
    _, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="opponent_strength_bucket",
        y="is_win",
        order=order,
        ax=ax,
    )
    ax.set_title("Win Rate by Opponent Strength Bucket")
    ax.set_xlabel("Opponent Strength Bucket")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Win Rate")
    plt.tight_layout()
    return ax

def plot_win_rate_by_time_of_day(games):
    summary = games.groupby("hour_bucket_et")["is_win"].mean().reset_index()
    order = ["late_night", "morning", "afternoon", "evening"]
    _, ax = plt.subplots()
    sns.barplot(
        data=summary,
        x="hour_bucket_et",
        y="is_win",
        order=order,
        ax=ax,
    )
    ax.set_title("Win Rate by Time of Day (ET)")
    ax.set_xlabel("Time of Day")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Win Rate")
    plt.tight_layout()
    return ax

def plot_class_distribution(games):
    class_counts = games["result_group"].value_counts().reset_index() 
    class_counts.columns = ["outcome", "number_of_games"]

    class_counts = class_counts[class_counts["outcome"].isin(["win", "loss"])]
    _, ax = plt.subplots()
    sns.barplot(
        data=class_counts,
        x="outcome",
        y="number_of_games",
        ax=ax,
    )
    ax.set_title("Class Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return ax


def _prepare_classification_data(games):
    clf_data = games[games["result_group"].isin(["win", "loss"])].copy()
    clf_data["target"] = (clf_data["result_group"] == "win").astype(int)
    for column in NUMERIC_FEATURES:
        clf_data[column] = pd.to_numeric(clf_data[column], errors="coerce")
        clf_data[column] = clf_data[column].fillna(clf_data[column].median())

    feature_cols = NUMERIC_FEATURES
    X = clf_data[feature_cols]
    y = clf_data["target"]
    return X, y, feature_cols


def class_label_distribution(games):
    subset = games[games["result_group"].isin(["win", "loss"])]
    counts = subset["result_group"].value_counts()
    pcts = subset["result_group"].value_counts(normalize=True).round(4) * 100
    dist = pd.DataFrame({"count": counts, "percent": pcts})
    print("Class Label Distribution (win vs loss, draws excluded):")
    print(dist)
    print()
    return dist


def majority_class_baseline(games, print_result=True):
    subset = games[games["result_group"].isin(["win", "loss"])]
    baseline = float(subset["result_group"].value_counts(normalize=True).max())
    if print_result:
        print(f"Majority-Class Baseline Accuracy: {baseline:.4f}")
        print()
    return baseline


def _classification_split(games):
    X, y, feature_cols = _prepare_classification_data(games)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_cols


def _evaluate_classifier(name, model, X_test, y_test, baseline):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["loss", "win"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        index=["actual_loss", "actual_win"],
        columns=["predicted_loss", "predicted_win"],
    )
    print(name)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Majority baseline: {baseline:.4f}")
    print("  Confusion matrix:")
    print(confusion_df)
    print()
    return {
        "accuracy": acc,
        "baseline_accuracy": baseline,
        "report": report_df,
        "confusion_matrix": confusion_df,
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
    }


def run_knn_classifier(games, k=5):
    X_train, X_test, y_train, y_test, _ = _classification_split(games)
    baseline = majority_class_baseline(games, print_result=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train_scaled, y_train)
    results = _evaluate_classifier(
        f"kNN Classifier (neighbors={k})",
        knn,
        X_test_scaled,
        y_test,
        baseline,
    )
    results["scaler"] = scaler
    results["feature_cols"] = NUMERIC_FEATURES
    return results


def run_decision_tree_classifier(games, max_depth=10):
    X_train, X_test, y_train, y_test, feature_cols = _classification_split(games)
    baseline = majority_class_baseline(games, print_result=False)
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        random_state=DEFAULT_RANDOM_STATE,
    )
    tree.fit(X_train, y_train)
    results = _evaluate_classifier(
        f"Decision Tree Classifier (max_depth={max_depth})",
        tree,
        X_test,
        y_test,
        baseline,
    )

    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": tree.feature_importances_}
    ).sort_values("importance", ascending=False)

    results["importances"] = importances
    results["feature_cols"] = feature_cols
    return results


def plot_classifier_comparison(knn_results, dt_results):
    data = pd.DataFrame(
        {
            "classifier": ["Majority Baseline", "kNN", "Decision Tree"],
            "accuracy": [
                knn_results["baseline_accuracy"],
                knn_results["accuracy"],
                dt_results["accuracy"],
            ],
        }
    )
    _, ax = plt.subplots()
    sns.barplot(data=data, x="classifier", y="accuracy", ax=ax)
    ax.set_title("Classifier Accuracy Comparison")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return ax

def plot_decision_tree(dt_results, max_depth_display=3):
    dt = dt_results["model"]
    feature_cols = dt_results["feature_cols"]

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt,
        feature_names=feature_cols,
        class_names=["loss", "win"],
        filled=True,
        rounded=True,
        max_depth=max_depth_display,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree (first 3 levels)")
    plt.tight_layout()
    return ax
