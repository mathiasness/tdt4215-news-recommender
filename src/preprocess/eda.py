"""
EDA focused on behaviours.tsv (MIND-small train split).
"""

from mind_reader import read_behaviors_train
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def set_columns(df):
    """
    Ensure correct column names regardless of header settings.
    """
    if df.shape[1] == 5:
        df.columns = ["impression_id", "user_id", "time", "history", "impressions"]
    return df


def expand_impressions(df):
    rows = []
    for _, r in df.iterrows():
        imp_id = r["impression_id"]
        user = r["user_id"]

        for token in str(r["impressions"]).split():
            if "-" not in token:
                continue
            news_id, label = token.rsplit("-", 1)
            try:
                label = int(label)
            except:
                continue
            rows.append((imp_id, user, news_id, label))

    return pd.DataFrame(
        rows,
        columns=["impression_id", "user_id", "news_id", "label"]
    )


# ============================================================
# A. Missing Values
# ============================================================

def check_missing(df):
    print("\n===== MISSING VALUES =====")

    print("Null timestamps:", df["time"].isna().sum())

    empty_history = df["history"].isna() | (df["history"] == "")
    print("Users with empty history:", empty_history.sum())

    malformed = df["impressions"].apply(
        lambda x: any("-" not in token for token in str(x).split())
    )
    print("Malformed impressions rows:", malformed.sum())


# ============================================================
# B. User Activity Analysis
# ============================================================

def user_activity(df):
    print("\n===== USER ACTIVITY =====")

    impressions_per_user = df.groupby("user_id")["impression_id"].count()

    print("Users:", impressions_per_user.shape[0])
    print("Mean impressions per user:", impressions_per_user.mean())
    print("Median impressions per user:", impressions_per_user.median())
    print("Max impressions (most active user):", impressions_per_user.max())

    # History length
    history_length = df["history"].fillna("").apply(lambda x: len(str(x).split()))
    print("\nHistory length stats:")
    print(history_length.describe())

    # Cold start users
    cold_start = (history_length == 0).sum()
    print("Cold-start users (no history):", cold_start)


# ============================================================
# C. Click Distribution
# ============================================================

def click_analysis(df):
    print("\n===== CLICK DISTRIBUTION =====")

    interactions = expand_impressions(df)

    total_clicks = interactions["label"].sum()
    total_samples = len(interactions)

    print("Total candidate samples:", total_samples)
    print("Total clicks:", total_clicks)

    ctr = total_clicks / total_samples
    print("CTR (Click Through Rate):", round(ctr, 5))

    print("\nClass distribution:")
    print(interactions["label"].value_counts(normalize=True))

    # Clicks per impression
    clicks_per_impression = interactions.groupby("impression_id")["label"].sum()

    print("\nAverage clicks per impression:", clicks_per_impression.mean())
    print("Distribution of clicks per impression:")
    print(clicks_per_impression.describe())


# ============================================================
# Run Full behaviors EDA
# ============================================================

def run():
    df = read_behaviors_train()
    df = set_columns(df)

    check_missing(df)
    user_activity(df)
    click_analysis(df)


if __name__ == "__main__":
    run()