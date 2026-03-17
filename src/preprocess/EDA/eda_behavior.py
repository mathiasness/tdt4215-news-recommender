"""EDA for MIND-small train split: behaviors, category stats, and entity embeddings."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = REPO_ROOT / "data" / "raw"
TRAIN_PATH = BASE_PATH / "MINDlarge_train"

BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def _read_tsv(path: Path, columns: list[str]) -> pd.DataFrame:
    path = path.resolve()

    print(f"Reading: {path}")
    print(f"Exists: {path.exists()}")
    print(f"Is file: {path.is_file()}")

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    try:
        df = pd.read_csv(
            str(path),
            sep="\t",
            header=None,
            names=columns,
            dtype=str,
            engine="python",
            encoding="utf-8",
            on_bad_lines="warn",
        )
        return df.fillna("")
    except Exception as e:
        print(f"Failed while reading: {path}")
        raise e

def _read_tsv(path: Path, columns: list[str], disable_quoting: bool = False) -> pd.DataFrame:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    read_kwargs = {
        "filepath_or_buffer": str(path),
        "sep": "\t",
        "header": None,
        "names": columns,
        "dtype": str,
        "encoding": "utf-8",
        "engine": "python",
        "on_bad_lines": "warn",
    }

    if disable_quoting:
        read_kwargs["quoting"] = csv.QUOTE_NONE

    df = pd.read_csv(**read_kwargs)
    return df.fillna("")

def read_behaviors_train() -> pd.DataFrame:
    path = TRAIN_PATH / "behaviors.tsv"
    return _read_tsv(path, BEHAVIORS_COLUMNS)


def read_news_train() -> pd.DataFrame:
    path = TRAIN_PATH / "news.tsv"
    return _read_tsv(path, NEWS_COLUMNS, disable_quoting=True)


def set_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible helper if caller loaded without explicit names."""
    if df.shape[1] == 5:
        df.columns = BEHAVIORS_COLUMNS
    return df


def _parse_entity_ids(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        entities = json.loads(raw)
    except json.JSONDecodeError:
        return []

    out: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        wikidata_id = entity.get("WikidataId")
        if isinstance(wikidata_id, str) and wikidata_id:
            out.append(wikidata_id)
    return out


def expand_impressions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[str, str, str, int]] = []
    for row in df.itertuples(index=False):
        imp_id = str(row.impression_id)
        user = str(row.user_id)

        for token in str(row.impressions).split():
            if "-" not in token:
                continue
            news_id, label = token.rsplit("-", 1)
            try:
                parsed_label = int(label)
            except ValueError:
                continue
            rows.append((imp_id, user, news_id, parsed_label))

    return pd.DataFrame(rows, columns=["impression_id", "user_id", "news_id", "label"])


# ============================================================
# A. Missing Values
# ============================================================

def check_missing(df: pd.DataFrame) -> None:
    print("\n===== MISSING VALUES =====")

    print("Null timestamps:", int(df["time"].isna().sum()))

    empty_history = df["history"].isna() | (df["history"] == "")
    print("Users with empty history:", int(empty_history.sum()))

    malformed = df["impressions"].apply(
        lambda x: any("-" not in token for token in str(x).split())
    )
    print("Malformed impressions rows:", int(malformed.sum()))


# ============================================================
# B. User Activity Analysis
# ============================================================

def user_activity(df: pd.DataFrame) -> None:
    print("\n===== USER ACTIVITY =====")

    impressions_per_user = df.groupby("user_id")["impression_id"].count()

    print("Users:", int(impressions_per_user.shape[0]))
    print("Mean impressions per user:", float(impressions_per_user.mean()))
    print("Median impressions per user:", float(impressions_per_user.median()))
    print("Max impressions (most active user):", int(impressions_per_user.max()))

    history_length = df["history"].fillna("").apply(lambda x: len(str(x).split()))
    print("\nHistory length stats:")
    print(history_length.describe())

    cold_start = int((history_length == 0).sum())
    print("Cold-start users (no history):", cold_start)


# ============================================================
# C. Click Distribution
# ============================================================

def click_analysis(df: pd.DataFrame) -> None:
    print("\n===== CLICK DISTRIBUTION =====")

    interactions = expand_impressions(df)
    total_samples = int(len(interactions))
    total_clicks = int(interactions["label"].sum())

    print("Total candidate samples:", total_samples)
    print("Total clicks:", total_clicks)
    print("CTR (Click Through Rate):", round(total_clicks / max(total_samples, 1), 5))

    print("\nClass distribution:")
    print(interactions["label"].value_counts(normalize=True))

    clicks_per_impression = interactions.groupby("impression_id")["label"].sum()

    print("\nAverage clicks per impression:", float(clicks_per_impression.mean()))
    print("Distribution of clicks per impression:")
    print(clicks_per_impression.describe())
    print(
        "Average candidates per impression:",
        float(interactions.groupby("impression_id").size().mean()),
    )

    print("\nClicks per impression distribution:")
    print(clicks_per_impression.value_counts().sort_index())

    print("\nCTR:")
    print(float(interactions["label"].mean()))


# ============================================================
# D. Category Analysis
# ============================================================

def category_analysis(behaviors_df: pd.DataFrame, news_df: pd.DataFrame) -> None:
    print("\n===== CATEGORY ANALYSIS =====")

    news_meta = news_df[["news_id", "category", "subcategory"]].copy()
    news_meta["category"] = news_meta["category"].replace("", "UNKNOWN")
    news_meta["subcategory"] = news_meta["subcategory"].replace("", "UNKNOWN")

    print("News items:", int(len(news_meta)))
    print("\nTop categories in news catalog:")
    print(news_meta["category"].value_counts().head(15))

    print("\nTop subcategories in news catalog:")
    print(news_meta["subcategory"].value_counts().head(20))

    interactions = expand_impressions(behaviors_df)
    interactions = interactions.merge(news_meta, on="news_id", how="left")
    interactions["category"] = interactions["category"].fillna("UNKNOWN")
    interactions["subcategory"] = interactions["subcategory"].fillna("UNKNOWN")

    missing_category = int((interactions["category"] == "UNKNOWN").sum())
    print("\nCandidate rows without matched category:", missing_category)

    category_perf = (
        interactions.groupby("category")["label"]
        .agg(shown="count", clicks="sum", ctr="mean")
        .sort_values("shown", ascending=False)
    )

    print("\nCategory performance by exposure (top 15 by shown):")
    print(category_perf.head(15))

    clicked = interactions[interactions["label"] == 1]
    print("\nTop clicked categories:")
    print(clicked["category"].value_counts().head(15))


# ============================================================
# E. Embedding Analysis
# ============================================================

def _scan_entity_embeddings(path: Path) -> tuple[dict[str, float | int | str], set[str]]:
    total_rows = 0
    parsed_rows = 0
    malformed_rows = 0
    duplicate_ids = 0
    zero_norm_rows = 0
    dim_counter: Counter[int] = Counter()
    entity_ids: set[str] = set()

    norm_sum = 0.0
    norm_min = float("inf")
    norm_max = 0.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                malformed_rows += 1
                continue

            entity_id = parts[0].strip()
            if not entity_id:
                malformed_rows += 1
                continue

            if entity_id in entity_ids:
                duplicate_ids += 1
            entity_ids.add(entity_id)

            try:
                vec = [float(x) for x in parts[1:] if x != ""]
            except ValueError:
                malformed_rows += 1
                continue

            if not vec:
                malformed_rows += 1
                continue

            parsed_rows += 1
            dim = len(vec)
            dim_counter[dim] += 1

            norm = math.sqrt(sum(v * v for v in vec))
            norm_sum += norm
            norm_min = min(norm_min, norm)
            norm_max = max(norm_max, norm)
            if norm == 0.0:
                zero_norm_rows += 1

    dominant_dim = None
    if dim_counter:
        dominant_dim = dim_counter.most_common(1)[0][0]

    stats: dict[str, float | int | str] = {
        "total_rows": total_rows,
        "parsed_rows": parsed_rows,
        "malformed_rows": malformed_rows,
        "unique_entity_ids": len(entity_ids),
        "duplicate_entity_ids": duplicate_ids,
        "num_dimensions_found": len(dim_counter),
        "dominant_dimension": "N/A" if dominant_dim is None else dominant_dim,
        "mean_vector_norm": 0.0 if parsed_rows == 0 else norm_sum / parsed_rows,
        "min_vector_norm": 0.0 if parsed_rows == 0 else norm_min,
        "max_vector_norm": 0.0 if parsed_rows == 0 else norm_max,
        "zero_norm_vectors": zero_norm_rows,
    }
    return stats, entity_ids


def embedding_analysis(news_df: pd.DataFrame) -> None:
    print("\n===== ENTITY EMBEDDING ANALYSIS =====")

    emb_path = TRAIN_PATH / "entity_embedding.vec"
    if not emb_path.exists():
        print("No embedding file found at:", emb_path)
        return

    stats, embedded_entity_ids = _scan_entity_embeddings(emb_path)
    print("Embedding file:", emb_path)
    for key, value in stats.items():
        print(f"{key}: {value}")

    news_entities = []
    for row in news_df.itertuples(index=False):
        title_ids = _parse_entity_ids(str(row.title_entities))
        abstract_ids = _parse_entity_ids(str(row.abstract_entities))
        merged = list(dict.fromkeys(title_ids + abstract_ids))
        news_entities.append((str(row.news_id), str(row.category), merged))

    entities_df = pd.DataFrame(news_entities, columns=["news_id", "category", "entity_ids"])
    entities_df["category"] = entities_df["category"].replace("", "UNKNOWN")
    entities_df["num_entities"] = entities_df["entity_ids"].apply(len)
    entities_df["num_embedded_entities"] = entities_df["entity_ids"].apply(
        lambda ids: sum(1 for eid in ids if eid in embedded_entity_ids)
    )
    entities_df["has_entities"] = entities_df["num_entities"] > 0
    entities_df["has_embedding_covered_entity"] = entities_df["num_embedded_entities"] > 0

    all_referenced_entities = set()
    for ids in entities_df["entity_ids"]:
        all_referenced_entities.update(ids)
    covered_entities = all_referenced_entities & embedded_entity_ids

    print("\nNews-level entity coverage:")
    print("News items:", int(len(entities_df)))
    print("News with >=1 entity:", int(entities_df["has_entities"].sum()))
    print(
        "News with >=1 entity covered by embedding:",
        int(entities_df["has_embedding_covered_entity"].sum()),
    )
    print("Unique referenced entities in news:", int(len(all_referenced_entities)))
    print("Referenced entities found in embeddings:", int(len(covered_entities)))
    print(
        "Entity-level coverage ratio:",
        round(len(covered_entities) / max(len(all_referenced_entities), 1), 5),
    )

    category_coverage = (
        entities_df.groupby("category")
        .agg(
            news=("news_id", "count"),
            with_entities=("has_entities", "sum"),
            with_embedding=("has_embedding_covered_entity", "sum"),
        )
        .sort_values("news", ascending=False)
    )
    category_coverage["coverage_ratio"] = category_coverage["with_embedding"] / category_coverage[
        "news"
    ].clip(lower=1)

    print("\nEmbedding coverage by category (top 15 by news count):")
    print(category_coverage.head(15))

    missing_counts = Counter()
    for ids in entities_df["entity_ids"]:
        missing_counts.update(eid for eid in ids if eid not in embedded_entity_ids)
    if missing_counts:
        print("\nTop missing entities (not found in embedding file):")
        for entity_id, count in missing_counts.most_common(20):
            print(f"{entity_id}: {count}")
    else:
        print("\nNo missing entity IDs from news were found.")


def run() -> None:
    behaviors_df = set_columns(read_behaviors_train())
    news_df = read_news_train()

    check_missing(behaviors_df)
    user_activity(behaviors_df)
    click_analysis(behaviors_df)
    category_analysis(behaviors_df, news_df)
    embedding_analysis(news_df)


if __name__ == "__main__":
    run()
