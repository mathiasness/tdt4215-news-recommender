"""Utilities for reading and caching MIND data splits."""

import json
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SPLIT_DIRS = {
    "train": "MINDsmall_train",
    "test": "MINDsmall_dev",
}
NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]
BEHAVIORS_COLUMNS = [
    "impression_id", "user_id", "time", "history", "impressions"
]
CANONICAL_NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title", "abstract"
]
CANONICAL_BEHAVIORS_COLUMNS = [
    "impression_id", "user_id", "time",
    "history", "candidates", "labels",
]

### Parsing helpers ###

def _parse_history(value: Any) -> List[str]:
    """parse 'n1 n2 ' -> ['n1', 'n2']"""
    if not isinstance(value, str) or not value.strip(): return []
    return value.strip().split()


def _parse_impressions(value: Any) -> Tuple[List[str], List[int]]:
    """parse 'n1-0 n2-1 ' -> (['n1', 'n2'], [0, 1])"""
    if not isinstance(value, str) or not value.strip(): return [], []
    candidates: List[str] = []
    labels: List[int] = []
    for item in value.split():
        if "-" in item:
            news_id, label = item.rsplit("-", 1)
            candidates.append(news_id)
            try:
                labels.append(int(label))
            except ValueError:
                labels.append(0)
        else:
            candidates.append(item)
            labels.append(0)
    return (candidates, labels)


def _serialize_list_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Serialize list columns to JSON strings for CSV storage."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].apply(json.dumps)
    return out


def _deserialize_list_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Deserialize JSON string columns back to lists."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else []
        )
    return out

### Main parsing functions ###

def parse_news(news_path: Path):
    df = pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLUMNS, dtype=str).fillna("")
    news_df = df[CANONICAL_NEWS_COLUMNS]
    return news_df


def parse_behaviors(behaviors_path: Path):
    df = pd.read_csv(behaviors_path, sep="\t", header=None, names=BEHAVIORS_COLUMNS, dtype=str).fillna("")
    df["history"] = df["history"].apply(_parse_history)

    parsed = df["impressions"].apply(_parse_impressions)
    df["candidates"] = parsed.apply(lambda x: x[0])
    df["labels"] = parsed.apply(lambda x: x[1])

    behaviors_df = df[CANONICAL_BEHAVIORS_COLUMNS].copy()
    return behaviors_df


### splitting and caching ###

def get_split_dir(split: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    if split not in SPLIT_DIRS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(SPLIT_DIRS)}")
    return Path(data_dir) / SPLIT_DIRS[split]


def build_processed_split(
    split: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse and store a MIND split."""

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    split_dir = get_split_dir(split, data_dir)
    
    news_path = split_dir / "news.tsv"
    behaviors_path = split_dir / "behaviors.tsv"

    news_df = parse_news(news_path)
    behaviors_df = parse_behaviors(behaviors_path)

    news_out = processed_dir / f"{split}_news.csv"
    behaviors_out = processed_dir / f"{split}_behaviors.csv"

    news_df.to_csv(news_out, index=False)
    _serialize_list_columns(behaviors_df, ["history", "candidates", "labels"]).to_csv(behaviors_out, index=False)
    
    return news_df, behaviors_df


def load_processed_split(
    split: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a cached MIND split."""

    processed_dir = Path(processed_dir)
    news_path = processed_dir / f"{split}_news.csv"
    behaviors_path = processed_dir / f"{split}_behaviors.csv"

    news_df = pd.read_csv(news_path, dtype=str).fillna("")
    behaviors_df = pd.read_csv(behaviors_path, dtype=str).fillna("")
    behaviors_df = _deserialize_list_columns(behaviors_df, ["history", "candidates", "labels"])
    behaviors_df["labels"] = behaviors_df["labels"].apply(lambda xs: [int(x) for x in xs])
    
    return news_df[CANONICAL_NEWS_COLUMNS], behaviors_df[CANONICAL_BEHAVIORS_COLUMNS]
