from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations

import pandas as pd


NewsMetadata = Mapping[str, tuple[str, str]]


def build_news_metadata_lookup(news_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Build a lightweight lookup of (category, subcategory) by news_id."""
    required = {"news_id", "category", "subcategory"}
    missing = required.difference(news_df.columns)
    if missing:
        raise ValueError(f"news_df is missing required columns: {sorted(missing)}")

    lookup: dict[str, tuple[str, str]] = {}
    for row in news_df[["news_id", "category", "subcategory"]].drop_duplicates("news_id").itertuples(index=False):
        lookup[str(row.news_id)] = (str(row.category), str(row.subcategory))
    return lookup



def coverage_at_k(
    recommendations: Sequence[Sequence[str]],
    candidate_universe: set[str],
) -> float:
    """Catalog coverage@k = unique recommended items / recommendable item universe."""
    if not candidate_universe:
        return 0.0

    unique_recommended = set().union(*(set(rec) for rec in recommendations)) if recommendations else set()
    return float(len(unique_recommended) / len(candidate_universe))



def _pairwise_metadata_dissimilarity(
    left_item: str,
    right_item: str,
    news_metadata: NewsMetadata,
) -> float:
    """Simple metadata-based dissimilarity in [0, 1].

    - same subcategory -> 0.0
    - same category, different subcategory -> 0.5
    - different category -> 1.0
    Missing metadata falls back to category/subcategory strings being empty.
    """
    left_cat, left_sub = news_metadata.get(str(left_item), ("", ""))
    right_cat, right_sub = news_metadata.get(str(right_item), ("", ""))

    similarity = 0.0
    if left_cat and right_cat and left_cat == right_cat:
        similarity += 0.5
    if left_sub and right_sub and left_sub == right_sub:
        similarity += 0.5

    return 1.0 - similarity



def diversity_at_k(
    recommended_items: Sequence[str],
    news_metadata: NewsMetadata,
    k: int = 10,
) -> float:
    """Mean pairwise metadata dissimilarity of the top-k list."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    topk = [str(item) for item in recommended_items[:k]]
    if len(topk) < 2:
        return 0.0

    pairwise_scores = [
        _pairwise_metadata_dissimilarity(left, right, news_metadata)
        for left, right in combinations(topk, 2)
    ]
    return float(sum(pairwise_scores) / len(pairwise_scores)) if pairwise_scores else 0.0



def mean_diversity_at_k(
    all_recommendations: Sequence[Sequence[str]],
    news_metadata: NewsMetadata,
    k: int = 10,
) -> float:
    """Mean diversity@k across recommendation lists."""
    if not all_recommendations:
        return 0.0

    values = [diversity_at_k(rec, news_metadata, k=k) for rec in all_recommendations]
    return float(sum(values) / len(values)) if values else 0.0



def click_popularity(behaviour_train: pd.DataFrame) -> tuple[Counter[str], int]:
    """Count item popularity from clicked training impressions."""
    clicked_items: list[str] = []

    for candidates, labels in zip(behaviour_train["candidates"], behaviour_train["labels"]):
        clicked_items.extend(str(item) for item, label in zip(candidates, labels) if int(label) == 1)

    click_counts = Counter(clicked_items)
    total_clicks = len(clicked_items)
    return click_counts, total_clicks



def novelty_at_k(
    recommended_items: Sequence[str],
    click_counts: Counter[str],
    total_clicks: int,
    k: int = 10,
    catalog_size: int | None = None,
) -> float:
    """Novelty@k for one recommendation list using self-information.

    A higher value means the recommended items are rarer in the training clicks.
    Laplace smoothing avoids log(0).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    topk = [str(item) for item in recommended_items[:k]]
    if not topk:
        return 0.0

    num_items = int(catalog_size) if catalog_size is not None else len(click_counts)
    num_items = max(num_items, 1)
    denom = total_clicks + num_items

    scores = []
    for item in topk:
        count = click_counts.get(item, 0)
        p = (count + 1) / denom
        scores.append(-math.log2(p))

    return float(sum(scores) / len(scores))



def mean_novelty_at_k(
    all_recommendations: Sequence[Sequence[str]],
    click_counts: Counter[str],
    total_clicks: int,
    k: int = 10,
    catalog_size: int | None = None,
) -> float:
    """Mean novelty@k across recommendation lists."""
    if not all_recommendations:
        return 0.0

    values = [
        novelty_at_k(
            rec,
            click_counts=click_counts,
            total_clicks=total_clicks,
            k=k,
            catalog_size=catalog_size,
        )
        for rec in all_recommendations
    ]
    return float(sum(values) / len(values)) if values else 0.0
