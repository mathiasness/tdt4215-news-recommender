import math
from typing import Counter, List, Sequence

import pandas as pd


def coverage_at_k(
    recommendations : List[List[str]],
    candidates : set[str],
) -> float:
    """Coverage@k = unique recommended items / unique candidate universe."""
    if not candidates:
        return 0.0

    unique_recommended_items = set().union(*recommendations) if recommendations else set()
    return len(unique_recommended_items) / len(candidates)


def diversity_at_k():
    ...


def click_popularity(behaviour_train: pd.DataFrame) -> tuple[Counter[str], int]:
    """Click popularity. divide by total clicks instead of #shown (CTR)"""
    click_counts = Counter()

    clicked_items = []

    for candidates, labels in zip(behaviour_train["candidates"], behaviour_train["labels"]):
        clicked_items.extend(item for item, label in zip(candidates, labels) if int(label) == 1)

    click_counts = Counter(clicked_items)
    total_clicks = len(clicked_items)
    return click_counts, total_clicks


def novelty_at_k(
    recommended_items: Sequence[str],
    click_counts: Counter[str],
    total_clicks: int,
    k: int = 10,
) -> float:
    """
    Novelty@K for one recommendation list.
    Laplace smoothing to avoid log(0).
    """
    topk = recommended_items[:k]
    if not topk:
        return 0.0

    num_items = max(len(click_counts), 1)

    denom = total_clicks + num_items  # Laplace smoothing
    scores = []

    for item in topk:
        count = click_counts.get(item, 0)
        p = (count + 1) / denom
        scores.append(-math.log2(p))

    return sum(scores) / len(scores)


def mean_novelty_at_k(
    all_recommendations: Sequence[Sequence[str]],
    click_counts: Counter[str],
    total_clicks: int,
    k: int = 10,
) -> float:
    """
    mean novelty@K across
    all_recommendations: list of top-K item lists
    """
    if not all_recommendations:
        return 0.0

    vals = [
        novelty_at_k(rec, click_counts, total_clicks, k=k)
        for rec in all_recommendations
    ]
    return sum(vals) / len(vals)