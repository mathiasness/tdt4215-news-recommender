"""Ranking metrics for binary relevance."""

from __future__ import annotations

import numpy as np


def _prepare_labels_and_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce 1D label/score arrays used by ranking metrics."""
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)

    if labels_arr.shape != scores_arr.shape:
        raise ValueError(
            f"labels and scores must have the same shape, got {labels_arr.shape} vs {scores_arr.shape}."
        )

    # Keep metric behavior deterministic even if a model returns NaN/inf.
    scores_arr = np.nan_to_num(scores_arr, nan=-1.0e12, posinf=1.0e12, neginf=-1.0e12)
    return labels_arr, scores_arr


def _ranked_labels(labels, scores) -> np.ndarray:
    """Return labels sorted by descending score using a stable sort for ties."""
    labels_arr, scores_arr = _prepare_labels_and_scores(labels, scores)
    order = np.argsort(-scores_arr, kind="mergesort")
    return labels_arr[order]


def ndcg_at_k(labels, scores, k: int) -> float:
    """Compute nDCG@k for binary relevance labels."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    ranked = _ranked_labels(labels, scores)
    if ranked.size == 0:
        return 0.0

    k = min(int(k), ranked.size)
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))

    dcg = np.sum(ranked[:k] / discounts, dtype=np.float64)
    ideal = np.sort(ranked)[::-1][:k]
    idcg = np.sum(ideal / discounts, dtype=np.float64)
    return float(dcg / idcg) if idcg > 0.0 else 0.0



def mrr_at_k(labels, scores, k: int) -> float:
    """Compute MRR@k for binary relevance labels."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    ranked = _ranked_labels(labels, scores)
    if ranked.size == 0:
        return 0.0

    topk = ranked[: min(int(k), ranked.size)]
    hits = np.flatnonzero(topk > 0)
    return float(1.0 / (hits[0] + 1)) if hits.size > 0 else 0.0



def recall_at_k(labels, scores, k: int) -> float:
    """Compute Recall@k for binary relevance labels."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    ranked = _ranked_labels(labels, scores)
    if ranked.size == 0:
        return 0.0

    total_relevant = int(np.sum(np.asarray(labels) > 0))
    if total_relevant == 0:
        return 0.0

    k = min(int(k), ranked.size)
    retrieved_relevant = int(np.sum(ranked[:k] > 0))
    return float(retrieved_relevant / total_relevant)
