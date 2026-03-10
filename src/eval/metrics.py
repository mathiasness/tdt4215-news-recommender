"""Ranking metrics for binary relevance."""

import numpy as np

def _ranked_labels(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    return labels[order]


def ndcg_at_k(labels, scores, k: int) -> float:
    """NDCG@k for binaries"""
    ranked = _ranked_labels(labels, scores)
    if ranked.size == 0: return 0.0
    k = min(k, ranked.size)

    dcg = np.sum(ranked[:k] / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(np.sort(ranked)[::-1][:k] / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(labels, scores, k: int) -> float:
    """MRR@k for binaries"""
    ranked = _ranked_labels(labels, scores)
    if ranked.size == 0: return 0.0
    k = min(k, ranked.size)

    topk = ranked[:k]
    hits = np.where(topk == 1)[0]
    return 1.0 / (hits[0] + 1) if hits.size > 0 else 0.0


def recall_at_k(labels, scores, k: int) -> float:
    """Recall@k for binaries"""
    ranked = _ranked_labels(labels, scores)
    if ranked.size == 0: return 0.0
    k = min(k, ranked.size)

    relevant = np.sum(ranked[:k])
    total_relevant = np.sum(labels)
    return relevant / total_relevant if total_relevant > 0 else 0.0
