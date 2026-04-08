"""Hybrid recommender combining EASE and TF-IDF.

Combines CF: EASE and feature-based: TF-IDF.
Popularity baseline fallback is handled within EASE/TF-IDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommenders.base import BaseRecommender
from src.recommenders.collaborative.ease import EASERecommender
from src.recommenders.content.tfidf import TfidfContentRecommender

from sklearn.metrics.pairwise import cosine_similarity


class HybridNewsRecommender(BaseRecommender):
    def __init__(
        self,
        ease_weight: float = 0.5,
        tfidf_weight: float = 0.5,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        l2: float = 500.0,
        max_items: int = 15000,
        min_item_support: int = 2,
    ):
        weights = np.array([ease_weight, tfidf_weight], dtype=np.float32)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Weights must be non-negative and sum to > 0.")
        weights /= weights.sum()
        self.ease_weight, self.tfidf_weight = float(weights[0]), float(weights[1])

        self.ease = EASERecommender(l2=l2, max_items=max_items, min_item_support=min_item_support)
        self.tfidf = TfidfContentRecommender(max_features=max_features, ngram_range=ngram_range)

        self.news_id_to_idx: dict[str, int] = {}

    def fit(
        self,
        news_df: pd.DataFrame,
        behaviors_df: pd.DataFrame,
        text_col: str = "text",
    ) -> "HybridNewsRecommender":
        self.ease.fit(behaviors_df)
        self.tfidf.fit(news_df, behaviors_df, text_col=text_col)
        self.news_id_to_idx = {str(nid): i for i, nid in enumerate(self.tfidf.news_index)}
        return self

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi <= lo:
            return np.zeros_like(scores)
        return ((scores - lo) / (hi - lo)).astype(np.float32)

    def score(
        self,
        user_id: str,
        candidates: list[str],
        history: list[str] | None = None,
    ) -> np.ndarray:
        candidates = [str(c) for c in candidates]
        if not candidates:
            return np.array([], dtype=np.float32)

        ease_scores = self.ease.score(user_id, candidates, history=history)

        # Score only the candidates, not the full news matrix
        hist = history or list(self.ease.user_history.get(user_id, []))
        profile = self.tfidf._profile_from_history(hist) if hist else None
        if profile is None:
            tfidf_scores = np.array(
                [self.tfidf.popularity.get(c, 0.0) for c in candidates], dtype=np.float32
            )
        else:
            cand_idxs = [self.tfidf.news_id_to_idx[c] for c in candidates if c in self.tfidf.news_id_to_idx]
            cand_pos  = [i for i, c in enumerate(candidates) if c in self.tfidf.news_id_to_idx]
            tfidf_scores = np.array(
                [self.tfidf.popularity.get(c, 0.0) for c in candidates], dtype=np.float32
            )
            if cand_idxs:
                sims = cosine_similarity(profile, self.tfidf.news_tfidf[cand_idxs]).ravel()
                for pos, sim in zip(cand_pos, sims):
                    tfidf_scores[pos] = float(sim)

        combined = (
            self.ease_weight  * self._minmax(ease_scores)
            + self.tfidf_weight * self._minmax(tfidf_scores)
        )
        return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    def recommend(
        self,
        user_id: str,
        candidates: list[str],
        k: int = 10,
        history: list[str] | None = None,
    ) -> list[str]:
        scores = self.score(user_id, candidates, history=history)
        return [candidates[i] for i in np.argsort(scores)[::-1][:k]]