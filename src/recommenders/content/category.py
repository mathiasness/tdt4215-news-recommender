"""Category-profile content recommender.

Represents user interest as a normalised distribution over news categories
and subcategories derived from click history. Candidates are scored by how
well their category/subcategory matches the user's interest profile,
weighted by an IDF correction that downweights ubiquitous categories
(e.g. "news", "sports") and rewards matches on rarer ones.

Conceptually orthogonal to TF-IDF: uses article metadata rather than text,
making it robust to vocabulary mismatch but limited by the coarseness of the
category taxonomy.

Falls back to global popularity when no usable history is available.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from src.recommenders.base import BaseRecommender


class CategoryRecommender(BaseRecommender):
    def __init__(
        self,
        subcategory_weight: float = 0.5,
    ):
        # Subcategory is a finer-grained signal but noisier — weighted below 1.
        self.subcategory_weight = subcategory_weight

        self.news_category: dict[str, str] = {}     # news_id -> category
        self.news_subcategory: dict[str, str] = {}  # news_id -> subcategory
        self.category_idf: dict[tuple, float] = {}  # ("cat"|"sub", value) -> idf
        self.popularity: pd.Series = pd.Series(dtype=np.float32)

    def fit(
        self,
        news_df: pd.DataFrame,
        behaviors_df: pd.DataFrame,
        text_col: str = "text",
    ) -> "CategoryRecommender":
        del text_col

        for row in news_df.itertuples(index=False):
            nid = str(row.news_id)
            self.news_category[nid] = str(getattr(row, "category", "") or "")
            self.news_subcategory[nid] = str(getattr(row, "subcategory", "") or "")

        # IDF over the news corpus: log(N / df) where df = number of articles
        # in that category. Penalises dominant categories like "news"/"sports"
        # and rewards matches on rare ones.
        N = len(news_df)
        cat_counts: Counter = Counter(self.news_category.values())
        sub_counts: Counter = Counter(self.news_subcategory.values())
        self.category_idf = {}
        for cat, df in cat_counts.items():
            if cat:
                self.category_idf[("cat", cat)] = math.log((N + 1) / (df + 1)) + 1.0
        for sub, df in sub_counts.items():
            if sub:
                self.category_idf[("sub", sub)] = math.log((N + 1) / (df + 1)) + 1.0

        exploded = behaviors_df[["candidates", "labels"]].explode(["candidates", "labels"])
        clicked = exploded.loc[exploded["labels"].astype(int) == 1, "candidates"].astype(str)
        self.popularity = clicked.value_counts().astype(np.float32)

        return self

    def _build_profile(self, history: list[str]) -> dict[tuple, float]:
        """IDF-weighted category interest distribution over click history."""
        counts: Counter = Counter()
        for nid in history:
            cat = self.news_category.get(nid)
            sub = self.news_subcategory.get(nid)
            if cat:
                counts[("cat", cat)] += 1.0
            if sub:
                counts[("sub", sub)] += self.subcategory_weight

        # Weight each count by IDF before normalising
        weighted = {
            k: v * self.category_idf.get(k, 1.0)
            for k, v in counts.items()
        }
        total = sum(weighted.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in weighted.items()}

    def score(
        self,
        user_id: str,
        candidates: list[str],
        history: list[str] | None = None,
    ) -> np.ndarray:
        candidates = [str(c) for c in candidates]
        if not candidates:
            return np.array([], dtype=np.float32)

        history = history or []
        profile = self._build_profile(history)

        if not profile:
            return np.array(
                [self.popularity.get(c, 0.0) for c in candidates], dtype=np.float32
            )

        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, c in enumerate(candidates):
            cat = self.news_category.get(c)
            sub = self.news_subcategory.get(c)
            s = 0.0
            if cat:
                s += profile.get(("cat", cat), 0.0)
            if sub:
                s += profile.get(("sub", sub), 0.0)
            scores[i] = s if s > 0.0 else self.popularity.get(c, 0.0)

        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def recommend(
        self,
        user_id: str,
        candidates: list[str],
        k: int = 10,
        history: list[str] | None = None,
    ) -> list[str]:
        scores = self.score(user_id, candidates, history=history)
        return [candidates[i] for i in np.argsort(scores)[::-1][:k]]