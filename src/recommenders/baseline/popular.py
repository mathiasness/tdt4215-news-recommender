"""Popularity baseline recommender."""

import numpy as np
import pandas as pd

from src.recommenders.base import BaseRecommender


class PopularRecommender(BaseRecommender):
    """Recommend items by global click popularity."""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k
        self.popularity: pd.Series | None = None

    def fit(self, behaviors: pd.DataFrame) -> "PopularRecommender":
        df = behaviors[["candidates", "labels"]].explode(["candidates", "labels"])
        clicks = df[df["labels"].astype(int) == 1]["candidates"]

        pop = pd.Series(clicks).value_counts()
        if self.top_k is not None:
            pop = pop.head(self.top_k)

        self.popularity = pop
        return self

    def score(self, user_id: str, candidates: list[str]) -> np.ndarray:
        del user_id
        if self.popularity is None:
            raise RuntimeError("Model must be fit() before calling score().")
        return np.array([self.popularity.get(nid, 0.0) for nid in candidates], dtype=np.float32)

    def recommend(self, user_id: str, candidates: list[str], k: int = 10) -> list[str]:
        scores = self.score(user_id, candidates)
        idx = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in idx]
