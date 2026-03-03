"""Random baseline recommender."""

import numpy as np
import pandas as pd

from src.recommenders.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Recommend candidates by random scores."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fit(self, behaviors: pd.DataFrame) -> "RandomRecommender":
        del behaviors
        return self

    def score(self, user_id: str, candidates: list[str]) -> np.ndarray:
        del user_id
        return self.rng.random(len(candidates), dtype=np.float32)

    def recommend(self, user_id: str, candidates: list[str], k: int = 10) -> list[str]:
        scores = self.score(user_id, candidates)
        idx = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in idx]
