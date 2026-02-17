"""Random baseline recommender"""

import numpy as np
import pandas as pd
from src.recommenders.base import BaseRecommender

class RandomRecommender(BaseRecommender):
    """
    Random recommender system.
    Recommends items randomly from the candidate set.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fit(self, behaviors: pd.DataFrame) -> "RandomRecommender":
        """
        Fit method for the random recommender. No training is needed, so this is a no-op.
        Parameters:
            behaviors (pd.DataFrame): DataFrame containing user-item interactions (not used in this model
        """
        return self
    
    def score(self, user_id: str, candidates: list) -> np.ndarray:
        """
        Compute random scores for the given items.
        Parameters:
            user_id: ID of the user for whom to compute scores (not used in this model).
            candidates: List of candidate item IDs to score.
        Returns:
            np.ndarray: Array of random scores corresponding to the candidate items.
        """
        return self.rng.random(len(candidates), dtype=np.float32)
    
    def recommend(self, user_id: str, candidates: list, k: int = 10) -> list:
        """
        Recommend top-k items for a given user based on random scores.
        Parameters:
            user_id: ID of the user for whom to generate recommendations.
            candidates: List of candidate item IDs to recommend from.
            k: Number of items to recommend.
        Returns:
            list: List of recommended item IDs.
        """
        scores = self.score(user_id, candidates)
        idx = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in idx]