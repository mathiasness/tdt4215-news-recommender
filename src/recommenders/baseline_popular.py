"""Popularity baseline recommender."""


import numpy as np
import pandas as pd
from src.recommenders.base import BaseRecommender

# TODO: Check the preprocessing to see that behaviors["impressions"] is in the expected format (list of "nid-label" strings).

class PopularRecommender(BaseRecommender):
    """
    Popularity-based recommender system.
    Recommends items based on their overall popularity.
    """

    def __init__(self, top_k: int = None):
        self.top_k = top_k
        self.popularity = None


    def fit(self, behaviors: pd.DataFrame) -> "PopularRecommender":
        """
        Compute global popularity based on training behaviors.
        Parameters:
            behaviors (pd.DataFrame): DataFrame containing user-item interactions.
        Returns:
            PopularRecommender: The fitted recommender instance.
        """

        df = behaviors[["candidates", "labels"]].explode(["candidates", "labels"])
        clicks = df[df["labels"].astype(int) == 1]["candidates"]

        pop = pd.Series(clicks).value_counts()

        if self.top_k is not None:
            pop = pop.head(self.top_k)

        self.popularity = pop
        return self

    def score(self, user_id: str, candidates: list) -> np.ndarray:
       """
       Compute scores for the given items based on their popularity.
         Parameters:
            user_id: ID of the user for whom to compute scores (not used in this model).
            candidates: List of candidate item IDs to score.
       Returns:
            np.ndarray: Array of scores corresponding to the candidate items.
       """
       if self.popularity is None:
            raise RuntimeError("Model must be fit() before calling score().")
       
       return np.array([self.popularity.get(nid, 0) for nid in candidates], dtype=np.float32)


    def recommend(self, user_id: str, candidates: list, k: int = 10) -> list:
        """
        Recommend top-k items for a given user based on popularity scores.
        Parameters:
            user_id: ID of the user for whom to generate recommendations.
            candidates: List of candidate item IDs to consider for recommendation.
            k: Number of top items to recommend.
        Returns:
            List of top-k recommended item IDs.
        """
        scores = self.score(user_id, candidates)
        idx = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in idx]