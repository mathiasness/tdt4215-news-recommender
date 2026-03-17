from typing import Any

import numpy as np
import pandas as pd


class BaseRecommender():
    """Abstract base class for recommenders."""

    def fit(self, behaviors: pd.DataFrame) -> Any:
        """Fit the model to the training data."""
        ...


    def score(self, user_id: str, candidates: list[str]) -> np.ndarray:
        """Return relevance scores for the given user and candidate items."""
        ...


    def recommend(self, user_id: str, candidates: list[str], k: int) -> list[str]:
        """Recommend top-k items for a user from a list of candidates."""
        ...

