"""Collaborative filtering recommenders."""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.recommenders.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """
    Item-based kNN recommender for implicit feedback.
    Builds item-item similarity from user click histories and
    scores candidates by summing similarities to a user's clicked items.
    """

    def __init__(self, k_neighbors: int = 50, top_k_popular: int | None = None):
        self.k_neighbors = k_neighbors
        self.top_k_popular = top_k_popular
        self.user_history: dict[str, set[str]] = {}
        self.item_similarity: dict[str, dict[str, float]] = {}
        self.popularity: pd.Series | None = None

    @staticmethod
    def _parse_clicked_from_impressions(impressions: str) -> list[str]:
        clicked = []
        if not isinstance(impressions, str) or not impressions.strip():
            return clicked

        for imp in impressions.split():
            if "-" not in imp:
                continue
            nid, label = imp.rsplit("-", 1)
            if label == "1":
                clicked.append(nid)
        return clicked

    @staticmethod
    def _parse_history(history: str) -> list[str]:
        if not isinstance(history, str) or not history.strip():
            return []
        return history.split()

    def fit(self, behaviors: pd.DataFrame) -> "ItemKNNRecommender":
        """
        Fit item-item similarities using user click histories.
        Expected columns: `user_id` and either `history` or `impressions`.
        """
        if "user_id" not in behaviors.columns:
            raise ValueError("behaviors must contain a `user_id` column.")

        history_by_user: dict[str, set[str]] = defaultdict(set)
        click_counter = Counter()

        has_history = "history" in behaviors.columns
        has_impressions = "impressions" in behaviors.columns

        if not has_history and not has_impressions:
            raise ValueError(
                "behaviors must contain either `history` or `impressions`."
            )

        for row in behaviors.itertuples(index=False):
            user_id = str(getattr(row, "user_id"))
            if has_history:
                items = self._parse_history(getattr(row, "history"))
            else:
                items = self._parse_clicked_from_impressions(getattr(row, "impressions"))

            if not items:
                continue

            unique_items = set(items)
            history_by_user[user_id].update(unique_items)
            click_counter.update(unique_items)

        self.user_history = dict(history_by_user)
        self.popularity = pd.Series(click_counter).sort_values(ascending=False)
        if self.top_k_popular is not None:
            self.popularity = self.popularity.head(self.top_k_popular)

        pair_counts = Counter()
        for items in self.user_history.values():
            if len(items) < 2:
                continue
            for i, j in combinations(sorted(items), 2):
                pair_counts[(i, j)] += 1

        neighbors = defaultdict(list)
        for (i, j), cooc in pair_counts.items():
            denom = np.sqrt(click_counter[i] * click_counter[j])
            if denom == 0:
                continue
            sim = float(cooc / denom)
            neighbors[i].append((j, sim))
            neighbors[j].append((i, sim))

        similarity: dict[str, dict[str, float]] = {}
        for item, nbs in neighbors.items():
            nbs.sort(key=lambda x: x[1], reverse=True)
            top_nbs = nbs[: self.k_neighbors]
            similarity[item] = {nb: score for nb, score in top_nbs}

        self.item_similarity = similarity
        return self

    def score(self, user_id: str, candidates: list[str]) -> np.ndarray:
        """
        Score candidates for one user.
        If user has no history, falls back to global popularity.
        """
        if self.popularity is None:
            raise RuntimeError("Model must be fit() before calling score().")

        user_items = self.user_history.get(str(user_id), set())
        if not user_items:
            return np.array(
                [self.popularity.get(nid, 0.0) for nid in candidates], dtype=np.float32
            )

        scores = []
        for candidate in candidates:
            if candidate in user_items:
                scores.append(-1.0)
                continue

            sim_row = self.item_similarity.get(candidate, {})
            cf_score = sum(sim_row.get(hist_item, 0.0) for hist_item in user_items)
            pop_score = float(self.popularity.get(candidate, 0.0))
            scores.append(cf_score + 1e-6 * pop_score)

        return np.array(scores, dtype=np.float32)

    def recommend(self, user_id: str, candidates: list[str], k: int = 10) -> list[str]:
        """Recommend top-k candidates for one user."""
        if k <= 0:
            return []
        scores = self.score(user_id, candidates)
        idx = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in idx]
