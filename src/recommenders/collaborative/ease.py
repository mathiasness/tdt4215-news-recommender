from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.recommenders.base import BaseRecommender


class EASERecommender(BaseRecommender):
    """
    EASE (Embarrassingly Shallow Autoencoders) recommender.
    Learns a dense item-item weight matrix B by solving a closed-form regularised least squares problem on the user-item 
    interaction matrix. At inference, a user's score for a candidate is the sum of B[hist_item, candidate] over their click history.
    """
    def __init__(
        self,
        l2: float = 500.0,
        max_items: int = 15000,
        min_item_support: int = 2,
    ):
        self.l2 = float(l2)
        self.max_items = int(max_items)
        self.min_item_support = int(min_item_support)

        self.item_to_idx: dict[str, int] = {}
        self.B: np.ndarray | None = None
        self.popularity: pd.Series = pd.Series(dtype=np.float32)
        self.user_history: dict[str, list[str]] = {}

    def fit(self, behaviors_df: pd.DataFrame) -> "EASERecommender":
        # Build user histories from click history + impression clicks
        user_history: dict[str, list[str]] = {}
        for row in behaviors_df.itertuples(index=False):
            uid = str(row.user_id)
            hist = [str(x) for x in (getattr(row, "history", None) or []) if str(x).strip()]
            candidates = [str(x) for x in (getattr(row, "candidates", None) or [])]
            labels = [int(y) for y in (getattr(row, "labels", None) or [])]
            clicked = [c for c, y in zip(candidates, labels) if y == 1]
            merged = (user_history.get(uid, []) + hist + clicked)[-200:]
            if merged:
                user_history[uid] = merged
        self.user_history = user_history

        # Popularity for cold-start fallback
        exploded = behaviors_df[["candidates", "labels"]].explode(["candidates", "labels"])
        clicked_all = exploded.loc[exploded["labels"].astype(int) == 1, "candidates"].astype(str)
        self.popularity = clicked_all.value_counts().astype(np.float32)

        # Prune to the most-supported items
        item_counts: dict[str, int] = {}
        user_items: dict[str, set[str]] = {}
        for uid, hist in user_history.items():
            uniq = set(hist)
            user_items[uid] = uniq
            for nid in uniq:
                item_counts[nid] = item_counts.get(nid, 0) + 1

        kept = [
            nid for nid, cnt
            in sorted(item_counts.items(), key=lambda kv: -kv[1])
            if cnt >= self.min_item_support
        ][: self.max_items]

        self.item_to_idx = {nid: i for i, nid in enumerate(kept)}

        # Build sparse user-item matrix X
        rows, cols = [], []
        for row_idx, items in enumerate(user_items.values()):
            for nid in items:
                if nid in self.item_to_idx:
                    rows.append(row_idx)
                    cols.append(self.item_to_idx[nid])

        n_users = len(user_items)
        n_items = len(kept)
        X = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )

        # Closed-form EASE
        G = (X.T @ X).toarray().astype(np.float64)
        diag_idx = np.arange(n_items)
        G[diag_idx, diag_idx] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag_idx, diag_idx] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(
        self,
        user_id: str,
        candidates: list[str],
        history: list[str] | None = None,
    ) -> np.ndarray:
        candidates = [str(c) for c in candidates]
        history = [str(h) for h in (history or self.user_history.get(str(user_id), []))]

        # Fall back to popularity when there is no usable signal
        if self.B is None or not history:
            return np.array([self.popularity.get(c, 0.0) for c in candidates], dtype=np.float32)

        hist_idxs = [self.item_to_idx[h] for h in set(history) if h in self.item_to_idx]
        if not hist_idxs:
            return np.array([self.popularity.get(c, 0.0) for c in candidates], dtype=np.float32)

        profile = self.B[hist_idxs].sum(axis=0)  # (n_items,)
        scores = np.array(
            [
                float(profile[self.item_to_idx[c]]) if c in self.item_to_idx
                else float(self.popularity.get(c, 0.0))
                for c in candidates
            ],
            dtype=np.float32,
        )

        # Penalise already-seen items
        seen = set(history)
        floor = float(scores.min()) - 1.0
        for i, c in enumerate(candidates):
            if c in seen:
                scores[i] = floor

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