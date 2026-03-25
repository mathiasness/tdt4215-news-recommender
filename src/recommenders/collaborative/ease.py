"""Practical EASE recommender for implicit-feedback news recommendation.

This implementation prunes the item universe before fitting because dense EASE
scales quadratically in the number of items and is otherwise not feasible on
news corpora with tens of thousands of articles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.recommenders.base import BaseRecommender


class EASERecommender(BaseRecommender):
    def __init__(
        self,
        l2: float = 500.0,
        max_items: int = 8000,
        min_item_support: int = 2,
    ):
        if l2 <= 0:
            raise ValueError("l2 must be positive.")
        self.l2 = float(l2)
        self.max_items = int(max_items)
        self.min_item_support = int(min_item_support)

        self.item_to_idx: dict[str, int] = {}
        self.idx_to_item: list[str] = []
        self.B: np.ndarray | None = None
        self.popularity = pd.Series(dtype=np.float32)
        self.user_history: dict[str, list[str]] = {}

    @staticmethod
    def _build_popularity(behaviors_df: pd.DataFrame) -> pd.Series:
        exploded = behaviors_df[["candidates", "labels"]].explode(["candidates", "labels"])
        clicked = exploded.loc[exploded["labels"].astype(int) == 1, "candidates"].astype(str)
        return clicked.value_counts().astype(np.float32) if not clicked.empty else pd.Series(dtype=np.float32)

    def _build_user_histories(self, behaviors_df: pd.DataFrame) -> dict[str, list[str]]:
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
        return user_history

    def fit(self, behaviors_df: pd.DataFrame) -> "EASERecommender":
        self.popularity = self._build_popularity(behaviors_df)
        self.user_history = self._build_user_histories(behaviors_df)

        item_counts: dict[str, int] = {}
        user_items: dict[str, set[str]] = {}
        for uid, hist in self.user_history.items():
            uniq = {str(x) for x in hist}
            if not uniq:
                continue
            user_items[uid] = uniq
            for nid in uniq:
                item_counts[nid] = item_counts.get(nid, 0) + 1

        kept_items = [
            nid
            for nid, cnt in sorted(item_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if cnt >= self.min_item_support
        ][: self.max_items]
        self.idx_to_item = list(kept_items)
        self.item_to_idx = {nid: i for i, nid in enumerate(self.idx_to_item)}
        if not self.idx_to_item:
            self.B = np.zeros((0, 0), dtype=np.float32)
            return self

        rows, cols = [], []
        for row_idx, items in enumerate(user_items.values()):
            valid = [self.item_to_idx[nid] for nid in items if nid in self.item_to_idx]
            rows.extend([row_idx] * len(valid))
            cols.extend(valid)

        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(user_items), len(self.idx_to_item)))
        G = (X.T @ X).toarray().astype(np.float64)
        diag = np.arange(G.shape[0])
        G[diag, diag] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag, diag] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, user_id: str, candidates: list[str], history: list[str] | None = None) -> np.ndarray:
        candidates = [str(x) for x in candidates]
        if not candidates:
            return np.array([], dtype=np.float32)
        if self.B is None or self.B.size == 0:
            return np.asarray([self.popularity.get(cid, 0.0) for cid in candidates], dtype=np.float32)

        history = [str(x) for x in (history or self.user_history.get(str(user_id), [])) if str(x).strip()]
        if not history:
            return np.asarray([self.popularity.get(cid, 0.0) for cid in candidates], dtype=np.float32)

        hist_idxs = [self.item_to_idx[h] for h in set(history) if h in self.item_to_idx]
        if not hist_idxs:
            return np.asarray([self.popularity.get(cid, 0.0) for cid in candidates], dtype=np.float32)

        profile = self.B[hist_idxs].sum(axis=0)
        scores = np.array(
            [float(profile[self.item_to_idx[cid]]) if cid in self.item_to_idx else float(self.popularity.get(cid, 0.0)) for cid in candidates],
            dtype=np.float32,
        )
        floor = float(scores.min()) - 1.0 if len(scores) else -1.0
        seen = set(history)
        for i, cid in enumerate(candidates):
            if cid in seen:
                scores[i] = floor
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def recommend(self, user_id: str, candidates: list[str], k: int = 10, history: list[str] | None = None) -> list[str]:
        if k <= 0:
            return []
        scores = self.score(user_id, candidates, history=history)
        order = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in order]
