from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommenders.base import BaseRecommender
from src.recommenders.collaborative.item_knn import ItemKNNRecommender
from src.recommenders.content.tfidf import TfidfContentRecommender


class HybridNewsRecommender(BaseRecommender):
    """Weighted hybrid of popularity, item-kNN, and TF-IDF."""

    def __init__(
        self,
        pop_weight: float = 0.20,
        itemknn_weight: float = 0.45,
        tfidf_weight: float = 0.35,
        normalize: str = "minmax",
        k_neighbors: int = 50,
        top_k_popular: int | None = None,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ):
        weights = np.array([pop_weight, itemknn_weight, tfidf_weight], dtype=np.float32)
        if np.any(weights < 0):
            raise ValueError("Hybrid weights must be non-negative.")
        if weights.sum() <= 0:
            raise ValueError("At least one hybrid weight must be positive.")
        if normalize not in {"none", "minmax", "zscore"}:
            raise ValueError("normalize must be one of: 'none', 'minmax', 'zscore'.")

        self.pop_weight, self.itemknn_weight, self.tfidf_weight = (weights / weights.sum()).tolist()
        self.normalize = normalize

        self.itemknn = ItemKNNRecommender(
            k_neighbors=int(k_neighbors),
            top_k_popular=top_k_popular,
        )
        self.tfidf = TfidfContentRecommender(
            max_features=int(max_features),
            ngram_range=ngram_range,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        self.popularity = pd.Series(dtype=np.float32)
        self.user_history: dict[str, set[str]] = {}
        self.news_id_to_idx: dict[str, int] = {}

    @staticmethod
    def _build_popularity(behaviors_df: pd.DataFrame) -> pd.Series:
        required = {"candidates", "labels"}
        missing = required - set(behaviors_df.columns)
        if missing:
            raise ValueError(f"behaviors_df missing columns: {sorted(missing)}")

        exploded = behaviors_df[["candidates", "labels"]].explode(["candidates", "labels"])
        clicked = exploded.loc[exploded["labels"].astype(int) == 1, "candidates"].astype(str)
        return clicked.value_counts().astype(np.float32) if not clicked.empty else pd.Series(dtype=np.float32)

    ### Internal helpers ###
    
    @staticmethod
    def _sanitize(scores: np.ndarray, n: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) != n:
            raise ValueError(f"Expected {n} scores, got shape {scores.shape}.")
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        scores = self._sanitize(scores, len(scores))
        if self.normalize == "none" or len(scores) == 0:
            return scores

        if self.normalize == "minmax":
            lo, hi = float(scores.min()), float(scores.max())
            return np.zeros_like(scores) if hi <= lo else ((scores - lo) / (hi - lo)).astype(np.float32)

        mean, std = float(scores.mean()), float(scores.std())
        return np.zeros_like(scores) if std <= 1e-12 else ((scores - mean) / std).astype(np.float32)

    @staticmethod
    def _clean_history(history: list[str] | None) -> list[str]:
        return [str(x) for x in (history or []) if str(x).strip()]
    
    ### Scoring helpers ###

    def _pop_scores(self, candidates: list[str]) -> np.ndarray:
        return np.array([self.popularity.get(str(cid), 0.0) for cid in candidates], dtype=np.float32)

    def _itemknn_scores(self, candidates: list[str], history: list[str]) -> np.ndarray:
        seen = set(history)
        if not seen:
            return self._pop_scores(candidates)

        popularity = self.itemknn.popularity if self.itemknn.popularity is not None else self.popularity
        out = []

        for cid in map(str, candidates):
            if cid in seen:
                out.append(-1.0)
                continue

            sim_row = self.itemknn.item_similarity.get(cid, {})
            cf_score = sum(sim_row.get(h, 0.0) for h in seen)
            pop_score = popularity.get(cid, 0.0) if popularity is not None else 0.0
            out.append(float(cf_score + 1e-6 * pop_score))

        return np.asarray(out, dtype=np.float32)

    def _tfidf_scores(self, candidates: list[str], history: list[str]) -> np.ndarray:
        if self.tfidf.news_tfidf is None or not history:
            return self._pop_scores(candidates)

        profile = self.tfidf._profile_from_history(history)
        if profile is None:
            return self._pop_scores(candidates)

        scores = self._pop_scores(candidates)
        pairs = [(i, self.news_id_to_idx[str(cid)]) for i, cid in enumerate(candidates) if str(cid) in self.news_id_to_idx]
        if not pairs:
            return scores

        positions, idxs = zip(*pairs)
        sims = cosine_similarity(profile, self.tfidf.news_tfidf[list(idxs)]).ravel()
        for pos, sim in zip(positions, sims):
            scores[pos] = float(sim)

        return scores.astype(np.float32)

    @staticmethod
    def _mask_seen(scores: np.ndarray, candidates: list[str], history: list[str]) -> np.ndarray:
        if not history or len(scores) == 0:
            return scores

        seen = set(map(str, history))
        masked = scores.copy()
        floor = float(masked.min()) - 1.0 if len(masked) else -1.0

        for i, cid in enumerate(candidates):
            if str(cid) in seen:
                masked[i] = floor

        return masked

    ### Public API ###

    def fit(
        self,
        news_df: pd.DataFrame,
        behaviors_df: pd.DataFrame,
        text_col: str = "text",
    ) -> "HybridNewsRecommender":
        self.itemknn.fit(behaviors_df)
        self.tfidf.fit(news_df, behaviors_df, text_col=text_col)

        self.popularity = self._build_popularity(behaviors_df)
        self.user_history = dict(self.itemknn.user_history)
        self.news_id_to_idx = {str(nid): i for i, nid in enumerate(self.tfidf.news_index)}
        return self

    def score(
        self,
        user_id: str,
        candidates: list[str],
        history: list[str] | None = None,
    ) -> np.ndarray:
        candidates = [str(x) for x in candidates]
        if not candidates:
            return np.array([], dtype=np.float32)

        history = self._clean_history(history) or sorted(self.user_history.get(str(user_id), set()))

        pop_raw = self._pop_scores(candidates)
        if not history:
            return self._sanitize(pop_raw, len(candidates))

        item_raw = self._itemknn_scores(candidates, history)
        tfidf_raw = self._tfidf_scores(candidates, history)

        combined = (
            self.pop_weight * self._normalize(pop_raw)
            + self.itemknn_weight * self._normalize(item_raw)
            + self.tfidf_weight * self._normalize(tfidf_raw)
        ).astype(np.float32)

        return self._sanitize(self._mask_seen(combined, candidates, history), len(candidates))

    def recommend(
        self,
        user_id: str,
        candidates: list[str],
        k: int = 10,
        history: list[str] | None = None,
    ) -> list[str]:
        if k <= 0:
            return []
        scores = self.score(user_id=user_id, candidates=candidates, history=history)
        return [candidates[i] for i in np.argsort(scores)[::-1][:k]]

"""
python -m src.run train --model hybrid_pop_itemknn_tfidf

python -m src.run eval \
  --model hybrid_pop_itemknn_tfidf \
  --k 10 \
  --hybrid-weight-pop 0.20 \
  --hybrid-weight-itemknn 0.45 \
  --hybrid-weight-tfidf 0.35 \
  --hybrid-normalize minmax

---

python -m src.run eval --model popular --k 10
res:
num_impressions=73152
nDCG@10=0.3088
MRR@10=0.2478
Recall@10=0.5481

python -m src.run eval --model itemknn --k 10
res:
num_impressions=73152
nDCG@10=0.2675
MRR@10=0.2021
Recall@10=0.5072

python -m src.run eval --model content_tfidf --k 10
res:
num_impressions=73152
nDCG@10=0.3765
MRR@10=0.3205
Recall@10=0.6204

---

python -m src.run eval --model hybrid_pop_itemknn_tfidf --k 10 --hybrid-weight-pop 0.10 --hybrid-weight-itemknn 0.60 --hybrid-weight-tfidf 0.30
res:
num_impressions=73152
nDCG@10=0.3373
MRR@10=0.2649
Recall@10=0.6048

python -m src.run eval --model hybrid_pop_itemknn_tfidf --k 10 --hybrid-weight-pop 0.20 --hybrid-weight-itemknn 0.45 --hybrid-weight-tfidf 0.35
res:
model=hybrid_pop_itemknn_tfidf
num_impressions=73152
nDCG@10=0.3383
MRR@10=0.2670
Recall@10=0.6036

python -m src.run eval --model hybrid_pop_itemknn_tfidf --k 10 --hybrid-weight-pop 0.10 --hybrid-weight-itemknn 0.35 --hybrid-weight-tfidf 0.55
res:
num_impressions=73152
nDCG@10=0.3597
MRR@10=0.2998
Recall@10=0.6102

python -m src.run eval --model hybrid_pop_itemknn_tfidf --k 10 --hybrid-weight-pop 0.25 --hybrid-weight-itemknn 0.15 --hybrid-weight-tfidf 0.60
res:
num_impressions=73152
nDCG@10=0.3677
MRR@10=0.3111
Recall@10=0.6118

"""