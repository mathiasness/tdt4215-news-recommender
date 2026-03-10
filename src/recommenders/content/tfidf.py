"""TF-IDF content recommender."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommenders.base import BaseRecommender


class TfidfContentRecommender(BaseRecommender):
    """Build user profiles from averaged TF-IDF vectors of clicked news."""

    CACHE_SCHEMA_VERSION = "tfidf-v1"

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_cache = use_cache
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[3]
            self.cache_dir = repo_root / "data" / "processed" / "cache" / "tfidf"
        else:
            self.cache_dir = Path(cache_dir)
        self.news_index: list[str] = []
        self.news_tfidf = None
        self.user_profiles: dict[str, np.ndarray] = {}

    def _build_cache_key(self, news: pd.DataFrame, text_col: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.CACHE_SCHEMA_VERSION.encode("utf-8"))
        hasher.update(str(self.max_features).encode("utf-8"))
        hasher.update(str(self.ngram_range).encode("utf-8"))
        for news_id, text in zip(news["news_id"], news[text_col], strict=False):
            hasher.update(str(news_id).encode("utf-8", errors="ignore"))
            hasher.update(b"\t")
            hasher.update(str(text).encode("utf-8", errors="ignore"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _cache_paths(self, cache_key: str) -> tuple[Path, Path, Path]:
        matrix_path = self.cache_dir / f"{cache_key}.npz"
        meta_path = self.cache_dir / f"{cache_key}.json"
        vectorizer_path = self.cache_dir / f"{cache_key}.pkl"
        return matrix_path, meta_path, vectorizer_path

    def _load_from_cache(self, cache_key: str) -> bool:
        matrix_path, meta_path, vectorizer_path = self._cache_paths(cache_key)
        if not (matrix_path.exists() and meta_path.exists() and vectorizer_path.exists()):
            return False
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            news_index = meta["news_index"]
            news_tfidf = sparse.load_npz(matrix_path)
            with vectorizer_path.open("rb") as f:
                vectorizer = pickle.load(f)
        except Exception:
            return False

        if news_tfidf.shape[0] != len(news_index):
            return False

        self.news_index = [str(x) for x in news_index]
        self.news_tfidf = news_tfidf
        self.vectorizer = vectorizer
        return True

    def _save_to_cache(self, cache_key: str) -> None:
        if self.news_tfidf is None:
            return
        matrix_path, meta_path, vectorizer_path = self._cache_paths(cache_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(matrix_path, self.news_tfidf)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"news_index": self.news_index}, f)
        with vectorizer_path.open("wb") as f:
            pickle.dump(self.vectorizer, f)

    def fit(
        self,
        news_df: pd.DataFrame,
        user_history_df: pd.DataFrame,
        text_col: str = "text",
    ) -> "TfidfContentRecommender":
        news = news_df.copy()
        news["text"] = news["title"] + " " + news["abstract"]
        news = news[["news_id", "text"]]
        news["news_id"] = news["news_id"].astype(str)
        news[text_col] = news[text_col].astype(str)

        cache_loaded = False
        if self.use_cache:
            cache_key = self._build_cache_key(news, text_col)
            cache_loaded = self._load_from_cache(cache_key)
        if not cache_loaded:
            self.news_index = news["news_id"].tolist()
            self.news_tfidf = self.vectorizer.fit_transform(news[text_col])
            if self.use_cache:
                self._save_to_cache(cache_key)

        news_id_to_idx = {nid: i for i, nid in enumerate(self.news_index)}
        self.user_profiles = {}
        for row in user_history_df[["user_id", "history"]].itertuples(index=False):
            user = str(row.user_id)
            clicked = row.history if isinstance(row.history, list) else []
            idxs = [news_id_to_idx[str(nid)] for nid in clicked if str(nid) in news_id_to_idx]
            if idxs:
                self.user_profiles[user] = np.asarray(self.news_tfidf[idxs].mean(axis=0))

        return self

    def score(self, user_id: str) -> pd.DataFrame:
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        scores = cosine_similarity(self.user_profiles[user_id], self.news_tfidf).flatten()
        return pd.DataFrame({"news_id": self.news_index, "score": scores}).sort_values(
            "score", ascending=False
        )

    def recommend(
        self, user_id: str, k: int = 10, seen_news: list[str] | None = None
    ) -> list[str]:
        rankings = self.score(user_id)
        if seen_news is not None:
            rankings = rankings[~rankings["news_id"].isin(seen_news)]
        return rankings.head(k)["news_id"].tolist()
