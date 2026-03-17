"""Entity-embedding content recommender."""

import numpy as np
import pandas as pd

from src.preprocess.mind_reader import load_entity_embeddings, load_news_entity_ids
from src.recommenders.base import BaseRecommender


class EntityContentRecommender(BaseRecommender):
    """
    Content model that represents a news article as the mean of its entity embeddings.
    User profile is the mean of embeddings from clicked history items.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        train_split_dir: str = "MINDsmall_train",
        test_split_dir: str = "MINDsmall_dev",
    ):
        self.data_dir = data_dir
        self.split_dirs = {
            "train": train_split_dir,
            "test": test_split_dir,
        }
        self.news_index: list[str] = []
        self.news_embeddings: np.ndarray | None = None
        self.news_norms: np.ndarray | None = None
        self.user_profiles: dict[str, np.ndarray] = {}

    def fit(
        self,
        news_df: pd.DataFrame,
        user_history_df: pd.DataFrame,
        text_col: str = "text",
    ) -> "EntityContentRecommender":
        del text_col

        raw_entity_vectors = load_entity_embeddings(
            data_dir=self.data_dir,
            split_dirs=self.split_dirs,
        )
        if not raw_entity_vectors:
            raise ValueError("No entity embeddings were found.")
        entity_vectors = {
            key: np.asarray(vec, dtype=np.float32) for key, vec in raw_entity_vectors.items()
        }
        news_entity_ids = load_news_entity_ids(
            data_dir=self.data_dir,
            split_dirs=self.split_dirs,
        )

        sample_vec = next(iter(entity_vectors.values()))
        dim = int(sample_vec.shape[0])

        self.news_index = news_df["news_id"].astype(str).drop_duplicates(keep="first").tolist()
        self.news_embeddings = np.zeros((len(self.news_index), dim), dtype=np.float32)
        self.news_norms = np.zeros(len(self.news_index), dtype=np.float32)

        news_id_to_idx: dict[str, int] = {}
        for idx, news_id in enumerate(self.news_index):
            news_id_to_idx[news_id] = idx
            entities = news_entity_ids.get(news_id, [])
            vectors = [entity_vectors[eid] for eid in entities if eid in entity_vectors]
            if not vectors:
                continue
            vec = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
            self.news_embeddings[idx] = vec
            self.news_norms[idx] = float(np.linalg.norm(vec))

        self.user_profiles = {}
        for row in user_history_df[["user_id", "history"]].itertuples(index=False):
            user_id = str(row.user_id)
            history = row.history if isinstance(row.history, list) else []
            idxs = []
            for news_id in history:
                idx = news_id_to_idx.get(str(news_id))
                if idx is None:
                    continue
                if self.news_norms[idx] <= 0.0:
                    continue
                idxs.append(idx)
            if not idxs:
                continue
            user_vec = np.mean(self.news_embeddings[idxs], axis=0).astype(np.float32)
            if float(np.linalg.norm(user_vec)) > 0.0:
                self.user_profiles[user_id] = user_vec

        return self

    def score(self, user_id: str) -> pd.DataFrame:
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        if self.news_embeddings is None or self.news_norms is None:
            raise RuntimeError("Model must be fit() before calling score().")

        user_vec = self.user_profiles[user_id]
        user_norm = float(np.linalg.norm(user_vec))
        if user_norm <= 0.0:
            scores = np.zeros(len(self.news_index), dtype=np.float32)
        else:
            numerators = self.news_embeddings @ user_vec
            denominators = self.news_norms * user_norm
            scores = np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators, dtype=np.float32),
                where=denominators > 0.0,
            )

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
