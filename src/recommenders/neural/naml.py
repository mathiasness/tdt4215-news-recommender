"""NAML-style news recommender.

Compact multi-view implementation using title, abstract, category, and
subcategory encoders with attentive fusion. User encoding is attention over
clicked-news vectors. Training uses impression-level softmax over one positive
and sampled negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.recommenders.base import BaseRecommender
from src.recommenders.neural.utility import (
    AdditiveAttention,
    NewsNeuralBase,
    build_category_map,
    build_vocab,
    cosine_scores,
    encode_text,
)


class _TextViewEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.pool = AdditiveAttention(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0)
        x = self.dropout(self.embedding(token_ids))
        return self.pool(x, mask=mask)


class _NAMLModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, category_size: int, category_dim: int, dropout: float):
        super().__init__()
        self.title_encoder = _TextViewEncoder(vocab_size, embed_dim, dropout)
        self.abstract_encoder = _TextViewEncoder(vocab_size, embed_dim, dropout)
        self.category_embedding = nn.Embedding(category_size, category_dim)
        self.subcategory_embedding = nn.Embedding(category_size, category_dim)
        self.cat_proj = nn.Linear(category_dim, embed_dim)
        self.subcat_proj = nn.Linear(category_dim, embed_dim)
        self.view_attention = AdditiveAttention(embed_dim)
        self.user_attention = AdditiveAttention(embed_dim)

    def encode_news(
        self,
        title_ids: torch.Tensor,
        abstract_ids: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
    ) -> torch.Tensor:
        title_vec = self.title_encoder(title_ids)
        abstract_vec = self.abstract_encoder(abstract_ids)
        category_vec = self.cat_proj(self.category_embedding(category_ids))
        subcategory_vec = self.subcat_proj(self.subcategory_embedding(subcategory_ids))
        views = torch.stack([title_vec, abstract_vec, category_vec, subcategory_vec], dim=1)
        mask = torch.ones((views.size(0), views.size(1)), dtype=torch.bool, device=views.device)
        return self.view_attention(views, mask=mask)

    def encode_user(self, clicked_news_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.user_attention(clicked_news_vecs, mask=mask)


class NAMLRecommender(BaseRecommender, NewsNeuralBase):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_word_freq: int = 2,
        max_title_len: int = 24,
        max_abstract_len: int = 48,
        embed_dim: int = 128,
        category_dim: int = 64,
        dropout: float = 0.2,
        history_size: int = 20,
        neg_ratio: int = 4,
        batch_size: int = 64,
        epochs: int = 3,
        lr: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
    ):
        NewsNeuralBase.__init__(
            self,
            seed=seed,
            device=device,
            history_size=history_size,
            neg_ratio=neg_ratio,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
        )
        self.vocab_size = int(vocab_size)
        self.min_word_freq = int(min_word_freq)
        self.max_title_len = int(max_title_len)
        self.max_abstract_len = int(max_abstract_len)
        self.embed_dim = int(embed_dim)
        self.category_dim = int(category_dim)
        self.dropout = float(dropout)

        self.vocab: dict[str, int] = {}
        self.category_map: dict[str, int] = {}
        self.subcategory_map: dict[str, int] = {}
        self.news_title_tokens: np.ndarray | None = None
        self.news_abstract_tokens: np.ndarray | None = None
        self.news_category_ids: np.ndarray | None = None
        self.news_subcategory_ids: np.ndarray | None = None
        self.news_matrix: np.ndarray | None = None
        self.model: _NAMLModel | None = None

    def _news_inputs(self, ids: list[str]):
        idxs = [self.news_id_to_idx[nid] for nid in ids if nid in self.news_id_to_idx]
        if not idxs:
            return (
                torch.zeros((1, self.max_title_len), dtype=torch.long, device=self.device),
                torch.zeros((1, self.max_abstract_len), dtype=torch.long, device=self.device),
                torch.zeros((1,), dtype=torch.long, device=self.device),
                torch.zeros((1,), dtype=torch.long, device=self.device),
            )
        arr = np.asarray(idxs)
        return (
            torch.tensor(self.news_title_tokens[arr], dtype=torch.long, device=self.device),
            torch.tensor(self.news_abstract_tokens[arr], dtype=torch.long, device=self.device),
            torch.tensor(self.news_category_ids[arr], dtype=torch.long, device=self.device),
            torch.tensor(self.news_subcategory_ids[arr], dtype=torch.long, device=self.device),
        )

    def forward_batch(self, model: _NAMLModel, histories, candidates) -> torch.Tensor:
        batch_scores = []
        for hist_ids, cand_ids in zip(histories, candidates):
            hist_inputs = self._news_inputs(list(hist_ids))
            cand_inputs = self._news_inputs(list(cand_ids))
            hist_vecs = model.encode_news(*hist_inputs).unsqueeze(0)
            hist_mask = torch.ones((1, hist_vecs.size(1)), dtype=torch.bool, device=self.device)
            user_vec = model.encode_user(hist_vecs, hist_mask).squeeze(0)
            cand_vecs = model.encode_news(*cand_inputs)
            batch_scores.append(torch.matmul(cand_vecs, user_vec).unsqueeze(0))
        return torch.cat(batch_scores, dim=0)

    def fit(self, news_df: pd.DataFrame, behaviors_df: pd.DataFrame, text_col: str = "text") -> "NAMLRecommender":
        news = news_df.drop_duplicates(subset=["news_id"]).copy()
        news["news_id"] = news["news_id"].astype(str)
        self.news_index = news["news_id"].tolist()
        self.news_id_to_idx = {nid: i for i, nid in enumerate(self.news_index)}
        self.popularity = self.build_popularity(behaviors_df)
        self.user_history = self.build_user_history(behaviors_df)

        titles = news["title"].fillna(news.get(text_col, "")) if "title" in news.columns else news[text_col].fillna("")
        abstracts = news["abstract"].fillna("") if "abstract" in news.columns else news[text_col].fillna("")
        self.vocab = build_vocab((titles.fillna("") + " " + abstracts.fillna("")).tolist(), max_vocab_size=self.vocab_size, min_freq=self.min_word_freq)
        self.category_map = build_category_map(news["category"] if "category" in news.columns else [])
        self.subcategory_map = build_category_map(news["subcategory"] if "subcategory" in news.columns else [])

        self.news_title_tokens = np.asarray([encode_text(t, self.vocab, self.max_title_len) for t in titles.tolist()], dtype=np.int64)
        self.news_abstract_tokens = np.asarray([encode_text(t, self.vocab, self.max_abstract_len) for t in abstracts.tolist()], dtype=np.int64)
        self.news_category_ids = np.asarray([self.category_map.get(str(v), 0) for v in (news["category"].tolist() if "category" in news.columns else [""] * len(news))], dtype=np.int64)
        self.news_subcategory_ids = np.asarray([self.subcategory_map.get(str(v), 0) for v in (news["subcategory"].tolist() if "subcategory" in news.columns else [""] * len(news))], dtype=np.int64)

        category_size = max(max(self.category_map.values(), default=0), max(self.subcategory_map.values(), default=0)) + 1
        self.model = _NAMLModel(len(self.vocab), self.embed_dim, category_size, self.category_dim, self.dropout)
        examples = self.build_training_examples(behaviors_df)
        self.fit_training_loop(self.model, examples)

        self.model.eval()
        with torch.no_grad():
            title_t = torch.tensor(self.news_title_tokens, dtype=torch.long, device=self.device)
            abs_t = torch.tensor(self.news_abstract_tokens, dtype=torch.long, device=self.device)
            cat_t = torch.tensor(self.news_category_ids, dtype=torch.long, device=self.device)
            subcat_t = torch.tensor(self.news_subcategory_ids, dtype=torch.long, device=self.device)
            self.news_matrix = self.model.encode_news(title_t, abs_t, cat_t, subcat_t).cpu().numpy().astype(np.float32)
        return self

    def encode_user(self, history) -> np.ndarray:
        if self.model is None or self.news_matrix is None:
            return np.array([], dtype=np.float32)
        idxs = [self.news_id_to_idx[h] for h in history if h in self.news_id_to_idx]
        if not idxs:
            return np.array([], dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            hist = torch.tensor(self.news_matrix[np.asarray(idxs)], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.ones((1, hist.size(1)), dtype=torch.bool, device=self.device)
            user = self.model.encode_user(hist, mask).squeeze(0).cpu().numpy().astype(np.float32)
        return user

    def candidate_scores(self, user_vec: np.ndarray, candidates) -> np.ndarray:
        if self.news_matrix is None:
            return np.asarray([self.popularity.get(str(cid), 0.0) for cid in candidates], dtype=np.float32)
        idxs, positions = [], []
        scores = np.asarray([self.popularity.get(str(cid), 0.0) for cid in candidates], dtype=np.float32)
        for i, cid in enumerate(candidates):
            idx = self.news_id_to_idx.get(str(cid))
            if idx is not None:
                idxs.append(idx)
                positions.append(i)
        if idxs:
            sims = cosine_scores(user_vec, self.news_matrix, idxs)
            for pos, sim in zip(positions, sims):
                scores[pos] = float(sim)
        return scores.astype(np.float32)
