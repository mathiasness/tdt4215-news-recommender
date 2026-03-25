"""NRMS-style news recommender.

A compact implementation using title/text tokenization, multi-head self-attention
for news encoding, and self-attention over clicked-news vectors for user
encoding. The training objective is impression-level softmax over one positive
and sampled negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.recommenders.base import BaseRecommender
from src.recommenders.neural.common import (
    AdditiveAttention,
    NewsNeuralBase,
    SelfAttentionBlock,
    build_vocab,
    cosine_scores,
    encode_text,
)


class _NRMSModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.news_self_attn = SelfAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.news_pool = AdditiveAttention(embed_dim)
        self.user_self_attn = SelfAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.user_pool = AdditiveAttention(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_news(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0)
        x = self.dropout(self.embedding(token_ids))
        x = self.news_self_attn(x, mask=mask)
        return self.news_pool(x, mask=mask)

    def encode_user_from_news(self, news_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.user_self_attn(news_vecs, mask=mask)
        return self.user_pool(x, mask=mask)


class NRMSRecommender(BaseRecommender, NewsNeuralBase):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_word_freq: int = 2,
        max_title_len: int = 30,
        embed_dim: int = 128,
        num_heads: int = 8,
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
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.vocab: dict[str, int] = {}
        self.news_tokens: np.ndarray | None = None
        self.news_matrix: np.ndarray | None = None
        self.model: _NRMSModel | None = None

    def _text_series(self, news_df: pd.DataFrame, text_col: str) -> pd.Series:
        if "title" in news_df.columns:
            return news_df["title"].fillna("")
        return news_df[text_col].fillna("")

    def _news_tensor(self, ids: list[str]) -> torch.Tensor:
        idxs = [self.news_id_to_idx[nid] for nid in ids if nid in self.news_id_to_idx]
        if not idxs:
            return torch.zeros((1, self.max_title_len), dtype=torch.long, device=self.device)
        return torch.tensor(self.news_tokens[np.asarray(idxs)], dtype=torch.long, device=self.device)

    def forward_batch(self, model: _NRMSModel, histories, candidates) -> torch.Tensor:
        batch_scores = []
        for hist_ids, cand_ids in zip(histories, candidates):
            hist_tensor = self._news_tensor(list(hist_ids))
            cand_tensor = self._news_tensor(list(cand_ids))
            hist_vecs = model.encode_news(hist_tensor).unsqueeze(0)
            hist_mask = torch.ones((1, hist_vecs.size(1)), dtype=torch.bool, device=self.device)
            user_vec = model.encode_user_from_news(hist_vecs, hist_mask)
            cand_vecs = model.encode_news(cand_tensor)
            batch_scores.append(torch.matmul(cand_vecs, user_vec.squeeze(0)).unsqueeze(0))
        return torch.cat(batch_scores, dim=0)

    def fit(self, news_df: pd.DataFrame, behaviors_df: pd.DataFrame, text_col: str = "text") -> "NRMSRecommender":
        news = news_df.drop_duplicates(subset=["news_id"]).copy()
        news["news_id"] = news["news_id"].astype(str)
        self.news_index = news["news_id"].tolist()
        self.news_id_to_idx = {nid: i for i, nid in enumerate(self.news_index)}
        self.popularity = self.build_popularity(behaviors_df)
        self.user_history = self.build_user_history(behaviors_df)

        texts = self._text_series(news, text_col)
        self.vocab = build_vocab(texts.tolist(), max_vocab_size=self.vocab_size, min_freq=self.min_word_freq)
        self.news_tokens = np.asarray([encode_text(t, self.vocab, self.max_title_len) for t in texts.tolist()], dtype=np.int64)

        self.model = _NRMSModel(len(self.vocab), self.embed_dim, self.num_heads, self.dropout)
        examples = self.build_training_examples(behaviors_df)
        self.fit_training_loop(self.model, examples)

        self.model.eval()
        with torch.no_grad():
            token_tensor = torch.tensor(self.news_tokens, dtype=torch.long, device=self.device)
            vecs = self.model.encode_news(token_tensor).cpu().numpy().astype(np.float32)
        self.news_matrix = vecs
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
            user = self.model.encode_user_from_news(hist, mask).squeeze(0).cpu().numpy().astype(np.float32)
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
