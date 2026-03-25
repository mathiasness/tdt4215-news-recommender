from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class TrainingExample:
    history: tuple[str, ...]
    candidates: tuple[str, ...]


class RankingDataset(Dataset):
    def __init__(self, examples: Sequence[TrainingExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        logits = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            weights = weights * mask.float()
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(x + self.dropout(attn_out))


class NewsNeuralBase:
    def __init__(
        self,
        *,
        seed: int = 42,
        device: str | None = None,
        history_size: int = 20,
        neg_ratio: int = 4,
        batch_size: int = 64,
        epochs: int = 3,
        lr: float = 1e-3,
    ):
        self.seed = int(seed)
        self.history_size = int(history_size)
        self.neg_ratio = int(neg_ratio)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = random.Random(self.seed)

        self.popularity = pd.Series(dtype=np.float32)
        self.user_history: dict[str, list[str]] = {}
        self.news_id_to_idx: dict[str, int] = {}
        self.news_index: list[str] = []
        self._trained = False

    @staticmethod
    def build_popularity(behaviors_df: pd.DataFrame) -> pd.Series:
        exploded = behaviors_df[["candidates", "labels"]].explode(["candidates", "labels"])
        clicked = exploded.loc[exploded["labels"].astype(int) == 1, "candidates"].astype(str)
        return clicked.value_counts().astype(np.float32) if not clicked.empty else pd.Series(dtype=np.float32)

    @staticmethod
    def clean_history(history: Sequence[str] | None, history_size: int) -> list[str]:
        if not history:
            return []
        out = [str(x) for x in history if str(x).strip()]
        return out[-history_size:]

    def build_user_history(self, behaviors_df: pd.DataFrame) -> dict[str, list[str]]:
        user_history: dict[str, list[str]] = {}
        for row in behaviors_df.itertuples(index=False):
            uid = str(row.user_id)
            hist = self.clean_history(getattr(row, "history", None), self.history_size)
            if hist:
                user_history.setdefault(uid, hist)
            candidates = list(map(str, getattr(row, "candidates", []) or []))
            labels = list(getattr(row, "labels", []) or [])
            clicked = [c for c, y in zip(candidates, labels) if int(y) == 1]
            if clicked:
                merged = user_history.get(uid, []) + clicked
                user_history[uid] = merged[-self.history_size :]
        return user_history

    def build_training_examples(self, behaviors_df: pd.DataFrame) -> list[TrainingExample]:
        examples: list[TrainingExample] = []

        for row in behaviors_df.itertuples(index=False):
            history = self.clean_history(getattr(row, "history", None), self.history_size)
            candidates = list(map(str, getattr(row, "candidates", []) or []))
            labels = [int(y) for y in (getattr(row, "labels", []) or [])]

            positives = [c for c, y in zip(candidates, labels) if y == 1]
            negatives = [c for c, y in zip(candidates, labels) if y == 0]

            if not history or not positives:
                continue

            # For sampled-softmax training, keep candidate count fixed.
            if self.neg_ratio > 0 and not negatives:
                continue

            for pos in positives:
                if self.neg_ratio <= 0:
                    sampled_negs = []
                elif len(negatives) >= self.neg_ratio:
                    sampled_negs = self.rng.sample(negatives, self.neg_ratio)
                else:
                    sampled_negs = list(negatives)
                    sampled_negs += self.rng.choices(negatives, k=self.neg_ratio - len(negatives))

                examples.append(
                    TrainingExample(
                        history=tuple(history),
                        candidates=tuple([pos] + sampled_negs),
                    )
                )

        return examples

    def loader(self, examples: Sequence[TrainingExample], shuffle: bool = True) -> DataLoader:
        return DataLoader(
            RankingDataset(examples),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: batch,
        )

    def fit_training_loop(self, model: nn.Module, examples: Sequence[TrainingExample]) -> None:
        if not examples:
            self._trained = True
            return

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for _ in range(self.epochs):
            for batch in self.loader(examples, shuffle=True):
                history_ids = [ex.history for ex in batch]
                candidate_ids = [ex.candidates for ex in batch]
                scores = self.forward_batch(model, history_ids, candidate_ids)
                labels = torch.zeros(scores.size(0), dtype=torch.long, device=self.device)
                loss = criterion(scores, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        self._trained = True

    def forward_batch(self, model: nn.Module, histories: Sequence[Sequence[str]], candidates: Sequence[Sequence[str]]) -> torch.Tensor:
        raise NotImplementedError

    def encode_user(self, history: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def candidate_scores(self, user_vec: np.ndarray, candidates: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def score(self, user_id: str, candidates: list[str], history: list[str] | None = None) -> np.ndarray:
        candidates = [str(x) for x in candidates]
        if not candidates:
            return np.array([], dtype=np.float32)

        history = self.clean_history(history, self.history_size) or self.user_history.get(str(user_id), [])
        if not history or not self._trained:
            return np.asarray([self.popularity.get(cid, 0.0) for cid in candidates], dtype=np.float32)

        user_vec = self.encode_user(history)
        if user_vec.size == 0:
            return np.asarray([self.popularity.get(cid, 0.0) for cid in candidates], dtype=np.float32)

        scores = self.candidate_scores(user_vec, candidates)
        if scores.shape != (len(candidates),):
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(f"Expected {len(candidates)} scores, got {len(scores)}")

        seen = set(map(str, history))
        if seen and len(scores):
            floor = float(np.min(scores)) - 1.0
            for i, cid in enumerate(candidates):
                if cid in seen:
                    scores[i] = floor
        return np.nan_to_num(scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def recommend(self, user_id: str, candidates: list[str], k: int = 10, history: list[str] | None = None) -> list[str]:
        if k <= 0:
            return []
        scores = self.score(user_id, candidates, history=history)
        order = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in order]


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: Iterable[str], max_vocab_size: int = 30000, min_freq: int = 2) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in items:
        if freq < min_freq:
            break
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, 1) for tok in tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def build_category_map(values: Iterable[str]) -> dict[str, int]:
    uniq = sorted({str(v) for v in values if pd.notna(v)})
    return {"<unk>": 0, **{v: i + 1 for i, v in enumerate(uniq)}}


def cosine_scores(user_vec: np.ndarray, item_matrix: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
    if user_vec.size == 0 or not idxs:
        return np.array([], dtype=np.float32)
    cand = item_matrix[np.asarray(idxs)]
    numer = cand @ user_vec
    denom = np.linalg.norm(cand, axis=1) * max(np.linalg.norm(user_vec), 1e-8)
    denom = np.clip(denom, 1e-8, None)
    return (numer / denom).astype(np.float32)
