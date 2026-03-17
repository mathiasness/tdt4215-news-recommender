"""Evaluation pipeline for recommenders."""

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

import src.run as runner
from src.eval.beyond_accuracy import (
    click_popularity,
    coverage_at_k,
    mean_novelty_at_k,
)
from src.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.preprocess.mind_reader import load_processed_split


class Evaluator:
    """Evaluate one or many registered models across multiple ranking metrics."""

    def __init__(
        self,
        ks: Iterable[int] = (5, 10),
        sample_impressions: int | None = None,
        random_seed: int = 42,
        processed_dir: str = "data/processed",
        min_history_len: int = 0,
        default_model_overrides: dict[str, int | float | str | bool] | None = None,
    ):
        ks = sorted({int(k) for k in ks if int(k) > 0})
        if not ks:
            raise ValueError("`ks` must contain at least one positive integer.")
        self.ks = ks
        self.sample_impressions = sample_impressions
        self.random_seed = random_seed
        self.processed_dir = processed_dir
        self.min_history_len = int(min_history_len)
        self.default_model_overrides = {} if default_model_overrides is None else default_model_overrides

    @staticmethod
    def _default_model_args(model_name: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        runner._add_data_args(parser)
        runner._add_model_args(parser)
        args = parser.parse_args([])
        setattr(args, "model", model_name)
        return args

    def evaluate(
        self,
        model_name: str,
        overrides: dict[str, int | float | str | bool] | None = None,
    ) -> dict[str, float | int | str]:
        if model_name not in runner.MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'.")

        args = self._default_model_args(model_name)
        for key, value in self.default_model_overrides.items():
            setattr(args, key, value)
        if overrides:
            for key, value in overrides.items():
                setattr(args, key, value)

        news_train, beh_train = load_processed_split(
            "train",
            processed_dir=self.processed_dir,
            min_history_len=self.min_history_len,
        )
        news_test, beh_test = load_processed_split(
            "test",
            processed_dir=self.processed_dir,
            min_history_len=self.min_history_len,
        )

        if self.sample_impressions is not None and self.sample_impressions > 0:
            sample_n = min(self.sample_impressions, len(beh_test))
            beh_test = beh_test.sample(n=sample_n, random_state=self.random_seed).reset_index(
                drop=True
            )

        model = runner._build_model(model_name, args)
        runner._fit_model(model_name, model, beh_train, news_train, news_test)

        click_counts, total_clicks = click_popularity(beh_train)
        popularity_prior = runner._build_popularity_prior(beh_train)
        candidate_universe: set[str] = set()
        recommendations_by_k = {k: [] for k in self.ks}
        ndcg_by_k = {k: [] for k in self.ks}
        mrr_by_k = {k: [] for k in self.ks}
        recall_by_k = {k: [] for k in self.ks}

        num_impressions = 0
        for row in beh_test.itertuples(index=False):
            candidates = list(row.candidates)
            labels = np.asarray(row.labels, dtype=int)
            if not candidates:
                continue

            candidate_universe.update(str(c) for c in candidates)
            history = [str(x) for x in row.history] if isinstance(row.history, list) else None
            scores = runner._score_candidates(
                model_name,
                model,
                str(row.user_id),
                candidates,
                history=history,
                popularity_prior=popularity_prior,
            )
            if len(scores) != len(labels):
                raise ValueError(
                    f"Model returned {len(scores)} scores for {len(labels)} candidates."
                )

            order = np.argsort(scores)[::-1]
            ranked_candidates = [str(candidates[i]) for i in order]

            for k in self.ks:
                ndcg_by_k[k].append(ndcg_at_k(labels, scores, k))
                mrr_by_k[k].append(mrr_at_k(labels, scores, k))
                recall_by_k[k].append(recall_at_k(labels, scores, k))
                recommendations_by_k[k].append(ranked_candidates[:k])

            num_impressions += 1

        result: dict[str, float | int | str] = {
            "model": model_name,
            "num_impressions": num_impressions,
        }
        for k in self.ks:
            if num_impressions == 0:
                result[f"ndcg@{k}"] = 0.0
                result[f"mrr@{k}"] = 0.0
                result[f"recall@{k}"] = 0.0
                result[f"coverage@{k}"] = 0.0
                result[f"novelty@{k}"] = 0.0
                continue

            result[f"ndcg@{k}"] = float(np.mean(ndcg_by_k[k]))
            result[f"mrr@{k}"] = float(np.mean(mrr_by_k[k]))
            result[f"recall@{k}"] = float(np.mean(recall_by_k[k]))
            result[f"coverage@{k}"] = float(
                coverage_at_k(recommendations_by_k[k], candidate_universe)
            )
            result[f"novelty@{k}"] = float(
                mean_novelty_at_k(recommendations_by_k[k], click_counts, total_clicks, k=k)
            )

        return result

    def evaluate_many(
        self,
        model_names: Iterable[str],
        overrides_by_model: dict[str, dict[str, int | float | str | bool]] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for model_name in model_names:
            overrides = None if overrides_by_model is None else overrides_by_model.get(model_name)
            rows.append(self.evaluate(model_name=model_name, overrides=overrides))

        return pd.DataFrame(rows)
