"""Evaluation pipeline for recommenders."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

import src.run as runner
from src.eval.beyond_accuracy import (
    build_news_metadata_lookup,
    click_popularity,
    coverage_at_k,
    mean_diversity_at_k,
    mean_novelty_at_k,
)
from src.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.preprocess.mind_reader import load_processed_split


class Evaluator:
    """Evaluate one or many registered models across accuracy and beyond-accuracy metrics."""

    def __init__(
        self,
        ks: Iterable[int] = (5, 10),
        sample_impressions: int | None = None,
        random_seed: int = 42,
    ):
        ks = sorted({int(k) for k in ks if int(k) > 0})
        if not ks:
            raise ValueError("`ks` must contain at least one positive integer.")
        self.ks = ks
        self.sample_impressions = sample_impressions
        self.random_seed = random_seed

    @staticmethod
    def _default_model_args(model_name: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
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
        if overrides:
            for key, value in overrides.items():
                setattr(args, key, value)

        news_train, beh_train = load_processed_split("train")
        news_test, beh_test = load_processed_split("test")

        # Keep coverage denominator stable even when only a sample of impressions is evaluated.
        evaluation_candidate_universe: set[str] = {
            str(candidate)
            for candidates in beh_test["candidates"]
            for candidate in candidates
        }

        if self.sample_impressions is not None and self.sample_impressions > 0:
            sample_n = min(self.sample_impressions, len(beh_test))
            beh_test = beh_test.sample(n=sample_n, random_state=self.random_seed).reset_index(
                drop=True
            )

        all_news = pd.concat([news_train, news_test], ignore_index=True).drop_duplicates(
            subset=["news_id"]
        )
        news_metadata = build_news_metadata_lookup(all_news)
        catalog_size = int(all_news["news_id"].nunique())

        model = runner._build_model(model_name, args)
        runner._fit_model(model_name, model, beh_train, news_train, news_test)

        click_counts, total_clicks = click_popularity(beh_train)
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

            scores = np.asarray(
                runner._score_candidates(model_name, model, str(row.user_id), candidates),
                dtype=np.float64,
            ).reshape(-1)
            scores = np.nan_to_num(scores, nan=-1.0e12, posinf=1.0e12, neginf=-1.0e12)

            if len(scores) != len(labels):
                raise ValueError(
                    f"Model returned {len(scores)} scores for {len(labels)} candidates."
                )

            order = np.argsort(-scores, kind="mergesort")
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
                result[f"diversity@{k}"] = 0.0
                continue

            result[f"ndcg@{k}"] = float(np.mean(ndcg_by_k[k]))
            result[f"mrr@{k}"] = float(np.mean(mrr_by_k[k]))
            result[f"recall@{k}"] = float(np.mean(recall_by_k[k]))
            result[f"coverage@{k}"] = float(
                coverage_at_k(recommendations_by_k[k], evaluation_candidate_universe)
            )
            result[f"novelty@{k}"] = float(
                mean_novelty_at_k(
                    recommendations_by_k[k],
                    click_counts,
                    total_clicks,
                    k=k,
                    catalog_size=catalog_size,
                )
            )
            result[f"diversity@{k}"] = float(
                mean_diversity_at_k(recommendations_by_k[k], news_metadata, k=k)
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
