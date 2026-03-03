"""preprocessing, training, and evaluating recommenders.

drafted usage:
- python -m src.run preprocess
- python -m src.run train --model popular
- python -m src.run eval --model popular --k 10
"""

import argparse
import importlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.preprocess.mind_reader import build_processed_split, load_processed_split


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "popular": {
        "class_path": "src.recommenders.baseline_popular:PopularRecommender",
        "fit_mode": "behaviors",
        "cli_args": [],
        "init_from_args": {},
    },
    "random": {
        "class_path": "src.recommenders.baseline_random:RandomRecommender",
        "fit_mode": "behaviors",
        "cli_args": [
            {"flags": ("--seed",), "kwargs": {"type": int, "default": 42}},
        ],
        "init_from_args": {"seed": "seed"},
    },
    "itemknn": {
        "class_path": "src.recommenders.collaborative_filtering:ItemKNNRecommender",
        "fit_mode": "behaviors",
        "cli_args": [
            {"flags": ("--k-neighbors",), "kwargs": {"type": int, "default": 50}},
            {"flags": ("--top-k-popular",), "kwargs": {"type": int, "default": None}},
        ],
        "init_from_args": {
            "k_neighbors": "k_neighbors",
            "top_k_popular": "top_k_popular",
        },
    },
    "content": {
        "class_path": "src.recommenders.content_based:ContentBasedRecommender",
        "fit_mode": "legacy_content",
        "cli_args": [
            {"flags": ("--max-features",), "kwargs": {"type": int, "default": 50000}},
            {"flags": ("--ngram-max",), "kwargs": {"type": int, "default": 2}},
        ],
        "init_from_args": {"max_features": "max_features"},
        "init_builder": lambda args: {"ngram_range": (1, args.ngram_max)},
    },
}


def _load_class(class_path: str):
    module_path, class_name = class_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    added_flags: set[tuple[str, ...]] = set()
    for spec in MODEL_REGISTRY.values():
        for arg_spec in spec.get("cli_args", []):
            flags = tuple(arg_spec["flags"])
            if flags in added_flags:
                continue
            parser.add_argument(*flags, **arg_spec.get("kwargs", {}))
            added_flags.add(flags)


def _build_model(model_name: str, args: argparse.Namespace):
    spec = MODEL_REGISTRY[model_name]
    model_cls = _load_class(spec["class_path"])

    init_from_args: Mapping[str, str] = spec.get("init_from_args", {})
    init_kwargs = {
        param_name: getattr(args, arg_name)
        for param_name, arg_name in init_from_args.items()
    }
    init_builder = spec.get("init_builder")
    if callable(init_builder):
        init_kwargs.update(init_builder(args))

    return model_cls(**init_kwargs)


def _fit_model(
    model_name: str,
    model,
    beh_train: pd.DataFrame,
    news_train: pd.DataFrame,
    news_test: pd.DataFrame,
) -> None:
    fit_mode = MODEL_REGISTRY[model_name].get("fit_mode", "behaviors")
    if fit_mode == "behaviors":
        model.fit(beh_train)
        return
    if fit_mode == "legacy_content":
        all_news = pd.concat([news_train, news_test], ignore_index=True).drop_duplicates(
            subset=["news_id"]
        )
        model.fit(all_news, beh_train, text_col="text")
        return
    raise ValueError(f"Unknown fit_mode='{fit_mode}' for model='{model_name}'")


def _score_candidates(model_name: str, model, user_id: str, candidates: list[str]) -> np.ndarray:
    fit_mode = MODEL_REGISTRY[model_name].get("fit_mode", "behaviors")
    if fit_mode == "legacy_content":
        if not candidates:
            return np.array([], dtype=np.float32)
        try:
            ranking = model.score(user_id)
        except ValueError:
            return np.zeros(len(candidates), dtype=np.float32)
        score_by_id = dict(zip(ranking["news_id"], ranking["score"]))
        return np.array([float(score_by_id.get(nid, 0.0)) for nid in candidates], dtype=np.float32)

    return np.asarray(model.score(user_id, candidates), dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIND news recommender runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preprocess")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), required=True)
    eval_parser.add_argument("--k", type=int, default=10)

    _add_model_args(train_parser)
    _add_model_args(eval_parser)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "preprocess":
        for split in ("train", "test"):
            build_processed_split(split)
        print("Built processed train/test splits.")
        return

    news_train, beh_train = load_processed_split("train")
    news_test, beh_test = load_processed_split("test")

    model = _build_model(args.model, args)
    _fit_model(args.model, model, beh_train, news_train, news_test)

    if args.command == "train":
        print(f"Trained model={args.model} on {len(beh_train)} train impressions")
        return

    ndcgs, mrrs, recalls = [], [], []
    for row in beh_test.itertuples(index=False):
        candidates = list(row.candidates)
        labels = np.array(row.labels, dtype=int)
        if not candidates:
            continue

        scores = _score_candidates(args.model, model, str(row.user_id), candidates)
        if len(scores) != len(labels):
            raise ValueError(
                f"Model returned {len(scores)} scores for {len(labels)} candidates."
            )

        ndcgs.append(ndcg_at_k(labels, scores, args.k))
        mrrs.append(mrr_at_k(labels, scores, args.k))
        recalls.append(recall_at_k(labels, scores, args.k))

    if not ndcgs:
        print(f"model={args.model}")
        print("num_impressions=0")
        print(f"nDCG@{args.k}=0.0000")
        print(f"MRR@{args.k}=0.0000")
        print(f"Recall@{args.k}=0.0000")
        return

    print(f"model={args.model}")
    print(f"num_impressions={len(ndcgs)}")
    print(f"nDCG@{args.k}={float(np.mean(ndcgs)):.4f}")
    print(f"MRR@{args.k}={float(np.mean(mrrs)):.4f}")
    print(f"Recall@{args.k}={float(np.mean(recalls)):.4f}")


if __name__ == "__main__":
    main()
