"""preprocessing, training, and evaluating recommenders.

drafted usage:
- python -m src.run preprocess
- python -m src.run train --model popular
- python -m src.run eval --model popular --k 10
"""

import argparse
import importlib
import inspect
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.preprocess.mind_reader import build_processed_split, load_processed_split


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "popular": {
        "class_path": "src.recommenders.baseline.popular:PopularRecommender",
        "fit_mode": "behaviors",
        "cli_args": [],
        "init_from_args": {},
    },
    "random": {
        "class_path": "src.recommenders.baseline.random:RandomRecommender",
        "fit_mode": "behaviors",
        "cli_args": [
            {"flags": ("--seed",), "kwargs": {"type": int, "default": 42}},
        ],
        "init_from_args": {"seed": "seed"},
    },
    "itemknn": {
        "class_path": "src.recommenders.collaborative.item_knn:ItemKNNRecommender",
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
    "content_tfidf": {
        "class_path": "src.recommenders.content.tfidf:TfidfContentRecommender",
        "fit_mode": "legacy_content",
        "cli_args": [
            {"flags": ("--max-features",), "kwargs": {"type": int, "default": 50000}},
            {"flags": ("--ngram-max",), "kwargs": {"type": int, "default": 2}},
        ],
        "init_from_args": {"max_features": "max_features"},
        "init_builder": lambda args: {"ngram_range": (1, args.ngram_max)},
    },
    "content_entity": {
        "class_path": "src.recommenders.content.entity:EntityContentRecommender",
        "fit_mode": "legacy_content",
        "cli_args": [],
        "init_from_args": {},
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



def _build_popularity_prior(behaviors: pd.DataFrame) -> pd.Series:
    """Global click prior used as a cold-start fallback in evaluation."""
    df = behaviors[["candidates", "labels"]].explode(["candidates", "labels"])
    clicks = df[df["labels"].astype(int) == 1]["candidates"].astype(str)
    if clicks.empty:
        return pd.Series(dtype=np.float32)
    return clicks.value_counts().astype(np.float32)



def _get_news_id_to_idx(model) -> dict[str, int]:
    mapping = getattr(model, "_runner_news_id_to_idx", None)
    if mapping is None:
        mapping = {str(nid): i for i, nid in enumerate(getattr(model, "news_index", []))}
        setattr(model, "_runner_news_id_to_idx", mapping)
    return mapping



def _sanitize_scores(scores: np.ndarray, expected_len: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (expected_len,):
        scores = scores.reshape(-1)
    if len(scores) != expected_len:
        raise ValueError(f"Expected {expected_len} scores, got {len(scores)}.")
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)



def _mask_seen_candidates(
    scores: np.ndarray,
    candidates: list[str],
    history: list[str] | None,
) -> np.ndarray:
    if not history:
        return scores
    seen = set(str(x) for x in history)
    masked = scores.copy()
    if masked.size == 0:
        return masked
    finite = masked[np.isfinite(masked)]
    floor = (float(finite.min()) - 1.0) if finite.size else -1.0
    for idx, nid in enumerate(candidates):
        if str(nid) in seen:
            masked[idx] = floor
    return masked



def _popularity_scores(
    candidates: list[str],
    popularity_prior: pd.Series | None,
    history: list[str] | None = None,
) -> np.ndarray:
    if popularity_prior is None:
        scores = np.zeros(len(candidates), dtype=np.float32)
    else:
        scores = np.array(
            [float(popularity_prior.get(str(nid), 0.0)) for nid in candidates],
            dtype=np.float32,
        )
    return _mask_seen_candidates(scores, candidates, history)



def _score_itemknn_with_history(
    model,
    candidates: list[str],
    history: list[str] | None,
    popularity_prior: pd.Series | None,
) -> np.ndarray:
    hist_items = {str(x) for x in (history or []) if str(x).strip()}
    if not hist_items:
        return _popularity_scores(candidates, popularity_prior, history)

    popularity = model.popularity if getattr(model, "popularity", None) is not None else popularity_prior
    scores: list[float] = []
    for candidate in candidates:
        candidate = str(candidate)
        if candidate in hist_items:
            scores.append(-1.0)
            continue
        sim_row = model.item_similarity.get(candidate, {})
        cf_score = sum(float(sim_row.get(hist_item, 0.0)) for hist_item in hist_items)
        pop_score = float(popularity.get(candidate, 0.0)) if popularity is not None else 0.0
        scores.append(cf_score + 1e-6 * pop_score)
    return _sanitize_scores(np.array(scores, dtype=np.float32), len(candidates))



def _score_tfidf_with_history(
    model,
    candidates: list[str],
    history: list[str] | None,
    popularity_prior: pd.Series | None,
) -> np.ndarray:
    if not history or model.news_tfidf is None:
        return _popularity_scores(candidates, popularity_prior, history)

    news_id_to_idx = _get_news_id_to_idx(model)
    hist_idxs = [news_id_to_idx[nid] for nid in map(str, history) if nid in news_id_to_idx]
    if not hist_idxs:
        return _popularity_scores(candidates, popularity_prior, history)

    user_profile = np.asarray(model.news_tfidf[hist_idxs].mean(axis=0), dtype=np.float32)

    scores = np.array(
        [float(popularity_prior.get(str(nid), 0.0)) if popularity_prior is not None else 0.0 for nid in candidates],
        dtype=np.float32,
    )
    valid_positions: list[int] = []
    valid_candidate_idxs: list[int] = []
    for pos, nid in enumerate(candidates):
        idx = news_id_to_idx.get(str(nid))
        if idx is None:
            continue
        valid_positions.append(pos)
        valid_candidate_idxs.append(idx)

    if valid_candidate_idxs:
        sims = cosine_similarity(user_profile, model.news_tfidf[valid_candidate_idxs]).ravel()
        for pos, sim in zip(valid_positions, sims, strict=False):
            scores[pos] = float(sim)

    scores = _mask_seen_candidates(scores, candidates, history)
    return _sanitize_scores(scores, len(candidates))



def _score_entity_with_history(
    model,
    candidates: list[str],
    history: list[str] | None,
    popularity_prior: pd.Series | None,
) -> np.ndarray:
    if not history or model.news_embeddings is None or model.news_norms is None:
        return _popularity_scores(candidates, popularity_prior, history)

    news_id_to_idx = _get_news_id_to_idx(model)
    hist_idxs = []
    for nid in map(str, history):
        idx = news_id_to_idx.get(nid)
        if idx is None:
            continue
        if float(model.news_norms[idx]) <= 0.0:
            continue
        hist_idxs.append(idx)
    if not hist_idxs:
        return _popularity_scores(candidates, popularity_prior, history)

    user_vec = np.mean(model.news_embeddings[hist_idxs], axis=0).astype(np.float32)
    user_norm = float(np.linalg.norm(user_vec))
    if user_norm <= 0.0:
        return _popularity_scores(candidates, popularity_prior, history)

    scores = np.array(
        [float(popularity_prior.get(str(nid), 0.0)) if popularity_prior is not None else 0.0 for nid in candidates],
        dtype=np.float32,
    )
    for pos, nid in enumerate(candidates):
        idx = news_id_to_idx.get(str(nid))
        if idx is None:
            continue
        denom = float(model.news_norms[idx]) * user_norm
        if denom <= 0.0:
            continue
        scores[pos] = float(np.dot(model.news_embeddings[idx], user_vec) / denom)

    scores = _mask_seen_candidates(scores, candidates, history)
    return _sanitize_scores(scores, len(candidates))



def _call_score_maybe_with_history(model, user_id: str, candidates: list[str], history: list[str] | None):
    score_fn = model.score
    try:
        signature = inspect.signature(score_fn)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        params = signature.parameters
        has_history_param = "history" in params
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if history is not None and (has_history_param or accepts_kwargs):
            return score_fn(user_id, candidates, history=history)
    return score_fn(user_id, candidates)



def _score_candidates(
    model_name: str,
    model,
    user_id: str,
    candidates: list[str],
    history: list[str] | None = None,
    popularity_prior: pd.Series | None = None,
) -> np.ndarray:
    if not candidates:
        return np.array([], dtype=np.float32)

    fit_mode = MODEL_REGISTRY[model_name].get("fit_mode", "behaviors")

    # Use the current impression history at evaluation time for the existing
    # personalized models. This keeps evaluation aligned with the MIND setup
    # even though the legacy model interfaces only expose score(user_id, ...).
    if model_name == "itemknn":
        return _score_itemknn_with_history(model, candidates, history, popularity_prior)
    if model_name == "content_tfidf":
        return _score_tfidf_with_history(model, candidates, history, popularity_prior)
    if model_name == "content_entity":
        return _score_entity_with_history(model, candidates, history, popularity_prior)

    if fit_mode == "legacy_content":
        try:
            ranking = model.score(user_id)
        except ValueError:
            return _popularity_scores(candidates, popularity_prior, history)
        score_by_id = dict(zip(ranking["news_id"].astype(str), ranking["score"], strict=False))
        scores = np.array(
            [float(score_by_id.get(str(nid), 0.0)) for nid in candidates],
            dtype=np.float32,
        )
        scores = _mask_seen_candidates(scores, candidates, history)
        return _sanitize_scores(scores, len(candidates))

    scores = _call_score_maybe_with_history(model, user_id, candidates, history)
    scores = _sanitize_scores(scores, len(candidates))
    return _mask_seen_candidates(scores, candidates, history)



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
    popularity_prior = _build_popularity_prior(beh_train)

    if args.command == "train":
        print(f"Trained model={args.model} on {len(beh_train)} train impressions")
        return

    ndcgs, mrrs, recalls = [], [], []
    for row in beh_test.itertuples(index=False):
        candidates = [str(x) for x in row.candidates]
        labels = np.array(row.labels, dtype=int)
        history = [str(x) for x in row.history] if isinstance(row.history, list) else None
        if not candidates:
            continue

        scores = _score_candidates(
            args.model,
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
