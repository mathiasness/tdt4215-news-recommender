"""Run model comparison and generate metric plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE_ROOT = REPO_ROOT / "data" / "processed" / "cache"
LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR = LOCAL_CACHE_ROOT / "mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.run as runner
from src.eval.evaluator import Evaluator


PLOT_METRICS = ("ndcg", "mrr", "recall", "coverage", "novelty", "diversity")


def _parse_ks(raw: str) -> list[int]:
    ks = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        ks.append(int(value))
    unique = sorted({k for k in ks if k > 0})
    if not unique:
        raise ValueError("--ks must contain at least one positive integer.")
    return unique



def _plot_metric_by_k(
    df: pd.DataFrame,
    metric: str,
    ks: list[int],
    output_path: Path,
    dpi: int,
) -> None:
    metric_cols = [f"{metric}@{k}" for k in ks if f"{metric}@{k}" in df.columns]
    if not metric_cols:
        return

    plot_df = df[["model", *metric_cols]].copy()
    models = plot_df["model"].tolist()
    x = np.arange(len(models), dtype=float)
    width = 0.8 / max(1, len(metric_cols))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(models)), 5))
    for idx, col in enumerate(metric_cols):
        offset = (idx - (len(metric_cols) - 1) / 2) * width
        ax.bar(x + offset, plot_df[col].to_numpy(dtype=float), width=width, label=col)

    ax.set_title(f"{metric.upper()} comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)



def _plot_reference_heatmap(
    df: pd.DataFrame,
    reference_k: int,
    output_path: Path,
    dpi: int,
) -> None:
    columns = [
        f"ndcg@{reference_k}",
        f"mrr@{reference_k}",
        f"recall@{reference_k}",
        f"coverage@{reference_k}",
        f"novelty@{reference_k}",
        f"diversity@{reference_k}",
    ]
    if not all(col in df.columns for col in columns):
        return

    matrix = df[columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(df))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["model"].tolist())
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=20, ha="right")
    ax.set_title(f"Model comparison heatmap (k={reference_k})")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models and generate metric plots.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=sorted(runner.MODEL_REGISTRY.keys()),
        help="Model names to evaluate.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Comma-separated k values, e.g. '5,10,20'.",
    )
    parser.add_argument(
        "--sample-impressions",
        type=int,
        default=None,
        help="Optional number of test impressions to sample for faster runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison",
        help="Where to write CSV and plots.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="ndcg@10",
        help="Metric column to sort models by before plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    return parser



def main() -> None:
    args = build_parser().parse_args()
    ks = _parse_ks(args.ks)

    unknown = [m for m in args.models if m not in runner.MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Available: {sorted(runner.MODEL_REGISTRY.keys())}"
        )

    evaluator = Evaluator(ks=ks, sample_impressions=args.sample_impressions)
    results = evaluator.evaluate_many(args.models)

    sort_by = args.sort_by
    if sort_by not in results.columns:
        fallback = f"ndcg@{10 if 10 in ks else ks[-1]}"
        sort_by = fallback if fallback in results.columns else results.columns[0]
    results = results.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "metrics_summary.csv"
    results.to_csv(summary_csv, index=False)

    for metric in PLOT_METRICS:
        _plot_metric_by_k(
            df=results,
            metric=metric,
            ks=ks,
            output_path=output_dir / f"{metric}_by_k.png",
            dpi=args.dpi,
        )

    reference_k = 10 if 10 in ks else ks[-1]
    _plot_reference_heatmap(
        df=results,
        reference_k=reference_k,
        output_path=output_dir / "metrics_heatmap.png",
        dpi=args.dpi,
    )

    metadata = {
        "models": args.models,
        "ks": ks,
        "sample_impressions": args.sample_impressions,
        "sort_by": sort_by,
        "num_models": int(len(results)),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote comparison outputs to: {output_dir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
