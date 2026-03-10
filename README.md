# tdt4215-news-recommender

News recommendation project built on the MIND dataset.

## Dataset layout

Expected raw data folders:

- `data/raw/MINDsmall_train/`
- `data/raw/MINDsmall_dev/`

Each split should contain `news.tsv` and `behaviors.tsv`.
For the entity-based content model, `entity_embedding.vec` is also used.

## Run pipeline (`src/run.py`)

Run all commands from the repository root.

### 1. Preprocess data

```bash
python -m src.run preprocess
```

This parses raw MIND files and writes processed files to `data/processed/`:

- `train_news.csv`, `train_behaviors.csv`
- `test_news.csv`, `test_behaviors.csv`

### 2. Train a model

```bash
python -m src.run train --model popular
```

Available models:

- `popular`
- `random`
- `itemknn`
- `content_tfidf`
- `content_entity`

Model-specific examples:

```bash
python -m src.run train --model random --seed 42
python -m src.run train --model itemknn --k-neighbors 50 --top-k-popular 5000
python -m src.run train --model content_tfidf --max-features 50000 --ngram-max 2
python -m src.run train --model content_entity
```

Note: training here fits in-memory for the run and does not persist model artifacts.

### 3. Evaluate one model

```bash
python -m src.run eval --model itemknn --k 10
```

Outputs:

- `nDCG@k`
- `MRR@k`
- `Recall@k`
- `num_impressions`

## Compare all models + plots

Use the evaluation/plotting CLI:

```bash
python -m src.eval.plot_comparison --ks 5,10,20 --output-dir results/model_comparison
```

Useful option for faster iteration:

```bash
python -m src.eval.plot_comparison --ks 5,10 --sample-impressions 500 --output-dir results/model_comparison_smoke
```

Generated outputs include:

- `metrics_summary.csv`
- `ndcg_by_k.png`
- `mrr_by_k.png`
- `recall_by_k.png`
- `coverage_by_k.png`
- `novelty_by_k.png`
- `metrics_heatmap.png`
- `run_metadata.json`

The comparison pipeline reports both accuracy metrics (`nDCG`, `MRR`, `Recall`) and beyond-accuracy metrics (`Coverage`, `Novelty`).
