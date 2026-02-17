"""Utilities for reading and caching MIND data splits."""
import pandas as pd
import os

BASE_PATH = "../../data"

# Anticipating that the data is stored in "tdt4215-news-recommender/data"

def _read_file(split, filename, raw=False):
    folder = "raw" if raw else "preprocessed"
    path = os.path.join(BASE_PATH, folder, split, filename)

    if not os.path.exists(path):
        path = os.path.join(BASE_PATH, "raw", split, filename)

    return pd.read_csv(path, sep="\t")

def read_behaviors_train(raw=False):
    return _read_file("MINDsmall_train", "behaviours.tsv", raw)

def read_behaviors_test(raw=False):
    return _read_file("MINDsmall_dev", "behaviours.tsv", raw)

def read_news_train(raw=False):
    return _read_file("MINDsmall_train", "news.tsv", raw)

def read_news_test(raw=False):
    return _read_file("MINDsmall_test", "news.tsv", raw)
