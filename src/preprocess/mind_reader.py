"""Utilities for reading and caching MIND data splits."""
import pandas as pd

BASE_PATH = "../../data/"

# Anticipating that the data is stored in "tdt4215-news-recommender/data"

def read_behaviors_train():
    return pd.read_csv(BASE_PATH + "MINDsmall_train/behaviours.tsv", sep="\t")

def read_behaviors_test():
    return pd.read_csv(BASE_PATH + "MINDsmall_dev/behaviours.tsv", sep="\t")

def read_news_train():
    return pd.read_csv(BASE_PATH + "MINDsmall_train/news.tsv", sep="\t")

def read_news_test():
    return pd.read_csv(BASE_PATH + "MINDsmall_test/news.tsv", sep="\t")


