"""Baseline recommenders."""

from src.recommenders.baseline.popular import PopularRecommender
from src.recommenders.baseline.random import RandomRecommender

__all__ = ["PopularRecommender", "RandomRecommender"]
