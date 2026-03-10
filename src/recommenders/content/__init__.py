"""Content-based recommenders."""

from src.recommenders.content.entity import EntityContentRecommender
from src.recommenders.content.tfidf import TfidfContentRecommender

__all__ = [
    "TfidfContentRecommender",
    "EntityContentRecommender",
]
