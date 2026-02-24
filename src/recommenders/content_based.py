import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.recommenders.base import BaseRecommender

# Using TF-IDF vectors of news text to build user profiles and recommend similar news.
class ContentBasedRecommender(BaseRecommender):
    def __init__(self, max_features=50000, ngram_range=(1,2)):
        """
        max_features: vocabulary size
        ngram_range: (1,2) = unigrams + bigrams
        """
        self.vectorizer = TfidfVectorizer( # converts news text into TF-IDF vectors
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )

        self.news_index = None
        self.news_tfidf = None # ends up being matrix of shape (num_news_articles × vocabulary_size)
        self.user_profiles = {}


    def fit(self, news_df, user_history_df, text_col="text"):
        """
        news_df: DataFrame with columns [news_id, text]
        user_history_df: DataFrame with columns [user_id, clicked_news_ids]
        """
        news_df = news_df.copy()
        news_df["text"] = news_df["title"] + " " + news_df["abstract"]
        news_df = news_df[["news_id", "text"]]

        user_history_df = user_history_df.copy()
        user_history_df = user_history_df[["user_id", "history"]]

        # fit TF-IDF on all news articles
        self.news_index = news_df["news_id"].tolist()
        self.news_tfidf = self.vectorizer.fit_transform(news_df[text_col])

        # map news_id -> TF-IDF vector index
        news_id_to_idx = {nid: i for i, nid in enumerate(self.news_index)}

        # Bbild user profiles (mean TF-IDF vector of clicked news)
        for _, row in user_history_df.iterrows():
            user = row["user_id"]
            clicked = row["history"]

            # get all articles the user has clicked, 
            idxs = [news_id_to_idx[n] for n in clicked if n in news_id_to_idx]
            if len(idxs) > 0:
                # find their TF-IDF vectors, 
                # and average them to get the user profile
                user_vec = np.asarray(self.news_tfidf[idxs].mean(axis=0))
                self.user_profiles[user] = user_vec
                # user_vec is a 1 × vocabulary_size vector representing the user's interests


    def score(self, user_id):
        """
        Returns similarity scores for all news
        """
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        
        user_vec = self.user_profiles[user_id]
        scores = cosine_similarity(user_vec, self.news_tfidf).flatten()

        # ranking table with news_id and score, sorted by score
        return pd.DataFrame({
            "news_id": self.news_index,
            "score": scores
        }).sort_values("score", ascending=False)


    def recommend(self, user_id, k=10, seen_news=None):
        """
        seen_news: list of news to filter out (already clicked)
        """
        rankings = self.score(user_id)

        if seen_news is not None:
            rankings = rankings[~rankings["news_id"].isin(seen_news)]

        # recommends top-k (unseen) news_ids with highest similarity scores
        return rankings.head(k)["news_id"].tolist()


    def save():
        ...


    def load():
        ...