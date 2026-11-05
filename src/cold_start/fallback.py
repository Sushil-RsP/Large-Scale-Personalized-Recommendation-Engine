import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

class ColdStartEngine:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.movies_df = pd.read_parquet(os.path.join(data_dir, "movies_features.parquet"))
        with open(os.path.join(data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.genres = self.metadata["genres"]

        c = self.movies_df["item_avg_rating"].mean()
        m = self.movies_df["item_popularity"].quantile(0.85)

        v = self.movies_df["item_popularity"]
        r = self.movies_df["item_avg_rating"]
        self.movies_df["bayesian_score"] = (v / (v + m)) * r + (m / (v + m)) * c
        self.popular_items = self.movies_df.sort_values("bayesian_score", ascending=False).copy()

    def get_popular_diverse_fallback(self, top_k=100, max_per_genre=15):
        selected_movies = []
        seen_movie_ids = set()
        genre_counts = {g: 0 for g in self.genres}

        for _, row in self.popular_items.iterrows():
            mid = int(row["movieId"])
            if mid in seen_movie_ids:
                continue

            item_genres = [g for g in self.genres if row.get(f"genre_{g.lower()}", 0.0) == 1.0]

            can_add = False
            if not item_genres:
                can_add = True
            else:
                for g in item_genres:
                    if genre_counts[g] < max_per_genre:
                        can_add = True
                        break

            if can_add:
                seen_movie_ids.add(mid)
                for g in item_genres:
                    genre_counts[g] += 1
                selected_movies.append({
                    "movieId": mid,
                    "title": row["title"],
                    "genres": row["genres"],
                    "bayesian_score": float(row["bayesian_score"]),
                    "item_popularity": int(row["item_popularity"]),
                    "item_avg_rating": float(row["item_avg_rating"]),
                    "recommendation_source": "cold_start_popularity_diversity"
                })

            if len(selected_movies) >= top_k:
                break

        return selected_movies

    def get_content_based_genre_fallback(self, selected_genres, top_k=100):
        if not selected_genres:
            return self.get_popular_diverse_fallback(top_k=top_k)

        norm_genres = [g.capitalize() for g in selected_genres if g.capitalize() in self.genres]
        if not norm_genres:
            return self.get_popular_diverse_fallback(top_k=top_k)

        df = self.movies_df.copy()
        df["genre_match"] = df.apply(
            lambda r: sum(r.get(f"genre_{g.lower()}", 0.0) for g in norm_genres),
            axis=1
        )
        matched_df = df[df["genre_match"] > 0].copy()

        if matched_df.empty:
            return self.get_popular_diverse_fallback(top_k=top_k)

        matched_df["final_score"] = matched_df["genre_match"] * 2.0 + matched_df["bayesian_score"]
        sorted_df = matched_df.sort_values("final_score", ascending=False).head(top_k)

        results = []
        for _, row in sorted_df.iterrows():
            results.append({
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "genre_match_count": int(row["genre_match"]),
                "bayesian_score": float(row["bayesian_score"]),
                "item_popularity": int(row["item_popularity"]),
                "item_avg_rating": float(row["item_avg_rating"]),
                "recommendation_source": "cold_start_content_genre"
            })
        return results
