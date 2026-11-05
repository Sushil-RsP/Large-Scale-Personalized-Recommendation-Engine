import os
import re
import json
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "ml-latest-small")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def extract_year_from_title(title):
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        return int(match.group(1))
    return 2000

def run_pyspark_pipeline(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH, use_spark=True):
    os.makedirs(output_path, exist_ok=True)
    ratings_csv = os.path.join(raw_path, "ratings.csv")
    movies_csv = os.path.join(raw_path, "movies.csv")

    if not os.path.exists(ratings_csv) or not os.path.exists(movies_csv):
        raise FileNotFoundError(f"Ratings or Movies CSV missing from {raw_path}")

    spark_success = False
    if use_spark:
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F

            spark = SparkSession.builder \
                .appName("RecommendationEngineFeaturePipeline") \
                .master("local[2]") \
                .config("spark.driver.host", "127.0.0.1") \
                .config("spark.ui.enabled", "false") \
                .config("spark.sql.shuffle.partitions", "4") \
                .getOrCreate()

            ratings_sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(ratings_csv)
            movies_sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(movies_csv)

            ratings_sdf = ratings_sdf.orderBy("timestamp")

            timestamps = ratings_sdf.select("timestamp").rdd.flatMap(lambda x: x).collect()
            t_train_cutoff = np.percentile(timestamps, 80)
            t_val_cutoff = np.percentile(timestamps, 90)

            item_stats = ratings_sdf.groupBy("movieId").agg(
                F.count("userId").alias("item_popularity"),
                F.avg("rating").alias("item_avg_rating")
            )
            movies_feat = movies_sdf.join(item_stats, on="movieId", how="left").fillna({
                "item_popularity": 0,
                "item_avg_rating": 3.0
            })

            train_ratings = ratings_sdf.filter(F.col("timestamp") <= t_train_cutoff)
            user_stats = train_ratings.groupBy("userId").agg(
                F.count("movieId").alias("user_interaction_count"),
                F.avg("rating").alias("user_avg_rating")
            )

            ratings_with_label = ratings_sdf.withColumn("label", F.when(F.col("rating") >= 3.5, 1.0).otherwise(0.0))

            train_sdf = ratings_with_label.filter(F.col("timestamp") <= t_train_cutoff)
            val_sdf = ratings_with_label.filter((F.col("timestamp") > t_train_cutoff) & (F.col("timestamp") <= t_val_cutoff))
            test_sdf = ratings_with_label.filter(F.col("timestamp") > t_val_cutoff)

            train_df = train_sdf.toPandas()
            val_df = val_sdf.toPandas()
            test_df = test_sdf.toPandas()
            movies_df = movies_feat.toPandas()
            user_df = user_stats.toPandas()

            spark.stop()
            spark_success = True
        except Exception:
            pass

    if not spark_success:
        ratings_df = pd.read_csv(ratings_csv)
        movies_raw_df = pd.read_csv(movies_csv)

        ratings_df = ratings_df.sort_values("timestamp").reset_index(drop=True)
        t_train_cutoff = np.percentile(ratings_df["timestamp"], 80)
        t_val_cutoff = np.percentile(ratings_df["timestamp"], 90)

        item_stats = ratings_df.groupby("movieId").agg(
            item_popularity=("userId", "count"),
            item_avg_rating=("rating", "mean")
        ).reset_index()
        movies_df = pd.merge(movies_raw_df, item_stats, on="movieId", how="left").fillna({
            "item_popularity": 0,
            "item_avg_rating": 3.0
        })

        train_ratings = ratings_df[ratings_df["timestamp"] <= t_train_cutoff]
        user_df = train_ratings.groupby("userId").agg(
            user_interaction_count=("movieId", "count"),
            user_avg_rating=("rating", "mean")
        ).reset_index()

        ratings_df["label"] = (ratings_df["rating"] >= 3.5).astype(float)
        train_df = ratings_df[ratings_df["timestamp"] <= t_train_cutoff].copy()
        val_df = ratings_df[(ratings_df["timestamp"] > t_train_cutoff) & (ratings_df["timestamp"] <= t_val_cutoff)].copy()
        test_df = ratings_df[ratings_df["timestamp"] > t_val_cutoff].copy()

    movies_df["year"] = movies_df["title"].apply(extract_year_from_title)
    for g in ALL_GENRES:
        movies_df[f"genre_{g.lower()}"] = movies_df["genres"].apply(lambda x: 1.0 if g in str(x).split("|") else 0.0)

    user_genre_prefs = compute_user_genre_profiles(train_df, movies_df)
    user_df = pd.merge(user_df, user_genre_prefs, on="userId", how="left").fillna(0.0)

    user_ids = sorted(ratings_df["userId"].unique().tolist() if not spark_success else train_df["userId"].unique().tolist())
    movie_ids = sorted(movies_df["movieId"].unique().tolist())

    user2idx = {int(uid): idx + 1 for idx, uid in enumerate(user_ids)}
    movie2idx = {int(mid): idx + 1 for idx, mid in enumerate(movie_ids)}
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    metadata = {
        "num_users": len(user2idx) + 1,
        "num_items": len(movie2idx) + 1,
        "num_genres": len(ALL_GENRES),
        "genres": ALL_GENRES,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "t_train_cutoff": float(t_train_cutoff),
        "t_val_cutoff": float(t_val_cutoff)
    }

    train_df.to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_path, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_path, "test.parquet"), index=False)
    movies_df.to_parquet(os.path.join(output_path, "movies_features.parquet"), index=False)
    user_df.to_parquet(os.path.join(output_path, "user_features.parquet"), index=False)

    with open(os.path.join(output_path, "vocab_user2idx.json"), "w") as f:
        json.dump(user2idx, f)
    with open(os.path.join(output_path, "vocab_movie2idx.json"), "w") as f:
        json.dump(movie2idx, f)
    with open(os.path.join(output_path, "vocab_idx2movie.json"), "w") as f:
        json.dump(idx2movie, f)
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def compute_user_genre_profiles(train_df, movies_df):
    merged = pd.merge(train_df[train_df["label"] == 1.0], movies_df[["movieId"] + [f"genre_{g.lower()}" for g in ALL_GENRES]], on="movieId")
    genre_cols = [f"genre_{g.lower()}" for g in ALL_GENRES]
    user_genre_sums = merged.groupby("userId")[genre_cols].sum()
    row_sums = user_genre_sums.sum(axis=1).replace(0, 1.0)
    return user_genre_sums.div(row_sums, axis=0).reset_index()

if __name__ == "__main__":
    run_pyspark_pipeline()
