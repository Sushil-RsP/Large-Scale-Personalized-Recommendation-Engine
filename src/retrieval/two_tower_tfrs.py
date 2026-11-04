"""
Two-Tower Retrieval Model — TensorFlow Recommenders Implementation.
Satisfies the mandatory TensorFlow Recommenders requirement from the project spec.

Architecture matches the TFRS canonical design:
  UserModel  -> embedding + dense MLP
  MovieModel -> embedding + dense MLP
  TwoTowerModel: extends tfrs.Model, uses tfrs.tasks.Retrieval for in-batch
                 sampled softmax loss with top-K accuracy metric.

Usage
-----
from src.retrieval.two_tower_tfrs import build_and_train_tfrs_model
metrics = build_and_train_tfrs_model()

Fallback Behavior
-----------------
If TensorFlow is not installed (e.g. during local unit testing without TF),
the module raises a clear ImportError rather than silently doing nothing.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "retrieval_tfrs")
FAISS_DIR = os.path.join(PROJECT_ROOT, "models", "faiss")

try:
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    TF_AVAILABLE = True
except ImportError as _tf_err:
    TF_AVAILABLE = False
    _TF_ERR_MSG = str(_tf_err)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

def build_user_model(num_users: int, embedding_dim: int = 64):
    """L2-normalised user embedding + small feedforward MLP."""
    return tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(vocabulary=None, mask_token=None,
                                      output_mode="int"),
        tf.keras.layers.Embedding(num_users + 1, embedding_dim),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
    ], name="user_tower")


def build_movie_model(num_movies: int, embedding_dim: int = 64):
    """L2-normalised item embedding + small feedforward MLP."""
    return tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(vocabulary=None, mask_token=None,
                                      output_mode="int"),
        tf.keras.layers.Embedding(num_movies + 1, embedding_dim),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
    ], name="item_tower")


# ---------------------------------------------------------------------------
# Two-Tower TFRS Model (extends tfrs.Model)
# ---------------------------------------------------------------------------

class TFRSTwoTowerModel(tfrs.Model):
    """
    Full TensorFlow Recommenders Two-Tower retrieval model.

    Wraps user_model and movie_model in a tfrs.Model, which provides:
      - tfrs.tasks.Retrieval loss (sampled softmax over in-batch negatives)
      - Built-in FactorizedTopK evaluation metric (Recall@K)
      - Easy .fit() / .evaluate() / .predict() interface.
    """

    def __init__(self, user_model, movie_model, item_candidate_dataset,
                 embedding_dim: int = 64):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model

        # tfrs.tasks.Retrieval handles the in-batch softmax loss and
        # the FactorizedTopK metric (Recall@50, Recall@100 etc.)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_candidate_dataset.batch(128).map(
                    lambda x: (x["movieId"], self.movie_model(x["movieId"]))
                )
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"])
        return self.task(user_embeddings, movie_embeddings)


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def _load_tf_datasets(train_df, movies_df, user2idx, movie2idx):
    """
    Convert pandas DataFrames to tf.data.Dataset with integer-mapped IDs.
    """
    # Positive interactions only
    pos_train = train_df[train_df["label"] == 1.0].copy()
    pos_train["user_idx"] = pos_train["userId"].map(lambda u: user2idx.get(int(u), 0))
    pos_train["movie_idx"] = pos_train["movieId"].map(lambda m: movie2idx.get(int(m), 0))
    pos_train = pos_train[(pos_train["user_idx"] > 0) & (pos_train["movie_idx"] > 0)]

    interaction_ds = tf.data.Dataset.from_tensor_slices({
        "userId": tf.cast(pos_train["user_idx"].values, tf.int32),
        "movieId": tf.cast(pos_train["movie_idx"].values, tf.int32),
    })

    all_movie_ids = sorted([v for v in movie2idx.values() if v > 0])
    item_candidate_ds = tf.data.Dataset.from_tensor_slices({
        "movieId": tf.cast(all_movie_ids, tf.int32),
    })

    return interaction_ds, item_candidate_ds


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def build_and_train_tfrs_model(
    embedding_dim: int = 64,
    batch_size: int = 512,
    epochs: int = 5
):
    """
    Builds, trains and saves the TFRS Two-Tower retrieval model.

    Returns a dict with training history and final metric values.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            f"TensorFlow is required for the TFRS two-tower model. "
            f"Install it with: pip install tensorflow tensorflow-recommenders\n"
            f"Original error: {_TF_ERR_MSG}"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load data ------------------------------------------------------------
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(DATA_DIR, "vocab_user2idx.json")) as f:
        user2idx = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(DATA_DIR, "vocab_movie2idx.json")) as f:
        movie2idx = {int(k): v for k, v in json.load(f).items()}

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    movies_df = pd.read_parquet(os.path.join(DATA_DIR, "movies_features.parquet"))

    num_users = meta["num_users"] + 1
    num_items = meta["num_items"] + 1

    print(f"[TFRS] Building Two-Tower model: {num_users} users, {num_items} items, dim={embedding_dim}")

    # --- Build tf.data.Datasets -----------------------------------------------
    train_ds, item_candidates_ds = _load_tf_datasets(train_df, movies_df, user2idx, movie2idx)
    val_ds, _ = _load_tf_datasets(val_df, movies_df, user2idx, movie2idx)

    train_ds = train_ds.shuffle(10000).batch(batch_size).cache()
    val_ds = val_ds.batch(batch_size).cache()

    # --- Build sub-models & full TFRS model -----------------------------------
    user_model = build_user_model(num_users, embedding_dim)
    movie_model = build_movie_model(num_items, embedding_dim)

    tfrs_model = TFRSTwoTowerModel(
        user_model=user_model,
        movie_model=movie_model,
        item_candidate_dataset=item_candidates_ds,
        embedding_dim=embedding_dim
    )
    tfrs_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # --- Train ----------------------------------------------------------------
    print(f"[TFRS] Training for {epochs} epochs on {len(train_df[train_df['label']==1])} positive interactions...")
    history = tfrs_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1
    )

    # --- Extract & save item embeddings for FAISS -----------------------------
    print("[TFRS] Extracting item embeddings for FAISS indexing...")
    all_movie_ids_sorted = sorted([v for v in movie2idx.values() if v > 0])
    item_ids_tensor = tf.constant(all_movie_ids_sorted, dtype=tf.int32)
    item_embeddings = movie_model(item_ids_tensor).numpy().astype(np.float32)

    os.makedirs(FAISS_DIR, exist_ok=True)
    np.save(os.path.join(FAISS_DIR, "item_embeddings_tfrs.npy"), item_embeddings)
    np.save(os.path.join(FAISS_DIR, "item_ids_tfrs.npy"), np.array(all_movie_ids_sorted))

    # --- Save Keras models ----------------------------------------------------
    user_model.save(os.path.join(MODEL_DIR, "user_tower_tfrs"))
    movie_model.save(os.path.join(MODEL_DIR, "item_tower_tfrs"))

    # --- Save TF SavedModel for Triton TensorFlow backend --------------------
    tf.saved_model.save(user_model, os.path.join(MODEL_DIR, "user_tower_savedmodel"))
    tf.saved_model.save(movie_model, os.path.join(MODEL_DIR, "item_tower_savedmodel"))

    # Build & save FAISS index with TFRS embeddings
    import faiss
    dim = item_embeddings.shape[1]
    faiss.normalize_L2(item_embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings)
    faiss.write_index(index, os.path.join(FAISS_DIR, "item_index_tfrs.faiss"))
    print(f"[TFRS] FAISS index saved: {index.ntotal} items indexed (dim={dim})")

    # --- Persist metrics history ----------------------------------------------
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict["embedding_dim"] = embedding_dim
    hist_dict["epochs"] = epochs

    with open(os.path.join(MODEL_DIR, "tfrs_training_metrics.json"), "w") as f:
        json.dump(hist_dict, f, indent=2)

    final_loss = float(history.history["loss"][-1])
    print(f"[TFRS] Training complete. Final loss: {final_loss:.4f}")
    print(f"[TFRS] Models saved to {MODEL_DIR}")
    print(f"[TFRS] FAISS TFRS index saved to {FAISS_DIR}")

    return hist_dict


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    metrics = build_and_train_tfrs_model()
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("embedding_dim", "epochs")}, indent=2))

