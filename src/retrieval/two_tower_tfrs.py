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

def build_user_model(num_users: int, embedding_dim: int = 64):
    return tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(vocabulary=None, mask_token=None, output_mode="int"),
        tf.keras.layers.Embedding(num_users + 1, embedding_dim),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
    ], name="user_tower")

def build_movie_model(num_movies: int, embedding_dim: int = 64):
    return tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(vocabulary=None, mask_token=None, output_mode="int"),
        tf.keras.layers.Embedding(num_movies + 1, embedding_dim),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
    ], name="item_tower")

class TFRSTwoTowerModel(tfrs.Model):
    def __init__(self, user_model, movie_model, item_candidate_dataset, embedding_dim: int = 64):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
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

def _load_tf_datasets(train_df, movies_df, user2idx, movie2idx):
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

def build_and_train_tfrs_model(
    embedding_dim: int = 64,
    batch_size: int = 512,
    epochs: int = 5
):
    if not TF_AVAILABLE:
        raise ImportError(
            f"TensorFlow is required for TFRS two-tower training. "
            f"Original error: {_TF_ERR_MSG}"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

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

    train_ds, item_candidates_ds = _load_tf_datasets(train_df, movies_df, user2idx, movie2idx)
    val_ds, _ = _load_tf_datasets(val_df, movies_df, user2idx, movie2idx)

    train_ds = train_ds.shuffle(10000).batch(batch_size).cache()
    val_ds = val_ds.batch(batch_size).cache()

    user_model = build_user_model(num_users, embedding_dim)
    movie_model = build_movie_model(num_items, embedding_dim)

    tfrs_model = TFRSTwoTowerModel(
        user_model=user_model,
        movie_model=movie_model,
        item_candidate_dataset=item_candidates_ds,
        embedding_dim=embedding_dim
    )
    tfrs_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    history = tfrs_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1
    )

    all_movie_ids_sorted = sorted([v for v in movie2idx.values() if v > 0])
    item_ids_tensor = tf.constant(all_movie_ids_sorted, dtype=tf.int32)
    item_embeddings = movie_model(item_ids_tensor).numpy().astype(np.float32)

    os.makedirs(FAISS_DIR, exist_ok=True)
    np.save(os.path.join(FAISS_DIR, "item_embeddings_tfrs.npy"), item_embeddings)
    np.save(os.path.join(FAISS_DIR, "item_ids_tfrs.npy"), np.array(all_movie_ids_sorted))

    user_model.save(os.path.join(MODEL_DIR, "user_tower_tfrs"))
    movie_model.save(os.path.join(MODEL_DIR, "item_tower_tfrs"))

    tf.saved_model.save(user_model, os.path.join(MODEL_DIR, "user_tower_savedmodel"))
    tf.saved_model.save(movie_model, os.path.join(MODEL_DIR, "item_tower_savedmodel"))

    import faiss
    dim = item_embeddings.shape[1]
    faiss.normalize_L2(item_embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings)
    faiss.write_index(index, os.path.join(FAISS_DIR, "item_index_tfrs.faiss"))

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict["embedding_dim"] = embedding_dim
    hist_dict["epochs"] = epochs

    with open(os.path.join(MODEL_DIR, "tfrs_training_metrics.json"), "w") as f:
        json.dump(hist_dict, f, indent=2)

    return hist_dict

if __name__ == "__main__":
    metrics = build_and_train_tfrs_model()
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("embedding_dim", "epochs")}, indent=2))
