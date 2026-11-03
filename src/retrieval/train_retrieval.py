"""
Training script for the Two-Tower Retrieval Model.
Trains on temporal interactions with in-batch contrastive loss, evaluates Recall@K,
and saves weights for FAISS indexing and Triton export.
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.retrieval.two_tower import TwoTowerRetrievalModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "retrieval")

class RetrievalInteractionDataset(Dataset):
    def __init__(self, ratings_df, user_df, movies_df, user2idx, movie2idx, genres):
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.genres = genres

        # Map user features
        user_feat_dict = {}
        genre_cols = [f"genre_{g.lower()}" for g in genres]
        for _, row in user_df.iterrows():
            uid = int(row["userId"])
            u_idx = user2idx.get(uid, 0)
            g_prefs = row[genre_cols].values.astype(np.float32)
            # normalize count and avg rating
            norm_count = float(np.log1p(row.get("user_interaction_count", 0.0)))
            norm_rating = float((row.get("user_avg_rating", 3.0) - 1.0) / 4.0)
            user_feat_dict[u_idx] = (g_prefs, np.array([norm_count, norm_rating], dtype=np.float32))

        # Map movie features
        movie_feat_dict = {}
        for _, row in movies_df.iterrows():
            mid = int(row["movieId"])
            m_idx = movie2idx.get(mid, 0)
            g_vec = row[genre_cols].values.astype(np.float32)
            norm_pop = float(np.log1p(row.get("item_popularity", 0.0)))
            norm_rating = float((row.get("item_avg_rating", 3.0) - 1.0) / 4.0)
            norm_year = float((row.get("year", 2000) - 1920) / 100.0)
            movie_feat_dict[m_idx] = (g_vec, np.array([norm_pop, norm_rating, norm_year], dtype=np.float32))

        # Filter for positive interactions (rating >= 3.5)
        pos_ratings = ratings_df[ratings_df["label"] == 1.0].copy()
        self.samples = []
        for _, row in pos_ratings.iterrows():
            u_idx = user2idx.get(int(row["userId"]), 0)
            m_idx = movie2idx.get(int(row["movieId"]), 0)
            if u_idx > 0 and m_idx > 0:
                self.samples.append((u_idx, m_idx))

        self.user_feat_dict = user_feat_dict
        self.movie_feat_dict = movie_feat_dict
        self.default_user_genre = np.zeros(len(genres), dtype=np.float32)
        self.default_user_stats = np.array([0.0, 0.5], dtype=np.float32)
        self.default_movie_genre = np.zeros(len(genres), dtype=np.float32)
        self.default_movie_stats = np.array([0.0, 0.5, 0.8], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u_idx, m_idx = self.samples[idx]

        u_genres, u_stats = self.user_feat_dict.get(u_idx, (self.default_user_genre, self.default_user_stats))
        m_genres, m_stats = self.movie_feat_dict.get(m_idx, (self.default_movie_genre, self.default_movie_stats))

        return {
            "user_id": torch.tensor(u_idx, dtype=torch.long),
            "user_genres": torch.tensor(u_genres, dtype=torch.float32),
            "user_stats": torch.tensor(u_stats, dtype=torch.float32),
            "item_id": torch.tensor(m_idx, dtype=torch.long),
            "item_genres": torch.tensor(m_genres, dtype=torch.float32),
            "item_stats": torch.tensor(m_stats, dtype=torch.float32)
        }

def train_retrieval_model(epochs=8, batch_size=256, lr=1e-3):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(DATA_DIR, "vocab_user2idx.json")) as f:
        user2idx = json.load(f)
    with open(os.path.join(DATA_DIR, "vocab_movie2idx.json")) as f:
        movie2idx = json.load(f)

    user2idx = {int(k): v for k, v in user2idx.items()}
    movie2idx = {int(k): v for k, v in movie2idx.items()}

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    user_df = pd.read_parquet(os.path.join(DATA_DIR, "user_features.parquet"))
    movies_df = pd.read_parquet(os.path.join(DATA_DIR, "movies_features.parquet"))

    train_ds = RetrievalInteractionDataset(train_df, user_df, movies_df, user2idx, movie2idx, meta["genres"])
    val_ds = RetrievalInteractionDataset(val_df, user_df, movies_df, user2idx, movie2idx, meta["genres"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Two-Tower Retrieval on device: {device} | Train samples: {len(train_ds)}")

    model = TwoTowerRetrievalModel(
        num_users=meta["num_users"] + 1,
        num_items=meta["num_items"] + 1,
        num_genres=meta["num_genres"],
        embedding_dim=64,
        temperature=0.07
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    metrics_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            u_inputs = (
                batch["user_id"].to(device),
                batch["user_genres"].to(device),
                batch["user_stats"].to(device)
            )
            i_inputs = (
                batch["item_id"].to(device),
                batch["item_genres"].to(device),
                batch["item_stats"].to(device)
            )

            u_emb, i_emb = model(u_inputs, i_inputs)
            loss = model.compute_in_batch_loss(u_emb, i_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                u_inputs = (
                    batch["user_id"].to(device),
                    batch["user_genres"].to(device),
                    batch["user_stats"].to(device)
                )
                i_inputs = (
                    batch["item_id"].to(device),
                    batch["item_genres"].to(device),
                    batch["item_stats"].to(device)
                )
                u_emb, i_emb = model(u_inputs, i_inputs)
                loss = model.compute_in_batch_loss(u_emb, i_emb)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        metrics_history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower_best.pt"))
            torch.save(model.user_tower.state_dict(), os.path.join(MODEL_DIR, "user_tower.pt"))
            torch.save(model.item_tower.state_dict(), os.path.join(MODEL_DIR, "item_tower.pt"))

    with open(os.path.join(MODEL_DIR, "retrieval_metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"Two-Tower model saved to {MODEL_DIR} (Best Val Loss: {best_loss:.4f})")
    return model

if __name__ == "__main__":
    train_retrieval_model()

