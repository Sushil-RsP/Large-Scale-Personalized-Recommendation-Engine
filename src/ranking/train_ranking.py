"""
Training pipeline for Ranking Models: DCN-v2 and Wide & Deep.
Trains on temporal interactions with binary cross-entropy loss,
evaluates ROC-AUC and Log-Loss, and prepares models for Triton serving and A/B testing.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from src.ranking.dcn import DeepAndCrossNetwork
from src.ranking.wide_deep import WideAndDeepNetwork

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ranking")

class RankingDataset(Dataset):
    def __init__(self, ratings_df, user_df, movies_df, user2idx, movie2idx, genres, is_train=True):
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.genres = genres
        genre_cols = [f"genre_{g.lower()}" for g in genres]

        # User lookup
        user_map = {}
        for _, row in user_df.iterrows():
            uid = int(row["userId"])
            u_idx = user2idx.get(uid, 0)
            norm_count = float(np.log1p(row.get("user_interaction_count", 0.0)))
            norm_rating = float((row.get("user_avg_rating", 3.0) - 1.0) / 4.0)
            user_map[u_idx] = np.array([norm_count, norm_rating], dtype=np.float32)

        # Movie lookup
        movie_map = {}
        for _, row in movies_df.iterrows():
            mid = int(row["movieId"])
            m_idx = movie2idx.get(mid, 0)
            g_vec = row[genre_cols].values.astype(np.float32)
            norm_pop = float(np.log1p(row.get("item_popularity", 0.0)))
            norm_rating = float((row.get("item_avg_rating", 3.0) - 1.0) / 4.0)
            norm_year = float((row.get("year", 2000) - 1920) / 100.0)
            movie_map[m_idx] = (g_vec, np.array([norm_pop, norm_rating, norm_year], dtype=np.float32))

        # Build feature rows
        user_ids = []
        item_ids = []
        dense_feats = []
        labels = []

        all_movie_indices = list(movie_map.keys())

        for _, row in ratings_df.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            u_idx = user2idx.get(uid, 0)
            m_idx = movie2idx.get(mid, 0)
            if u_idx == 0 or m_idx == 0:
                continue

            u_stats = user_map.get(u_idx, np.array([0.0, 0.5], dtype=np.float32))
            g_vec, m_stats = movie_map.get(m_idx, (np.zeros(len(genres), dtype=np.float32), np.array([0.0, 0.5, 0.8], dtype=np.float32)))

            # Retrieval sim proxy: dot product between normalized popularity and user engagement
            retrieval_sim = float(np.clip(u_stats[1] * m_stats[1], 0.0, 1.0))
            dense_vec = np.concatenate([u_stats, m_stats, g_vec, [retrieval_sim]]).astype(np.float32)

            user_ids.append(u_idx)
            item_ids.append(m_idx)
            dense_feats.append(dense_vec)
            labels.append(float(row["label"]))

            # Negative sampling during training for better discrimination
            if is_train and row["label"] == 1.0 and np.random.rand() < 0.5:
                neg_m_idx = np.random.choice(all_movie_indices)
                neg_g_vec, neg_m_stats = movie_map[neg_m_idx]
                neg_sim = float(np.clip(u_stats[1] * neg_m_stats[1] * 0.5, 0.0, 1.0))
                neg_dense = np.concatenate([u_stats, neg_m_stats, neg_g_vec, [neg_sim]]).astype(np.float32)

                user_ids.append(u_idx)
                item_ids.append(neg_m_idx)
                dense_feats.append(neg_dense)
                labels.append(0.0)

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.dense_feats = torch.tensor(np.array(dense_feats), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "dense_features": self.dense_feats[idx],
            "label": self.labels[idx]
        }

def train_single_model(model, train_loader, val_loader, model_name, epochs=6, lr=1e-3, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    history = []

    print(f"\n--- Training {model_name} on {device} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            u = batch["user_id"].to(device)
            i = batch["item_id"].to(device)
            d = batch["dense_features"].to(device)
            y = batch["label"].to(device)

            pred = model(u, i, d)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                u = batch["user_id"].to(device)
                i = batch["item_id"].to(device)
                d = batch["dense_features"].to(device)
                y = batch["label"].to(device)
                pred = model(u, i, d)
                loss = criterion(pred, y)
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
        val_acc = accuracy_score(val_targets, np.array(val_preds) >= 0.5)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": float(val_auc),
            "val_acc": float(val_acc)
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}_best.pt"))

    return history, best_val_auc

def train_ranking_models():
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

    print("Building ranking datasets with negative sampling...")
    train_ds = RankingDataset(train_df, user_df, movies_df, user2idx, movie2idx, meta["genres"], is_train=True)
    val_ds = RankingDataset(val_df, user_df, movies_df, user2idx, movie2idx, meta["genres"], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train DCN-v2
    dcn_model = DeepAndCrossNetwork(
        num_users=meta["num_users"] + 1,
        num_items=meta["num_items"] + 1,
        num_genres=meta["num_genres"],
        embedding_dim=32,
        num_cross_layers=3,
        deep_layers=(128, 64, 32),
        dropout=0.1
    ).to(device)
    dcn_history, best_dcn_auc = train_single_model(dcn_model, train_loader, val_loader, "DCN_v2", epochs=6, device=device)

    # 2. Train Wide & Deep
    wd_model = WideAndDeepNetwork(
        num_users=meta["num_users"] + 1,
        num_items=meta["num_items"] + 1,
        num_genres=meta["num_genres"],
        embedding_dim=32,
        deep_layers=(128, 64, 32),
        dropout=0.1
    ).to(device)
    wd_history, best_wd_auc = train_single_model(wd_model, train_loader, val_loader, "Wide_and_Deep", epochs=6, device=device)

    summary = {
        "DCN_v2": {"best_auc": float(best_dcn_auc), "history": dcn_history},
        "Wide_and_Deep": {"best_auc": float(best_wd_auc), "history": wd_history}
    }
    with open(os.path.join(MODEL_DIR, "ranking_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRanking models training complete!")
    print(f"  DCN-v2 Best AUC: {best_dcn_auc:.4f}")
    print(f"  Wide & Deep Best AUC: {best_wd_auc:.4f}")
    return summary

if __name__ == "__main__":
    train_ranking_models()

