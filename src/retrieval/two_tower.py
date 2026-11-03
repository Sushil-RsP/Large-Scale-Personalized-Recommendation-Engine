"""
Two-Tower Candidate Retrieval Neural Network.
Architecture:
- User Tower: Embeds user ID, historical genre preferences, and engagement statistics.
- Item Tower: Embeds item ID, multi-hot genre features, popularity, and release year.
- Metric: Cosine similarity via L2-normalized 64-dimensional latent space.
- Dual export capability: PyTorch (.pt) and ONNX (.onnx) for Triton Inference Server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, num_genres=16, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.genre_dense = nn.Linear(num_genres, embedding_dim)
        self.stats_dense = nn.Linear(2, embedding_dim)  # count, avg_rating

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_id, genre_prefs, stats):
        """
        user_id: LongTensor [batch_size]
        genre_prefs: FloatTensor [batch_size, num_genres]
        stats: FloatTensor [batch_size, 2] (norm_count, norm_avg_rating)
        """
        u_emb = self.user_embedding(user_id)
        g_emb = F.relu(self.genre_dense(genre_prefs))
        s_emb = F.relu(self.stats_dense(stats))

        combined = torch.cat([u_emb, g_emb, s_emb], dim=-1)
        out = self.mlp(combined)
        # L2-normalize to unit sphere for cosine similarity
        return F.normalize(out, p=2, dim=-1)

class ItemTower(nn.Module):
    def __init__(self, num_items, num_genres=16, embedding_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.genre_dense = nn.Linear(num_genres, embedding_dim)
        self.stats_dense = nn.Linear(3, embedding_dim)  # popularity, avg_rating, normalized_year

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, item_id, item_genres, stats):
        """
        item_id: LongTensor [batch_size]
        item_genres: FloatTensor [batch_size, num_genres]
        stats: FloatTensor [batch_size, 3]
        """
        i_emb = self.item_embedding(item_id)
        g_emb = F.relu(self.genre_dense(item_genres))
        s_emb = F.relu(self.stats_dense(stats))

        combined = torch.cat([i_emb, g_emb, s_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)

class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres=16, embedding_dim=64, temperature=0.07):
        super().__init__()
        self.user_tower = UserTower(num_users, num_genres, embedding_dim)
        self.item_tower = ItemTower(num_items, num_genres, embedding_dim)
        self.temperature = temperature

    def forward(self, user_inputs, item_inputs):
        """
        Computes cosine similarity logits between user and item representations.
        """
        user_emb = self.user_tower(*user_inputs)
        item_emb = self.item_tower(*item_inputs)
        return user_emb, item_emb

    def compute_in_batch_loss(self, user_emb, item_emb):
        """
        In-batch sampled softmax loss with temperature scaling (standard TFRS / YouTube two-tower).
        """
        # Similarity matrix: [batch_size, batch_size]
        logits = torch.matmul(user_emb, item_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

