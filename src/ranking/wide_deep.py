"""
Wide & Deep Learning Model for Re-Ranking.
Paper: Cheng et al., 'Wide & Deep Learning for Recommender Systems' (Google, 2016)
Architecture:
- Wide component: Memorizes linear feature crosses and direct effects.
- Deep component: Generalizes to unseen feature combinations using dense embeddings & feedforward layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeepNetwork(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_genres=16,
        embedding_dim=32,
        deep_layers=(128, 64, 32),
        dropout=0.1
    ):
        super().__init__()
        # Deep embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        # Wide linear embeddings / weights
        self.wide_user_linear = nn.Embedding(num_users, 1, padding_idx=0)
        self.wide_item_linear = nn.Embedding(num_items, 1, padding_idx=0)

        # Dense features: user_stats (2) + item_stats (3) + item_genres (num_genres) + retrieval_sim (1)
        self.num_dense = 2 + 3 + num_genres + 1
        self.wide_dense_linear = nn.Linear(self.num_dense, 1)

        # Deep MLP
        total_deep_input_dim = (embedding_dim * 2) + self.num_dense
        mlp_layers = []
        in_dim = total_deep_input_dim
        for out_dim in deep_layers:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.deep_mlp = nn.Sequential(*mlp_layers)
        self.deep_linear = nn.Linear(deep_layers[-1], 1)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_id, item_id, dense_features):
        """
        user_id: [batch_size]
        item_id: [batch_size]
        dense_features: [batch_size, num_dense]
        """
        # Wide component
        wide_u = self.wide_user_linear(user_id)
        wide_i = self.wide_item_linear(item_id)
        wide_d = self.wide_dense_linear(dense_features)
        wide_out = wide_u + wide_i + wide_d

        # Deep component
        u_emb = self.user_embedding(user_id)
        i_emb = self.item_embedding(item_id)
        deep_in = torch.cat([u_emb, i_emb, dense_features], dim=-1)
        deep_feat = self.deep_mlp(deep_in)
        deep_out = self.deep_linear(deep_feat)

        # Combined output
        logits = wide_out + deep_out + self.bias
        return torch.sigmoid(logits).squeeze(-1)

