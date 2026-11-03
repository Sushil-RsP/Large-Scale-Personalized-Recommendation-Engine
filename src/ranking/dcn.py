"""
Deep & Cross Network (DCN-v2) for Candidate Re-Ranking.
Paper: Wang et al., 'DCN V2: Improved Deep & Cross Network for Feature Crosses in Web-scale Rank and Recommend Systems'
Features:
- Explicit Cross Network: Computes feature crossing of degree d at layer d with O(d) parameters.
- Deep MLP Network: Captures implicit high-order non-linear combinations.
- Stacking Combination: Combines cross and deep streams for calibrated CTR prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetwork(nn.Module):
    """
    DCN-v2 Cross Network layer:
    x_{l+1} = x_0 * (W_l x_l + b_l) + x_l
    """
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        # x0: [batch_size, input_dim]
        # Reshape to [batch_size, input_dim, 1] for matrix math
        xl = x0.unsqueeze(-1)
        x0_col = x0.unsqueeze(-1)

        for l in range(self.num_layers):
            # W_l^T x_l is a scalar per sample: [batch_size, 1, 1]
            # using broadcast: (input_dim, 1)^T * (input_dim, 1)
            wl = self.weights[l]  # [input_dim, 1]
            bl = self.biases[l]   # [input_dim, 1]

            # Linear transformation: W_l^T x_l + b_l
            # xl is [batch_size, input_dim, 1]
            dot = torch.sum(xl * wl, dim=1, keepdim=True) # [batch_size, 1, 1]
            xl = x0_col * (dot + bl) + xl

        return xl.squeeze(-1)

class DeepAndCrossNetwork(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_genres=16,
        embedding_dim=32,
        num_cross_layers=3,
        deep_layers=(128, 64, 32),
        dropout=0.1
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        # Dense features:
        # user_stats (2) + item_stats (3) + item_genres (num_genres) + retrieval_sim (1)
        self.num_dense = 2 + 3 + num_genres + 1
        total_input_dim = (embedding_dim * 2) + self.num_dense

        # Cross Network
        self.cross_net = CrossNetwork(total_input_dim, num_layers=num_cross_layers)

        # Deep MLP
        mlp_layers = []
        in_dim = total_input_dim
        for out_dim in deep_layers:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.deep_net = nn.Sequential(*mlp_layers)

        # Final prediction head
        combined_dim = total_input_dim + deep_layers[-1]
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_id, item_id, dense_features):
        """
        user_id: [batch_size]
        item_id: [batch_size]
        dense_features: [batch_size, num_dense]
        """
        u_emb = self.user_embedding(user_id)
        i_emb = self.item_embedding(item_id)

        x0 = torch.cat([u_emb, i_emb, dense_features], dim=-1)

        x_cross = self.cross_net(x0)
        x_deep = self.deep_net(x0)

        combined = torch.cat([x_cross, x_deep], dim=-1)
        logits = self.prediction_head(combined)
        return torch.sigmoid(logits).squeeze(-1)

