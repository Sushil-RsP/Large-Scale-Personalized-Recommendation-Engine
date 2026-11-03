"""Unit tests for Ranking Models: DCN-v2 and Wide & Deep."""
import torch
import pytest

from src.ranking.dcn import DeepAndCrossNetwork
from src.ranking.wide_deep import WideAndDeepNetwork

def test_dcn_v2_forward():
    batch_size = 8
    num_users = 50
    num_items = 100
    num_genres = 16
    num_dense = 2 + 3 + num_genres + 1

    dcn = DeepAndCrossNetwork(num_users, num_items, num_genres, embedding_dim=32)
    dcn.eval()

    u_ids = torch.randint(1, num_users, (batch_size,), dtype=torch.long)
    i_ids = torch.randint(1, num_items, (batch_size,), dtype=torch.long)
    dense_feats = torch.randn(batch_size, num_dense, dtype=torch.float32)

    with torch.no_grad():
        preds = dcn(u_ids, i_ids, dense_feats)

    assert preds.shape == (batch_size,)
    # Output should be valid probabilities in [0, 1]
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

def test_wide_deep_forward():
    batch_size = 8
    num_users = 50
    num_items = 100
    num_genres = 16
    num_dense = 2 + 3 + num_genres + 1

    wd = WideAndDeepNetwork(num_users, num_items, num_genres, embedding_dim=32)
    wd.eval()

    u_ids = torch.randint(1, num_users, (batch_size,), dtype=torch.long)
    i_ids = torch.randint(1, num_items, (batch_size,), dtype=torch.long)
    dense_feats = torch.randn(batch_size, num_dense, dtype=torch.float32)

    with torch.no_grad():
        preds = wd(u_ids, i_ids, dense_feats)

    assert preds.shape == (batch_size,)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

