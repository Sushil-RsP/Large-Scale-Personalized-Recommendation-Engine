"""Unit tests for Two-Tower Retrieval Model and FAISS Vector Search."""
import os
import torch
import numpy as np
import pytest

from src.retrieval.two_tower import UserTower, ItemTower, TwoTowerRetrievalModel
from src.retrieval.faiss_index import FAISSCandidateRetriever

def test_user_and_item_tower_shapes():
    num_users = 100
    num_items = 200
    num_genres = 16
    embedding_dim = 64

    model = TwoTowerRetrievalModel(num_users, num_items, num_genres, embedding_dim)
    model.eval()

    batch_size = 4
    u_id = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    u_genres = torch.randn(batch_size, num_genres)
    u_stats = torch.randn(batch_size, 2)

    i_id = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    i_genres = torch.randn(batch_size, num_genres)
    i_stats = torch.randn(batch_size, 3)

    with torch.no_grad():
        u_emb, i_emb = model((u_id, u_genres, u_stats), (i_id, i_genres, i_stats))

    assert u_emb.shape == (batch_size, embedding_dim)
    assert i_emb.shape == (batch_size, embedding_dim)

    # Check L2 normalization: norm should equal 1.0 within numerical precision
    u_norms = torch.norm(u_emb, p=2, dim=-1)
    i_norms = torch.norm(i_emb, p=2, dim=-1)
    assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-5)
    assert torch.allclose(i_norms, torch.ones_like(i_norms), atol=1e-5)

def test_faiss_retriever():
    retriever = FAISSCandidateRetriever()
    assert retriever.index.ntotal > 0, "FAISS index should contain indexed candidate vectors"

    # Query with dummy 64-dim vector
    dummy_query = np.random.randn(64).astype(np.float32)
    top_k = 25
    candidates, latency_ms = retriever.retrieve(dummy_query, top_k=top_k)

    assert len(candidates) == top_k, f"Expected {top_k} candidates, got {len(candidates)}"
    assert latency_ms > 0.0
    assert "movieId" in candidates[0]
    assert "retrieval_score" in candidates[0]
    # Scores should be sorted in descending order
    scores = [c["retrieval_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)

