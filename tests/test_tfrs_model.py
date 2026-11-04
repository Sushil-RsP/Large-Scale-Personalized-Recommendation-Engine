"""
Unit tests for the TFRS (TensorFlow Recommenders) Two-Tower module.
Tests gracefully skip if TensorFlow is not installed on this machine.
"""
import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed – skipping TFRS tests")
tfrs = pytest.importorskip("tensorflow_recommenders", reason="tensorflow-recommenders not installed – skipping TFRS tests")

from src.retrieval.two_tower_tfrs import build_user_model, build_movie_model, TFRSTwoTowerModel


def test_user_model_output_shape():
    model = build_user_model(num_users=100, embedding_dim=32)
    out = model(tf.constant([1, 2, 3], dtype=tf.int32))
    assert out.shape == (3, 32), f"Expected (3,32), got {out.shape}"


def test_item_model_output_shape():
    model = build_movie_model(num_movies=200, embedding_dim=32)
    out = model(tf.constant([10, 20], dtype=tf.int32))
    assert out.shape == (2, 32), f"Expected (2,32), got {out.shape}"


def test_user_embeddings_l2_normalized():
    """TFRS two-tower uses L2-normalized embeddings for cosine similarity scoring."""
    model = build_user_model(num_users=100, embedding_dim=32)
    out = model(tf.constant([1, 2, 3, 4], dtype=tf.int32))
    norms = tf.norm(out, axis=-1).numpy()
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-5,
                               err_msg="User tower embeddings are not L2-normalized")


def test_item_embeddings_l2_normalized():
    model = build_movie_model(num_movies=200, embedding_dim=32)
    out = model(tf.constant([10, 20, 30], dtype=tf.int32))
    norms = tf.norm(out, axis=-1).numpy()
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5,
                               err_msg="Item tower embeddings are not L2-normalized")

