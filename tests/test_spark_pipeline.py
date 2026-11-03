"""Unit tests for Spark feature engineering and temporal splitting."""
import os
import json
import pytest
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

def test_temporal_split_no_leakage():
    """Verify that train, validation, and test datasets do not have temporal leakage."""
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    assert len(train_df) > 0, "Train dataset should not be empty"
    assert len(val_df) > 0, "Validation dataset should not be empty"
    assert len(test_df) > 0, "Test dataset should not be empty"

    max_train_t = train_df["timestamp"].max()
    min_val_t = val_df["timestamp"].min()
    max_val_t = val_df["timestamp"].max()
    min_test_t = test_df["timestamp"].min()

    assert max_train_t <= min_val_t, f"Temporal leakage: max train timestamp {max_train_t} > min val timestamp {min_val_t}"
    assert max_val_t <= min_test_t, f"Temporal leakage: max val timestamp {max_val_t} > min test timestamp {min_test_t}"

def test_feature_integrity():
    """Verify movie and user feature tables have valid columns and shapes."""
    movies_df = pd.read_parquet(os.path.join(DATA_DIR, "movies_features.parquet"))
    user_df = pd.read_parquet(os.path.join(DATA_DIR, "user_features.parquet"))

    assert "item_popularity" in movies_df.columns
    assert "item_avg_rating" in movies_df.columns
    assert "year" in movies_df.columns
    assert "genre_action" in movies_df.columns

    assert "user_interaction_count" in user_df.columns
    assert "user_avg_rating" in user_df.columns
    assert "genre_action" in user_df.columns

    # Verify no nulls in critical stats
    assert not movies_df["item_popularity"].isnull().any()
    assert not user_df["user_interaction_count"].isnull().any()

