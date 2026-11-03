"""Unit tests for Redis & in-memory recommendation cache."""
import time
import pytest
from src.cache.redis_cache import RecommendationCache, InMemoryCache

def test_in_memory_cache_expiry():
    cache = InMemoryCache(default_ttl=1)
    cache.set("key1", "val1", ex=1)
    assert cache.get("key1") == "val1"

    time.sleep(1.2)
    assert cache.get("key1") is None, "Key should expire after TTL"

def test_recommendation_cache_workflow():
    cache = RecommendationCache(default_ttl=300)
    user_id = 12345
    dummy_recs = [{"movieId": 1, "title": "Test Movie"}]

    # Initially cache miss
    cached = cache.get_user_recommendations(user_id, top_k=5, model_name="dcn")
    assert cached is None

    # Set recommendations
    cache.set_user_recommendations(user_id, dummy_recs, top_k=5, model_name="dcn")

    # Now cache hit
    hit_recs = cache.get_user_recommendations(user_id, top_k=5, model_name="dcn")
    assert hit_recs == dummy_recs

    # Invalidate user
    cache.invalidate_user(user_id)
    assert cache.get_user_recommendations(user_id, top_k=5, model_name="dcn") is None

