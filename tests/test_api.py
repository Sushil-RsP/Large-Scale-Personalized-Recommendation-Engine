"""Unit & Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_system_info_endpoint():
    resp = client.get("/system/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["total_items"] > 0
    assert data["faiss_total_indexed"] > 0
    assert "cache" in data

def test_warm_recommendation_endpoint():
    payload = {
        "user_id": 1,
        "top_k": 5,
        "candidate_k": 50,
        "ranking_model": "dcn",
        "bypass_cache": True
    }
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == 1
    assert data["is_cold_start"] is False
    assert len(data["recommendations"]) == 5
    assert "predicted_ctr" in data["recommendations"][0]
    assert "stage_latencies_ms" in data

def test_cold_start_recommendation_endpoint():
    payload = {
        "user_id": None,
        "top_k": 5,
        "ranking_model": "ab_test",
        "onboarding_genres": ["Action", "Sci-Fi"]
    }
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_cold_start"] is True
    assert data["cold_start_strategy"] == "content_based_genre_onboarding"
    assert len(data["recommendations"]) == 5

def test_ab_test_metrics_and_event():
    # Record an event
    event_payload = {"variant": "variant_B", "event_type": "click"}
    post_resp = client.post("/ab-test/event", json=event_payload)
    assert post_resp.status_code == 200

    # Query metrics
    get_resp = client.get("/ab-test/metrics")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert "variant_A" in data
    assert "variant_B" in data
    assert "analysis" in data

