"""
FastAPI Serving Gateway & Interactive Dashboard Backend.
Provides endpoints for:
- Two-Stage Personalized Recommendations (with Redis Caching & Cold-Start handling)
- Online A/B Testing traffic routing, event tracking, and statistical telemetry
- Offline Benchmark Metrics
- System Health and Cache Management
"""

import os
import sys
import time
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.serving.pipeline import RecommendationPipeline
from src.evaluation.ab_testing import ABTestingManager

app = FastAPI(
    title="Large-Scale Two-Stage Recommendation Engine",
    description="Production-grade two-tower FAISS retrieval and deep ranking engine (DCN-v2 & Wide & Deep)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline and A/B test instances
pipeline: Optional[RecommendationPipeline] = None
ab_manager = ABTestingManager()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = RecommendationPipeline()
    except Exception as e:
        print(f"Warning during pipeline initialization: {e}")

class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="User ID. Null triggers cold-start fallback.")
    top_k: int = Field(10, ge=1, le=50, description="Number of final recommendations to return.")
    candidate_k: int = Field(100, ge=10, le=500, description="Stage 1 FAISS candidates to retrieve.")
    ranking_model: str = Field("ab_test", description="'dcn', 'wide_deep', or 'ab_test'.")
    onboarding_genres: Optional[List[str]] = Field(None, description="Genres chosen during onboarding for cold users.")
    bypass_cache: bool = Field(False, description="Set True to bypass Redis cache.")

class EventLogRequest(BaseModel):
    variant: str = Field(..., description="'variant_A' or 'variant_B'")
    event_type: str = Field("click", description="'click' or 'conversion'")

@app.get("/", response_class=HTMLResponse)
def index():
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Recommendation Engine Backend is Running</h1>"

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    global pipeline, ab_manager
    if pipeline is None:
        pipeline = RecommendationPipeline()

    model_to_use = req.ranking_model.lower()
    chosen_variant = None

    # Handle A/B Test routing
    if model_to_use == "ab_test":
        target_uid = req.user_id if req.user_id is not None else 0
        chosen_variant = ab_manager.assign_variant(target_uid)
        model_to_use = ab_manager.variants[chosen_variant]["model"]

    res = pipeline.recommend(
        user_id=req.user_id,
        top_k=req.top_k,
        candidate_k=req.candidate_k,
        ranking_model=model_to_use,
        onboarding_genres=req.onboarding_genres,
        bypass_cache=req.bypass_cache
    )

    if chosen_variant:
        res["ab_test_variant"] = chosen_variant
        res["ab_test_variant_name"] = ab_manager.variants[chosen_variant]["name"]
        # Record impression
        ab_manager.record_impression(chosen_variant, res.get("total_latency_ms", 0.0))

    return res

@app.post("/ab-test/event")
def log_ab_event(req: EventLogRequest):
    ab_manager.record_event(req.variant, req.event_type)
    return {"status": "success", "variant": req.variant, "event": req.event_type}

@app.get("/ab-test/metrics")
def get_ab_metrics():
    return ab_manager.compute_statistics()

@app.post("/ab-test/simulate")
def simulate_traffic(count: int = 50):
    from src.evaluation.ab_testing import simulate_online_ab_traffic
    stats = simulate_online_ab_traffic(n_users=count)
    return stats

@app.get("/metrics/offline")
def get_offline_metrics():
    eval_file = os.path.join(PROJECT_ROOT, "models", "evaluation", "offline_benchmark.json")
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            return json.load(f)
    return {"error": "Offline benchmark has not been generated yet. Run offline_eval.py"}

@app.get("/system/info")
def get_system_info():
    global pipeline
    if pipeline is None:
        pipeline = RecommendationPipeline()

    cache_stats = pipeline.cache.get_stats()
    return {
        "status": "online",
        "total_users": len(pipeline.user_meta),
        "total_items": len(pipeline.movie_meta),
        "faiss_total_indexed": int(pipeline.retriever.index.ntotal),
        "genres": pipeline.genres,
        "cache": cache_stats,
        "triton_live": pipeline.client.triton_live,
        "inference_engine": "Triton HTTP Server" if pipeline.client.triton_live else "Local ONNXRuntime Engine (Triton-identical Graph)"
    }

@app.post("/cache/clear")
def clear_cache():
    global pipeline
    if pipeline:
        pipeline.cache.clear_all()
    return {"status": "cache_cleared"}

