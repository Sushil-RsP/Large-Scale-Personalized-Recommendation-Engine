import os
import sys
import json
from typing import List, Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.serving.pipeline import RecommendationPipeline
from src.evaluation.ab_testing import ABTestingManager

pipeline: Optional[RecommendationPipeline] = None
ab_manager = ABTestingManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = RecommendationPipeline()
    except Exception as e:
        print(f"Warning during pipeline initialization: {e}")
    yield

app = FastAPI(
    title="Recommendation Engine",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="User ID. Null triggers cold-start fallback.")
    top_k: int = Field(10, ge=1, le=50)
    candidate_k: int = Field(100, ge=10, le=500)
    ranking_model: str = Field("ab_test")
    onboarding_genres: Optional[List[str]] = Field(None)
    bypass_cache: bool = Field(False)

class EventLogRequest(BaseModel):
    variant: str
    event_type: str = "click"

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
    return simulate_online_ab_traffic(n_users=count)

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

    return {
        "status": "online",
        "total_users": len(pipeline.user_meta),
        "total_items": len(pipeline.movie_meta),
        "faiss_total_indexed": int(pipeline.retriever.index.ntotal),
        "genres": pipeline.genres,
        "cache": pipeline.cache.get_stats(),
        "triton_live": pipeline.client.triton_live,
        "inference_engine": "Triton HTTP Server" if pipeline.client.triton_live else "Local ONNXRuntime Engine (Triton-identical Graph)"
    }

@app.post("/cache/clear")
def clear_cache():
    global pipeline
    if pipeline:
        pipeline.cache.clear_all()
    return {"status": "cache_cleared"}
