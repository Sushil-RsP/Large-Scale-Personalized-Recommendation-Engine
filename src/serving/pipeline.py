"""
End-to-End Two-Stage Recommendation Engine Orchestrator.
Architecture:
1. Cache-Aside Verification (Redis)
2. Cold-Start Detection & Fallbacks
3. Stage 1: Candidate Retrieval (Triton/ONNX User Tower -> FAISS MIPS ANN -> Top 100 Candidates)
4. Stage 2: Deep Re-Ranking (Triton/ONNX DCN-v2 or Wide & Deep -> Calibrated CTR Scoring)
5. Response Generation & Cache Storage with Stage Latency Profiles.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.faiss_index import FAISSCandidateRetriever
from src.cold_start.fallback import ColdStartEngine
from src.cache.redis_cache import RecommendationCache
from src.serving.triton_client import TritonInferenceClient

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
logger = logging.getLogger(__name__)

class RecommendationPipeline:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

        with open(os.path.join(data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        with open(os.path.join(data_dir, "vocab_user2idx.json")) as f:
            self.user2idx = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(data_dir, "vocab_movie2idx.json")) as f:
            self.movie2idx = {int(k): v for k, v in json.load(f).items()}

        self.genres = self.metadata["genres"]
        self.genre_cols = [f"genre_{g.lower()}" for g in self.genres]

        # Load feature tables
        self.movies_df = pd.read_parquet(os.path.join(data_dir, "movies_features.parquet"))
        self.user_df = pd.read_parquet(os.path.join(data_dir, "user_features.parquet"))

        # Fast dictionaries for metadata lookup
        self.movie_meta = {}
        for _, row in self.movies_df.iterrows():
            mid = int(row["movieId"])
            g_vec = row[self.genre_cols].values.astype(np.float32)
            norm_pop = float(np.log1p(row.get("item_popularity", 0.0)))
            norm_rating = float((row.get("item_avg_rating", 3.0) - 1.0) / 4.0)
            norm_year = float((row.get("year", 2000) - 1920) / 100.0)
            self.movie_meta[mid] = {
                "title": row["title"],
                "genres": row["genres"],
                "item_popularity": int(row.get("item_popularity", 0)),
                "item_avg_rating": float(row.get("item_avg_rating", 3.0)),
                "year": int(row.get("year", 2000)),
                "genre_vec": g_vec,
                "stats_vec": np.array([norm_pop, norm_rating, norm_year], dtype=np.float32)
            }

        self.user_meta = {}
        for _, row in self.user_df.iterrows():
            uid = int(row["userId"])
            g_prefs = row[self.genre_cols].values.astype(np.float32)
            norm_count = float(np.log1p(row.get("user_interaction_count", 0.0)))
            norm_rating = float((row.get("user_avg_rating", 3.0) - 1.0) / 4.0)
            self.user_meta[uid] = {
                "genre_prefs": g_prefs,
                "stats_vec": np.array([norm_count, norm_rating], dtype=np.float32)
            }

        # Subsystems
        self.retriever = FAISSCandidateRetriever()
        self.cold_start = ColdStartEngine(data_dir=data_dir)
        self.cache = RecommendationCache()
        self.client = TritonInferenceClient()

    def recommend(self, user_id, top_k=10, candidate_k=100, ranking_model="dcn", onboarding_genres=None, bypass_cache=False):
        """
        Executes end-to-end two-stage recommendation workflow.
        Returns dict with recommendations, stage metrics, and execution latency.
        """
        start_time = time.perf_counter()
        latencies = {}

        model_name = "ranking_dcn" if ranking_model.lower() in ("dcn", "ranking_dcn") else "ranking_wide_deep"

        # 1. Cache-Aside Check
        if not bypass_cache and user_id is not None:
            cached_res = self.cache.get_user_recommendations(user_id, top_k=top_k, model_name=model_name)
            if cached_res is not None:
                cached_res["cache_hit"] = True
                cached_res["total_latency_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
                return cached_res

        # 2. Check Cold-Start
        is_cold = (user_id is None) or (int(user_id) not in self.user_meta)
        if is_cold:
            t_cold = time.perf_counter()
            if onboarding_genres and len(onboarding_genres) > 0:
                fallback_items = self.cold_start.get_content_based_genre_fallback(onboarding_genres, top_k=top_k)
                cold_type = "content_based_genre_onboarding"
            else:
                fallback_items = self.cold_start.get_popular_diverse_fallback(top_k=top_k)
                cold_type = "popularity_diversity_fallback"

            latencies["cold_start_ms"] = round((time.perf_counter() - t_cold) * 1000.0, 3)
            latencies["total_pipeline_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)

            response = {
                "user_id": user_id,
                "is_cold_start": True,
                "cold_start_strategy": cold_type,
                "cache_hit": False,
                "ranking_model_used": "cold_start_fallback",
                "recommendations": fallback_items[:top_k],
                "stage_latencies_ms": latencies,
                "total_latency_ms": latencies["total_pipeline_ms"]
            }
            if not bypass_cache and user_id is not None:
                self.cache.set_user_recommendations(user_id, response, top_k=top_k, model_name=model_name)
            return response

        # 3. Stage 1: Candidate Retrieval (Two-Tower -> FAISS)
        t_retrieval_start = time.perf_counter()
        uid = int(user_id)
        u_idx = self.user2idx.get(uid, 0)
        u_info = self.user_meta[uid]

        # Get user embedding
        t_user_tower = time.perf_counter()
        user_emb = self.client.infer_user_tower(u_idx, u_info["genre_prefs"], u_info["stats_vec"])
        latencies["user_tower_inference_ms"] = round((time.perf_counter() - t_user_tower) * 1000.0, 3)

        # Query FAISS index for candidates
        candidates, faiss_latency = self.retriever.retrieve(user_emb, top_k=candidate_k)
        latencies["faiss_retrieval_ms"] = round(faiss_latency, 3)
        latencies["stage1_retrieval_total_ms"] = round((time.perf_counter() - t_retrieval_start) * 1000.0, 3)

        # 4. Stage 2: Deep Re-Ranking (DCN-v2 or Wide & Deep)
        t_ranking_start = time.perf_counter()
        user_ids_batch = []
        item_ids_batch = []
        dense_feats_batch = []
        candidate_meta_batch = []

        u_stats = u_info["stats_vec"]
        for cand in candidates:
            mid = cand["movieId"]
            if mid not in self.movie_meta:
                continue
            m_idx = self.movie2idx.get(mid, 0)
            m_info = self.movie_meta[mid]

            # Construct dense features: [user_stats (2), item_stats (3), genres (16), retrieval_sim (1)]
            dense_vec = np.concatenate([
                u_stats,
                m_info["stats_vec"],
                m_info["genre_vec"],
                [cand["retrieval_score"]]
            ]).astype(np.float32)

            user_ids_batch.append(u_idx)
            item_ids_batch.append(m_idx)
            dense_feats_batch.append(dense_vec)
            candidate_meta_batch.append({
                "movieId": mid,
                "title": m_info["title"],
                "genres": m_info["genres"],
                "item_popularity": m_info["item_popularity"],
                "item_avg_rating": m_info["item_avg_rating"],
                "retrieval_score": cand["retrieval_score"],
                "retrieval_rank": cand["retrieval_rank"]
            })

        # Batch scoring through Triton/ONNX ranker
        u_arr = np.array(user_ids_batch, dtype=np.int64)
        i_arr = np.array(item_ids_batch, dtype=np.int64)
        d_arr = np.array(dense_feats_batch, dtype=np.float32)

        ctr_predictions = self.client.infer_ranker(model_name, u_arr, i_arr, d_arr)
        latencies["deep_ranking_inference_ms"] = round((time.perf_counter() - t_ranking_start) * 1000.0, 3)

        # Sort candidates by predicted CTR
        for idx, score in enumerate(ctr_predictions):
            candidate_meta_batch[idx]["predicted_ctr"] = float(score)

        ranked_items = sorted(candidate_meta_batch, key=lambda x: x["predicted_ctr"], reverse=True)
        final_recommendations = []
        for rank, item in enumerate(ranked_items[:top_k]):
            item["final_rank"] = rank + 1
            item["rank_delta"] = item["retrieval_rank"] - item["final_rank"]
            final_recommendations.append(item)

        latencies["stage2_ranking_total_ms"] = round((time.perf_counter() - t_ranking_start) * 1000.0, 3)
        latencies["total_pipeline_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)

        response = {
            "user_id": uid,
            "is_cold_start": False,
            "cache_hit": False,
            "ranking_model_used": "DCN-v2" if model_name == "ranking_dcn" else "Wide & Deep",
            "candidates_retrieved_stage1": len(candidates),
            "candidates_scored_stage2": len(candidate_meta_batch),
            "recommendations": final_recommendations,
            "stage_latencies_ms": latencies,
            "total_latency_ms": latencies["total_pipeline_ms"]
        }

        # Cache result
        if not bypass_cache:
            self.cache.set_user_recommendations(uid, response, top_k=top_k, model_name=model_name)

        return response

