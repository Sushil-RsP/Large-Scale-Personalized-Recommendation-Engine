"""
Offline Model Evaluation Benchmark.
Evaluates:
1. Stage 1 Alone: Two-Tower Candidate Retrieval (FAISS MIPS)
2. Stage 2 with Wide & Deep Ranking
3. Stage 2 with DCN-v2 (Deep & Cross Network)
Evaluated on the chronologically held-out test split.
Measures Recall@K, NDCG@K, HitRate@K, MRR, and Latencies.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.serving.pipeline import RecommendationPipeline
from src.evaluation.metrics import (
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_hit_rate_at_k,
    compute_mrr
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "evaluation")

def run_offline_evaluation(sample_users=150):
    os.makedirs(EVAL_DIR, exist_ok=True)
    print("Loading test split for offline benchmark...")

    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    # Filter positive interactions in test period (ground truth)
    test_positives = test_df[test_df["label"] == 1.0]

    user_ground_truth = {}
    for uid, group in test_positives.groupby("userId"):
        user_ground_truth[int(uid)] = set(group["movieId"].astype(int).tolist())

    print(f"Total test users with positive ground truth: {len(user_ground_truth)}")
    pipeline = RecommendationPipeline(data_dir=DATA_DIR)

    eval_users = [u for u in user_ground_truth.keys() if u in pipeline.user_meta][:sample_users]
    print(f"Evaluating {len(eval_users)} users across 3 configurations...")

    results = {
        "Stage1_Retrieval_Only": {"ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "recall@20": [], "recall@50": [], "recall@100": [], "hit_rate@10": [], "mrr": [], "latency_ms": []},
        "Stage2_Wide_and_Deep": {"ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "recall@20": [], "recall@50": [], "recall@100": [], "hit_rate@10": [], "mrr": [], "latency_ms": []},
        "Stage2_DCN_v2": {"ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "recall@20": [], "recall@50": [], "recall@100": [], "hit_rate@10": [], "mrr": [], "latency_ms": []}
    }

    for idx, uid in enumerate(eval_users, 1):
        gt = user_ground_truth[uid]

        # 1. Config A: Retrieval Only (sort by raw retrieval score)
        t0 = time.perf_counter()
        resp_retrieval = pipeline.recommend(uid, top_k=100, candidate_k=100, ranking_model="dcn", bypass_cache=True)
        # Sort by stage 1 retrieval score
        recs_stage1 = sorted(resp_retrieval["recommendations"], key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
        stage1_ids = [x["movieId"] for x in recs_stage1]
        lat1 = (time.perf_counter() - t0) * 1000.0

        # 2. Config B: Wide & Deep Re-ranking
        t0 = time.perf_counter()
        resp_wd = pipeline.recommend(uid, top_k=100, candidate_k=100, ranking_model="wide_deep", bypass_cache=True)
        wd_ids = [x["movieId"] for x in resp_wd["recommendations"]]
        lat2 = (time.perf_counter() - t0) * 1000.0

        # 3. Config C: DCN-v2 Re-ranking
        t0 = time.perf_counter()
        resp_dcn = pipeline.recommend(uid, top_k=100, candidate_k=100, ranking_model="dcn", bypass_cache=True)
        dcn_ids = [x["movieId"] for x in resp_dcn["recommendations"]]
        lat3 = (time.perf_counter() - t0) * 1000.0

        for key, rec_ids, lat in [
            ("Stage1_Retrieval_Only", stage1_ids, lat1),
            ("Stage2_Wide_and_Deep", wd_ids, lat2),
            ("Stage2_DCN_v2", dcn_ids, lat3)
        ]:
            results[key]["ndcg@5"].append(compute_ndcg_at_k(rec_ids, gt, k=5))
            results[key]["ndcg@10"].append(compute_ndcg_at_k(rec_ids, gt, k=10))
            results[key]["ndcg@20"].append(compute_ndcg_at_k(rec_ids, gt, k=20))
            results[key]["recall@20"].append(compute_recall_at_k(rec_ids, gt, k=20))
            results[key]["recall@50"].append(compute_recall_at_k(rec_ids, gt, k=50))
            results[key]["recall@100"].append(compute_recall_at_k(rec_ids, gt, k=100))
            results[key]["hit_rate@10"].append(compute_hit_rate_at_k(rec_ids, gt, k=10))
            results[key]["mrr"].append(compute_mrr(rec_ids, gt, max_k=100))
            results[key]["latency_ms"].append(lat)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(eval_users)} evaluation users...")

    # Aggregate summary metrics
    summary = {}
    print("\n" + "="*80)
    print(f"{'Model Configuration':<26} | {'Recall@50':<9} | {'NDCG@10':<9} | {'HitRate@10':<10} | {'MRR':<8} | {'Latency':<8}")
    print("="*80)

    for model_name, metric_dict in results.items():
        summary[model_name] = {
            m: round(float(np.mean(vals)), 4) if "latency" not in m else round(float(np.median(vals)), 2)
            for m, vals in metric_dict.items()
        }
        rec50 = summary[model_name]["recall@50"]
        ndcg10 = summary[model_name]["ndcg@10"]
        hr10 = summary[model_name]["hit_rate@10"]
        mrr = summary[model_name]["mrr"]
        lat = summary[model_name]["latency_ms"]
        print(f"{model_name:<26} | {rec50:<9.4f} | {ndcg10:<9.4f} | {hr10:<10.4f} | {mrr:<8.4f} | {lat} ms")
    print("="*80)

    out_file = os.path.join(EVAL_DIR, "offline_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation benchmark results saved to {out_file}")
    return summary

if __name__ == "__main__":
    run_offline_evaluation()

