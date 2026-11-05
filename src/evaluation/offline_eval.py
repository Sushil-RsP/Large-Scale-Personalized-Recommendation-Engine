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

    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    test_positives = test_df[test_df["label"] == 1.0]

    user_ground_truth = {}
    for uid, group in test_positives.groupby("userId"):
        user_ground_truth[int(uid)] = set(group["movieId"].astype(int).tolist())

    pipeline = RecommendationPipeline(data_dir=DATA_DIR)
    eval_users = [u for u in user_ground_truth.keys() if u in pipeline.user_meta][:sample_users]

    results = {
        "Stage1_Retrieval_Only": {"ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "recall@20": [], "recall@50": [], "recall@100": [], "hit_rate@10": [], "mrr": [], "latency_ms": []},
        "Stage2_Wide_and_Deep": {"ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "recall@20": [], "recall@50": [], "recall@100": [], "hit_rate@10": [], "mrr": [], "latency_ms": []},
        "Stage2_DCN_v2": {"ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "recall@20": [], "recall@50": [], "recall@100": [], "hit_rate@10": [], "mrr": [], "latency_ms": []}
    }

    for idx, uid in enumerate(eval_users, 1):
        gt = user_ground_truth[uid]

        t0 = time.perf_counter()
        resp_retrieval = pipeline.recommend(uid, top_k=100, candidate_k=100, ranking_model="dcn", bypass_cache=True)
        recs_stage1 = sorted(resp_retrieval["recommendations"], key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
        stage1_ids = [x["movieId"] for x in recs_stage1]
        lat1 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        resp_wd = pipeline.recommend(uid, top_k=100, candidate_k=100, ranking_model="wide_deep", bypass_cache=True)
        wd_ids = [x["movieId"] for x in resp_wd["recommendations"]]
        lat2 = (time.perf_counter() - t0) * 1000.0

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

    summary = {}
    for model_name, metric_dict in results.items():
        summary[model_name] = {
            m: round(float(np.mean(vals)), 4) if "latency" not in m else round(float(np.median(vals)), 2)
            for m, vals in metric_dict.items()
        }

    out_file = os.path.join(EVAL_DIR, "offline_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    run_offline_evaluation()
