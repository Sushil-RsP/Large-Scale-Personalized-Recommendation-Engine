"""
Online A/B Testing Simulator & Statistical Inference Engine.
Compares two ranking architectures:
- Variant A: Wide & Deep
- Variant B: DCN-v2 (Deep & Cross Network)
Tracks:
- Impressions, Clicks, Conversions, Latency distributions
- Click-Through Rate (CTR) and Conversion Rate (CVR)
- Two-Sample Proportion Z-Test (z-statistic, p-value, 95% Confidence Interval)
"""

import os
import sys
import math
import json
import time
import hashlib
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class ABTestingManager:
    def __init__(self, experiment_name="ranking_model_dcn_vs_widedeep"):
        self.experiment_name = experiment_name
        self.variants = {
            "variant_A": {"model": "wide_deep", "name": "Wide & Deep (Baseline)"},
            "variant_B": {"model": "dcn", "name": "DCN-v2 (Deep & Cross)"}
        }
        self.telemetry = {
            "variant_A": {"impressions": 0, "clicks": 0, "conversions": 0, "latencies_ms": []},
            "variant_B": {"impressions": 0, "clicks": 0, "conversions": 0, "latencies_ms": []}
        }

    def assign_variant(self, user_id):
        """
        Deterministic consistent hashing for user routing.
        Ensures a user consistently sees the same variant across sessions.
        """
        hash_input = f"{self.experiment_name}:{user_id}".encode("utf-8")
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        bucket = hash_val % 100
        # 50/50 traffic split
        return "variant_A" if bucket < 50 else "variant_B"

    def record_impression(self, variant, latency_ms):
        if variant in self.telemetry:
            self.telemetry[variant]["impressions"] += 1
            self.telemetry[variant]["latencies_ms"].append(float(latency_ms))

    def record_event(self, variant, event_type="click"):
        if variant in self.telemetry:
            if event_type == "click":
                self.telemetry[variant]["clicks"] += 1
            elif event_type == "conversion":
                self.telemetry[variant]["conversions"] += 1

    def compute_statistics(self):
        """
        Computes CTR, CVR, and Two-Sample Proportion Z-test for statistical significance.
        """
        data_a = self.telemetry["variant_A"]
        data_b = self.telemetry["variant_B"]

        imp_a = max(1, data_a["impressions"])
        imp_b = max(1, data_b["impressions"])
        clk_a = data_a["clicks"]
        clk_b = data_b["clicks"]

        ctr_a = clk_a / imp_a
        ctr_b = clk_b / imp_b

        # Relative uplift
        relative_uplift_ctr = ((ctr_b - ctr_a) / ctr_a * 100.0) if ctr_a > 0 else 0.0

        # Two-sample Z-test for proportions
        total_clicks = clk_a + clk_b
        total_impressions = imp_a + imp_b
        p_pooled = total_clicks / total_impressions if total_impressions > 0 else 0.0

        se_pooled = math.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / imp_a + 1.0 / imp_b)) if p_pooled * (1.0 - p_pooled) > 0 else 1e-6
        z_stat = (ctr_b - ctr_a) / se_pooled

        # Two-tailed p-value
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

        # 95% Confidence Interval for difference
        se_diff = math.sqrt((ctr_a * (1.0 - ctr_a) / imp_a) + (ctr_b * (1.0 - ctr_b) / imp_b)) if (ctr_a > 0 or ctr_b > 0) else 1e-6
        margin_error = 1.96 * se_diff
        diff = ctr_b - ctr_a
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error

        is_significant = p_value < 0.05

        return {
            "experiment": self.experiment_name,
            "variant_A": {
                "name": self.variants["variant_A"]["name"],
                "impressions": imp_a,
                "clicks": clk_a,
                "ctr": round(ctr_a, 4),
                "median_latency_ms": round(float(np.median(data_a["latencies_ms"])), 2) if data_a["latencies_ms"] else 0.0
            },
            "variant_B": {
                "name": self.variants["variant_B"]["name"],
                "impressions": imp_b,
                "clicks": clk_b,
                "ctr": round(ctr_b, 4),
                "median_latency_ms": round(float(np.median(data_b["latencies_ms"])), 2) if data_b["latencies_ms"] else 0.0
            },
            "analysis": {
                "relative_ctr_uplift_pct": round(relative_uplift_ctr, 2),
                "z_statistic": round(float(z_stat), 4),
                "p_value": round(float(p_value), 5),
                "statistically_significant": bool(is_significant),
                "confidence_level": "95%",
                "ci_lower_diff": round(ci_lower, 5),
                "ci_upper_diff": round(ci_upper, 5),
                "conclusion": (
                    f"Variant B (DCN-v2) demonstrates a statistically significant CTR lift of {relative_uplift_ctr:.2f}% (p={p_value:.4f} < 0.05)"
                    if is_significant and relative_uplift_ctr > 0
                    else "No statistically significant difference detected yet between variants (p >= 0.05)."
                )
            }
        }

def simulate_online_ab_traffic(n_users=600):
    """
    Simulates live production traffic with realistic user engagement behavior.
    """
    from src.serving.pipeline import RecommendationPipeline
    pipeline = RecommendationPipeline()
    ab_manager = ABTestingManager()

    print(f"Simulating online A/B testing experiment across {n_users} user requests...")
    user_ids = list(pipeline.user_meta.keys())

    np.random.seed(42)
    for i in range(n_users):
        uid = int(np.random.choice(user_ids))
        variant = ab_manager.assign_variant(uid)
        model_name = ab_manager.variants[variant]["model"]

        t0 = time.perf_counter()
        resp = pipeline.recommend(uid, top_k=10, candidate_k=100, ranking_model=model_name, bypass_cache=True)
        lat = (time.perf_counter() - t0) * 1000.0

        ab_manager.record_impression(variant, lat)

        # Simulate user click behavior based on model predicted CTR
        recs = resp["recommendations"]
        for rank, item in enumerate(recs[:5], start=1):
            pred_score = item.get("predicted_ctr", 0.5)
            # Position bias: higher ranked items get higher base view prob
            view_prob = 1.0 / math.log2(rank + 1)
            # Probability of clicking
            click_prob = pred_score * view_prob
            if np.random.rand() < click_prob:
                ab_manager.record_event(variant, event_type="click")
                # Conversion probability
                if np.random.rand() < 0.25:
                    ab_manager.record_event(variant, event_type="conversion")
                break

    stats_report = ab_manager.compute_statistics()
    print("\n--- Online A/B Test Results Summary ---")
    print(json.dumps(stats_report, indent=2))
    return stats_report

