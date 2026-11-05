"""Unit tests for evaluation metrics: NDCG@K, Recall@K, HitRate, MRR."""
import pytest
from src.evaluation.metrics import (
    compute_dcg,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_hit_rate_at_k,
    compute_mrr
)

def test_perfect_ndcg():
    recs = [1, 2, 3, 4, 5]
    gt = {1, 2, 3}
    # When top 3 items match the 3 ground truth items perfectly in the first 3 positions
    ndcg = compute_ndcg_at_k(recs, gt, k=5)
    assert pytest.approx(ndcg, 1e-4) == 1.0

def test_zero_ndcg():
    recs = [10, 20, 30]
    gt = {1, 2, 3}
    ndcg = compute_ndcg_at_k(recs, gt, k=5)
    assert ndcg == 0.0

def test_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    gt = {1, 2, 9, 10} # 2 out of 4 retrieved
    recall = compute_recall_at_k(recs, gt, k=5)
    assert recall == 0.5

def test_mrr():
    # Hit at rank 1
    assert compute_mrr([10, 20, 30], {10}) == 1.0
    # Hit at rank 2
    assert compute_mrr([10, 20, 30], {20}) == 0.5
    # Hit at rank 4
    assert compute_mrr([10, 20, 30, 40], {40}) == 0.25
    # No hit
    assert compute_mrr([10, 20, 30], {99}) == 0.0

def test_ab_testing_clicks_exceed_impressions_no_math_domain_error():
    """Verify that multiple clicks on a single impression do not cause math domain error."""
    from src.evaluation.ab_testing import ABTestingManager
    mgr = ABTestingManager()
    # 1 impression, 5 clicks
    mgr.record_impression("variant_A", 10.0)
    for _ in range(5):
        mgr.record_event("variant_A", "click")

    # 1 impression, 3 clicks
    mgr.record_impression("variant_B", 12.0)
    for _ in range(3):
        mgr.record_event("variant_B", "click")

    stats = mgr.compute_statistics()
    assert stats["variant_A"]["clicks"] == 5
    assert stats["variant_A"]["ctr"] == 1.0
    assert "p_value" in stats["analysis"]


