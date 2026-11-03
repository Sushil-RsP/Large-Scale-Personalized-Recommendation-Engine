"""
Recommender System Evaluation Metrics.
Implements:
- NDCG@K (Normalized Discounted Cumulative Gain)
- Recall@K
- Precision@K
- Hit Rate@K
- MRR (Mean Reciprocal Rank)
- Click-Through Rate (CTR)
"""

import numpy as np

def compute_dcg(relevances, k=10):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if not relevances.size:
        return 0.0
    # Formula: sum_{i=1}^k (2^rel - 1) / log2(i + 1)
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return float(np.sum((np.power(2, relevances) - 1.0) / discounts))

def compute_ndcg_at_k(recommended_ids, ground_truth_set, k=10):
    """
    recommended_ids: list of recommended item IDs in rank order
    ground_truth_set: set of relevant ground truth item IDs
    """
    top_k_recs = recommended_ids[:k]
    # Binary relevance: 1 if item in ground truth set else 0
    relevances = [1.0 if item_id in ground_truth_set else 0.0 for item_id in top_k_recs]
    actual_dcg = compute_dcg(relevances, k=k)

    # Ideal DCG: all relevant items at the top
    num_relevant = min(len(ground_truth_set), k)
    ideal_relevances = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    ideal_dcg = compute_dcg(ideal_relevances, k=k)

    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)

def compute_recall_at_k(recommended_ids, ground_truth_set, k=50):
    if not ground_truth_set:
        return 0.0
    top_k = set(recommended_ids[:k])
    hits = len(top_k.intersection(ground_truth_set))
    return float(hits / len(ground_truth_set))

def compute_precision_at_k(recommended_ids, ground_truth_set, k=10):
    if k == 0:
        return 0.0
    top_k = set(recommended_ids[:k])
    hits = len(top_k.intersection(ground_truth_set))
    return float(hits / k)

def compute_hit_rate_at_k(recommended_ids, ground_truth_set, k=10):
    top_k = set(recommended_ids[:k])
    return 1.0 if len(top_k.intersection(ground_truth_set)) > 0 else 0.0

def compute_mrr(recommended_ids, ground_truth_set, max_k=100):
    for rank, item_id in enumerate(recommended_ids[:max_k], start=1):
        if item_id in ground_truth_set:
            return float(1.0 / rank)
    return 0.0

