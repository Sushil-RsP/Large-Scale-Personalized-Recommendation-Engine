# Recommendation Engine API Documentation

The Recommendation Engine provides a RESTful HTTP API documented with OpenAPI v3.0 schemas.

Base URL: `http://localhost:8080`

---

## 1. Recommendation Endpoint

### `POST /recommend`
Executes the two-stage recommendation pipeline, including cache lookup, cold-start handling, candidate retrieval, and deep re-ranking.

#### Request Body Schema
```json
{
  "user_id": 1,
  "top_k": 10,
  "candidate_k": 100,
  "ranking_model": "ab_test",
  "onboarding_genres": ["Action", "Sci-Fi"],
  "bypass_cache": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `user_id` | `integer \| null` | `null` | Target user ID. If `null`, triggers cold-start fallback. |
| `top_k` | `integer` | `10` | Number of final recommendations to return ($1 \le k \le 50$). |
| `candidate_k` | `integer` | `100` | Number of Stage 1 candidates to retrieve via FAISS MIPS ($10 \le k \le 500$). |
| `ranking_model` | `string` | `"ab_test"` | Ranking model to use: `"dcn"`, `"wide_deep"`, or `"ab_test"`. |
| `onboarding_genres` | `array[string] \| null` | `null` | Genres selected during user onboarding for cold-start targeting. |
| `bypass_cache` | `boolean` | `false` | If `true`, skips Redis cache lookup and forces fresh inference. |

#### Example Response (Warm User)
```json
{
  "user_id": 1,
  "is_cold_start": false,
  "cache_hit": false,
  "ranking_model_used": "DCN-v2",
  "candidates_retrieved_stage1": 100,
  "candidates_scored_stage2": 100,
  "ab_test_variant": "variant_B",
  "ab_test_variant_name": "DCN-v2 (Deep & Cross)",
  "recommendations": [
    {
      "movieId": 7720,
      "title": "Four Musketeers, The (1974)",
      "genres": "Action|Adventure|Comedy|Romance",
      "item_popularity": 4,
      "item_avg_rating": 2.75,
      "retrieval_score": 0.366,
      "retrieval_rank": 37,
      "predicted_ctr": 0.864,
      "final_rank": 1,
      "rank_delta": 36
    }
  ],
  "stage_latencies_ms": {
    "user_tower_inference_ms": 2.14,
    "faiss_retrieval_ms": 1.05,
    "stage1_retrieval_total_ms": 3.42,
    "deep_ranking_inference_ms": 3.88,
    "stage2_ranking_total_ms": 4.10,
    "total_pipeline_ms": 8.24
  },
  "total_latency_ms": 8.24
}
```

---

## 2. Online A/B Testing Endpoints

### `POST /ab-test/event`
Records user impression actions, clicks, or conversions for online experiment tracking.

#### Request Body
```json
{
  "variant": "variant_B",
  "event_type": "click"
}
```

### `GET /ab-test/metrics`
Returns current statistical telemetry, conversion rates, relative uplift, and hypothesis significance test results.

#### Example Response
```json
{
  "experiment": "ranking_model_dcn_vs_widedeep",
  "variant_A": {
    "name": "Wide & Deep (Baseline)",
    "impressions": 258,
    "clicks": 245,
    "ctr": 0.9496,
    "median_latency_ms": 3.87
  },
  "variant_B": {
    "name": "DCN-v2 (Deep & Cross)",
    "impressions": 242,
    "clicks": 229,
    "ctr": 0.9463,
    "median_latency_ms": 4.53
  },
  "analysis": {
    "relative_ctr_uplift_pct": -0.35,
    "z_statistic": -0.1677,
    "p_value": 0.8668,
    "statistically_significant": false,
    "confidence_level": "95%",
    "ci_lower_diff": -0.0423,
    "ci_upper_diff": 0.0356,
    "conclusion": "No statistically significant difference detected yet between variants (p >= 0.05)."
  }
}
```

### `POST /ab-test/simulate?count=50`
Simulates a burst of realistic online user interactions for testing statistical convergence.

---

## 3. Telemetry & Administration Endpoints

### `GET /metrics/offline`
Returns offline test-set evaluation results comparing Stage 1 retrieval against Stage 2 rankers across Recall@50, NDCG@10, Hit Rate@10, and MRR.

### `GET /system/info`
Returns health status, total indexed items, active cache backend, and Triton server connectivity.

### `POST /cache/clear`
Flushes all cached recommendation entries across Redis and in-memory fallback stores.
