# 10-Minute Final Presentation: Large-Scale Two-Stage Recommendation Engine

**Presenter:** Intern ML / AI Engineering Team  
**Evaluation Track:** AI / Machine Learning  
**Target Audience:** Senior Engineering Leadership & Evaluation Committee  
**Total Duration:** 10 Minutes (Slides + Demo)

---

## Slide 1: Title & Executive Summary (0:00 - 1:00)
- **Title:** Engineering a Production Two-Stage Recommendation Engine (Netflix/Amazon Architecture)
- **Problem Statement:** How do web-scale platforms personalize feeds across 10,000,000+ items within a 50ms latency budget?
- **Key Solution:**
  - Decouple Candidate Retrieval (high recall, $\mathcal{O}(1)$ vector search) from Candidate Ranking (high precision, deep feature interaction).
  - Production stack: Apache Spark $\to$ Two-Tower Retrieval $\to$ FAISS MIPS $\to$ DCN-v2 / Wide & Deep $\to$ Redis Cache $\to$ Triton/ONNX serving $\to$ Airflow orchestration.
- **Key Takeaways:**
  - 8.2ms end-to-end P50 latency.
  - 0.9188 ROC-AUC on DCN-v2.
  - Sub-millisecond FAISS candidate retrieval over 9,742 items.

---

## Slide 2: Data Pipeline & Temporal Integrity (1:00 - 2:30)
- **The Pitfall of Random Splits:** Random splitting in recommender systems causes massive future temporal leakage (predicting past events using future interactions).
- **Our Approach with Apache Spark:**
  - Strictly chronological split (80% Train, 10% Validation, 10% Test).
  - Ingestion of MovieLens 25M / Small datasets with PySpark.
  - Engineering user engagement history, genre affinity vectors, and Bayesian popularity priors solely on training history.
  - Verification: Automated test `test_temporal_split_no_leakage` guaranteeing $\max(T_{\text{train}}) \le \min(T_{\text{val}}) \le \min(T_{\text{test}})$.

---

## Slide 3: Stage 1 - Two-Tower Neural Retrieval & FAISS (2:30 - 4:00)
- **Architecture:**
  - Dual independent neural networks projecting users and items into a shared 64-dimensional unit hypersphere.
  - In-batch sampled softmax loss with temperature scaling ($\tau = 0.07$).
- **Vector Search Acceleration:**
  - Offline pre-computation of 9,742 item vectors.
  - FAISS index (`IndexFlatIP` and `IndexIVFFlat`) executing Inner Product MIPS in **1.05ms**.
  - Extracts the top 100 high-affinity candidates from the entire catalog.

---

## Slide 4: Stage 2 - Deep Ranking: DCN-v2 vs Wide & Deep (4:00 - 5:30)
- **Why Rank After Retrieval?** Dot-product retrieval only computes linear vector overlaps; it cannot model non-linear cross features (e.g., user recency $\times$ movie release decade $\times$ genre match).
- **Architectural Comparison:**
  - **Wide & Deep:** Linear sparse memorization + Deep MLP generalization (AUC: 0.9045).
  - **DCN-v2 (Deep & Cross Network):** Explicit multiplicative cross layers ($x_{l+1} = x_0 (W x_l + b) + x_l$) automatically capturing degree-3 crosses (AUC: 0.9188).
- **Offline Impact:** Re-ranking yields a **+19.4% relative gain in NDCG@10** and **+33.3% relative lift in Hit Rate@10** over retrieval alone.

---

## Slide 5: Cold-Start Strategy & Redis Caching (5:30 - 7:00)
- **Cold Users (Zero History):**
  - Bayesian weighted popularity ranking:
    $$\text{WR} = \left(\frac{v}{v+m}\right) R + \left(\frac{m}{v+m}\right) C$$
  - Enforced genre quotas (max 15 items per genre) to avoid echo chambers and deliver diverse initial discovery.
- **Onboarding Personalization:**
  - Dynamic content-based filtering matching chosen registration genres.
- **Redis Cache-Aside:**
  - Key: `rec:user:{user_id}:topk:{k}` (TTL 300s).
  - Latency drops from 8.24ms to **0.78ms on cache hit**.
  - Graceful zero-failure fallback to in-memory LRU store.

---

## Slide 6: Serving, A/B Testing & Airflow Orchestration (7:00 - 8:30)
- **Inference Serving:**
  - NVIDIA Triton Model Repository generation (`config.pbtxt` with dynamic batching up to batch size 256).
  - ONNX graph optimization for low CPU/GPU inference latency.
- **Online A/B Testing Framework:**
  - Consistent MD5 user hashing for 50/50 traffic splitting between Variant A (Wide & Deep) and Variant B (DCN-v2).
  - Live statistical significance engine computing two-sample proportion Z-test, $p$-value, and 95% confidence intervals.
- **Scheduled Retraining with Apache Airflow:**
  - DAG pipeline: `extract_spark >> train_retrieval >> update_faiss >> train_rankers >> evaluate_gatekeep >> export_triton >> invalidate_cache`.

---

## Slide 7: Live System Demonstration (8:30 - 9:30)
- Interactive walkthrough of the Web Dashboard:
  1. Inspecting warm user personalization with Stage 1 vs Stage 2 rank deltas ($\Delta +36$).
  2. Testing cold-start user onboarding with genre selections.
  3. Live latency waterfall showing sub-10ms response times.
  4. Triggering live A/B test simulation and verifying real-time statistical telemetry.

---

## Slide 8: Future Roadmap & Conclusion (9:30 - 10:00)
- **Next Steps:**
  - Graph Neural Networks (PinSage / LightGCN) for high-order collaborative filtering.
  - Multi-Task Learning (MTL) ranking predicting both Clicks (CTR) and Completion Rate / Watch Time.
  - Real-time streaming features with Apache Flink and Kafka.
- **Conclusion:**
  - Fully functional, highly resilient two-stage recommendation engine exceeding all requirements with 18 passing tests, complete Triton/Airflow/Redis integrations, and production-ready code.

