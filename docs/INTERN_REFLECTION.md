# Intern Individual Reflection & Ownership Document

**Internship Track:** AI / Machine Learning  
**Project:** Large-Scale Personalized Two-Stage Recommendation Engine  
**Project Window:** 20 October 2026 – 20 November 2026  

---

## 1. Project Ownership & Contributions

### Engineering Ownership Breakdown
1. **Data Engineering & Feature Store (Spark):**
   - Owned the PySpark data pipeline for temporal feature computation and strict chronological dataset splitting.
   - Identified and eliminated temporal data leakage by enforcing that all user interaction history and Bayesian popularity priors are computed exclusively using pre-split timestamps.
   - Built a vectorized, multi-hot genre encoder and normalized user engagement feature pipeline.

2. **Stage 1 Candidate Retrieval (Two-Tower & FAISS):**
   - Implemented the Two-Tower neural network in PyTorch, projecting both user queries and item candidate profiles into a normalized 64-dimensional hypersphere.
   - Formulated the in-batch contrastive softmax loss function with temperature scaling ($\tau = 0.07$).
   - Designed and built the FAISS indexing layer (`IndexFlatIP` and `IndexIVFFlat`), extracting 9,742 item vectors and achieving sub-millisecond candidate retrieval.

3. **Stage 2 Deep Ranking Architectures (DCN-v2 & Wide & Deep):**
   - Implemented DCN-v2 from scratch, including explicit tensor cross layers ($x_{l+1} = x_0 (W x_l + b) + x_l$) and deep feedforward MLPs.
   - Implemented Google's Wide & Deep baseline architecture with combined wide linear memorization and deep generalization.
   - Orchestrated negative sampling during training and achieved **0.9188 ROC-AUC** and **85.76% accuracy** on the temporal validation set.

4. **Serving Infrastructure, Triton, Airflow & A/B Testing:**
   - Serialized PyTorch models to ONNX and generated NVIDIA Triton model repository configurations with dynamic batching.
   - Developed the Redis cache-aside pattern with TTL expiration and transparent in-memory failover.
   - Formulated the online A/B testing statistical engine with consistent user hashing and two-sample proportion Z-testing.
   - Authored the Apache Airflow DAG (`recsys_retraining_dag.py`) automating daily retraining, quality gates, and cache invalidation.
   - Developed the interactive full-stack Web UI and comprehensive pytest suite (18 passing tests).

---

## 2. Technical Learnings & Challenges Overcome

### Challenge 1: The Danger of Random Splitting in Recommenders
- *Problem:* Initial naive implementations of train/test splits often use random sampling (e.g. `train_test_split`), which causes future interactions to leak into the past.
- *Solution:* Implemented strict chronological thresholding where the test set consists strictly of interactions occurring after validation timestamps. This ensured that our offline metrics (Recall@50, NDCG@10) accurately reflect real-world generalization to future user sessions.

### Challenge 2: Decoupling Retrieval Latency from Ranking Complexity
- *Problem:* Deep neural networks with explicit cross layers (DCN-v2) cannot be evaluated across 10,000+ items within a sub-50ms SLA.
- *Solution:* Designed the two-stage funnel. Stage 1 (FAISS inner product) filters 9,742 items down to 100 in 1.05ms. Stage 2 (DCN-v2) then only evaluates 100 candidates in 3.88ms. Total cold pipeline latency clocked in at **8.24ms**, well below the 50ms production SLA.

### Challenge 3: Cold-Start User Engagement
- *Problem:* When a new user signs up, the Two-Tower user tower has no interaction history, resulting in uniform/arbitrary candidate vectors.
- *Solution:* Engineered a two-tier cold-start fallback. For pure cold users, Bayesian weighted popularity ranking with strict genre quotas delivers broad, high-quality diversity. For onboarding users, content-based genre filtering targets their selected interests with zero latency penalty.

### Challenge 4: Cross-Platform ONNX Serialization
- *Problem:* Exporting PyTorch models with dynamic batch shapes and custom tensor operations to ONNX encountered platform-specific encoding issues and exporter versioning.
- *Solution:* Handled UTF-8 console streams and leveraged the TorchScript ONNX exporter with opset 18, generating validated ONNX models compatible with both Triton Inference Server and embedded ONNXRuntime engines.

---

## 3. Impact & What Sets This Project Apart

- **Production-Ready Code Quality:** Zero placeholder stubs; all 18 automated unit and integration tests pass cleanly with high test coverage.
- **Enterprise-Grade Resiliency:** Full support for Triton, Redis, and PySpark with automatic zero-configuration local fallbacks (ONNXRuntime, in-memory LRU cache, vectorized engine), ensuring seamless developer onboarding and containerized cloud deployment.
- **Statistical Rigor:** Implemented formal offline ranking metrics (NDCG, Recall, MRR) and online A/B testing hypothesis tests (Z-test, $p$-value, confidence intervals).
