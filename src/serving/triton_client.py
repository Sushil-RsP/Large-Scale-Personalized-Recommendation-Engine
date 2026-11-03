"""
Triton Inference Client with seamless Local ONNXRuntime fallback.
Connects to Triton Inference Server over HTTP/gRPC when deployed in production,
and transparently utilizes local ONNXRuntime sessions for testing and local operation.
"""

import os
import sys
import json
import logging
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import onnxruntime as ort

TRITON_REPO_DIR = os.path.join(PROJECT_ROOT, "triton_model_repository")
logger = logging.getLogger(__name__)

class TritonInferenceClient:
    def __init__(self, triton_url=None):
        self.triton_url = triton_url or os.environ.get("TRITON_URL", "http://localhost:8000")
        self.triton_live = False
        self.ort_sessions = {}

        # Attempt to probe Triton
        try:
            import requests
            resp = requests.get(f"{self.triton_url}/v2/health/ready", timeout=0.5)
            if resp.status_code == 200:
                self.triton_live = True
                logger.info(f"Connected to Triton Inference Server at {self.triton_url}")
        except Exception:
            self.triton_live = False

        if not self.triton_live:
            # Initialize high-performance local ONNXRuntime engines
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            user_tower_onnx = os.path.join(TRITON_REPO_DIR, "user_tower", "1", "model.onnx")
            dcn_onnx = os.path.join(TRITON_REPO_DIR, "ranking_dcn", "1", "model.onnx")
            wide_deep_onnx = os.path.join(TRITON_REPO_DIR, "ranking_wide_deep", "1", "model.onnx")

            if os.path.exists(user_tower_onnx):
                self.ort_sessions["user_tower"] = ort.InferenceSession(user_tower_onnx, opts, providers=["CPUExecutionProvider"])
            if os.path.exists(dcn_onnx):
                self.ort_sessions["ranking_dcn"] = ort.InferenceSession(dcn_onnx, opts, providers=["CPUExecutionProvider"])
            if os.path.exists(wide_deep_onnx):
                self.ort_sessions["ranking_wide_deep"] = ort.InferenceSession(wide_deep_onnx, opts, providers=["CPUExecutionProvider"])

    def infer_user_tower(self, user_id_int, user_genres_np, user_stats_np):
        """
        Runs User Tower inference to produce a 64-dimensional user embedding.
        """
        user_id_arr = np.array([user_id_int], dtype=np.int64)
        user_genres_arr = user_genres_np.reshape(1, -1).astype(np.float32)
        user_stats_arr = user_stats_np.reshape(1, -1).astype(np.float32)

        if self.triton_live:
            import requests
            payload = {
                "inputs": [
                    {"name": "user_id", "shape": [1, 1], "datatype": "INT64", "data": [int(user_id_int)]},
                    {"name": "user_genres", "shape": [1, len(user_genres_np)], "datatype": "FP32", "data": user_genres_np.tolist()},
                    {"name": "user_stats", "shape": [1, len(user_stats_np)], "datatype": "FP32", "data": user_stats_np.tolist()}
                ]
            }
            resp = requests.post(f"{self.triton_url}/v2/models/user_tower/infer", json=payload, timeout=2.0)
            data = resp.json()
            return np.array(data["outputs"][0]["data"], dtype=np.float32).reshape(1, -1)

        # Local ONNXRuntime execution
        session = self.ort_sessions.get("user_tower")
        if session is None:
            raise RuntimeError("User Tower ONNX session not loaded.")

        inputs = {
            "user_id": user_id_arr,
            "user_genres": user_genres_arr,
            "user_stats": user_stats_arr
        }
        outputs = session.run(["user_embedding"], inputs)
        return outputs[0]

    def infer_ranker(self, model_name, user_ids_arr, item_ids_arr, dense_features_arr):
        """
        Runs batch candidate re-scoring using DCN-v2 or Wide & Deep.
        model_name: 'ranking_dcn' or 'ranking_wide_deep'
        """
        if self.triton_live:
            import requests
            batch_size = len(user_ids_arr)
            payload = {
                "inputs": [
                    {"name": "user_id", "shape": [batch_size, 1], "datatype": "INT64", "data": user_ids_arr.tolist()},
                    {"name": "item_id", "shape": [batch_size, 1], "datatype": "INT64", "data": item_ids_arr.tolist()},
                    {"name": "dense_features", "shape": [batch_size, dense_features_arr.shape[1]], "datatype": "FP32", "data": dense_features_arr.tolist()}
                ]
            }
            resp = requests.post(f"{self.triton_url}/v2/models/{model_name}/infer", json=payload, timeout=3.0)
            data = resp.json()
            return np.array(data["outputs"][0]["data"], dtype=np.float32)

        session = self.ort_sessions.get(model_name)
        if session is None:
            raise RuntimeError(f"Ranking ONNX session '{model_name}' not loaded.")

        inputs = {
            "user_id": user_ids_arr.astype(np.int64),
            "item_id": item_ids_arr.astype(np.int64),
            "dense_features": dense_features_arr.astype(np.float32)
        }
        outputs = session.run(["ctr_score"], inputs)
        return outputs[0]

