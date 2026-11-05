import os
import sys
import json
import torch
import onnx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.two_tower import UserTower
from src.ranking.dcn import DeepAndCrossNetwork
from src.ranking.wide_deep import WideAndDeepNetwork

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TRITON_REPO_DIR = os.path.join(PROJECT_ROOT, "triton_model_repository")

def export_all_models_to_triton():
    os.makedirs(TRITON_REPO_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)

    num_users = meta["num_users"] + 1
    num_items = meta["num_items"] + 1
    num_genres = meta["num_genres"]

    export_user_tower(num_users, num_genres)
    export_ranking_dcn(num_users, num_items, num_genres)
    export_ranking_wide_deep(num_users, num_items, num_genres)

def export_user_tower(num_users, num_genres):
    model_dir = os.path.join(TRITON_REPO_DIR, "user_tower", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")

    model = UserTower(num_users, num_genres, embedding_dim=64)
    weights_path = os.path.join(MODEL_DIR, "retrieval", "user_tower.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_uid = torch.tensor([1, 2], dtype=torch.long)
    dummy_genres = torch.zeros(2, num_genres, dtype=torch.float32)
    dummy_stats = torch.zeros(2, 2, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_uid, dummy_genres, dummy_stats),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["user_id", "user_genres", "user_stats"],
        output_names=["user_embedding"],
        dynamic_axes={
            "user_id": {0: "batch_size"},
            "user_genres": {0: "batch_size"},
            "user_stats": {0: "batch_size"},
            "user_embedding": {0: "batch_size"}
        }
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    config_text = f"""name: "user_tower"
platform: "onnxruntime_onnx"
max_batch_size: 128

input [
  {{
    name: "user_id"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "user_genres"
    data_type: TYPE_FP32
    dims: [ {num_genres} ]
  }},
  {{
    name: "user_stats"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }}
]

output [
  {{
    name: "user_embedding"
    data_type: TYPE_FP32
    dims: [ 64 ]
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""
    with open(os.path.join(TRITON_REPO_DIR, "user_tower", "config.pbtxt"), "w") as f:
        f.write(config_text)

def export_ranking_dcn(num_users, num_items, num_genres):
    model_dir = os.path.join(TRITON_REPO_DIR, "ranking_dcn", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")

    model = DeepAndCrossNetwork(num_users, num_items, num_genres, embedding_dim=32)
    weights_path = os.path.join(MODEL_DIR, "ranking", "dcn_v2_best.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    num_dense = 2 + 3 + num_genres + 1
    dummy_u = torch.tensor([1, 2], dtype=torch.long)
    dummy_i = torch.tensor([10, 20], dtype=torch.long)
    dummy_d = torch.zeros(2, num_dense, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_u, dummy_i, dummy_d),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["user_id", "item_id", "dense_features"],
        output_names=["ctr_score"],
        dynamic_axes={
            "user_id": {0: "batch_size"},
            "item_id": {0: "batch_size"},
            "dense_features": {0: "batch_size"},
            "ctr_score": {0: "batch_size"}
        }
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    config_text = f"""name: "ranking_dcn"
platform: "onnxruntime_onnx"
max_batch_size: 256

input [
  {{
    name: "user_id"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "item_id"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "dense_features"
    data_type: TYPE_FP32
    dims: [ {num_dense} ]
  }}
]

output [
  {{
    name: "ctr_score"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""
    with open(os.path.join(TRITON_REPO_DIR, "ranking_dcn", "config.pbtxt"), "w") as f:
        f.write(config_text)

def export_ranking_wide_deep(num_users, num_items, num_genres):
    model_dir = os.path.join(TRITON_REPO_DIR, "ranking_wide_deep", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")

    model = WideAndDeepNetwork(num_users, num_items, num_genres, embedding_dim=32)
    weights_path = os.path.join(MODEL_DIR, "ranking", "wide_and_deep_best.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    num_dense = 2 + 3 + num_genres + 1
    dummy_u = torch.tensor([1, 2], dtype=torch.long)
    dummy_i = torch.tensor([10, 20], dtype=torch.long)
    dummy_d = torch.zeros(2, num_dense, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_u, dummy_i, dummy_d),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["user_id", "item_id", "dense_features"],
        output_names=["ctr_score"],
        dynamic_axes={
            "user_id": {0: "batch_size"},
            "item_id": {0: "batch_size"},
            "dense_features": {0: "batch_size"},
            "ctr_score": {0: "batch_size"}
        }
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    config_text = f"""name: "ranking_wide_deep"
platform: "onnxruntime_onnx"
max_batch_size: 256

input [
  {{
    name: "user_id"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "item_id"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "dense_features"
    data_type: TYPE_FP32
    dims: [ {num_dense} ]
  }}
]

output [
  {{
    name: "ctr_score"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""
    with open(os.path.join(TRITON_REPO_DIR, "ranking_wide_deep", "config.pbtxt"), "w") as f:
        f.write(config_text)

if __name__ == "__main__":
    export_all_models_to_triton()
