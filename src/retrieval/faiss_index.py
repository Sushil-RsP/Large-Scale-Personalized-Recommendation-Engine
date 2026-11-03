"""
FAISS Vector Indexing & Candidate Retrieval.
Extracts candidate item embeddings using the trained ItemTower and constructs
high-performance FAISS indices (FlatIP and IVFFlat) for sub-millisecond MIPS retrieval.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import faiss

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FAISS_DIR = os.path.join(MODEL_DIR, "faiss")

from src.retrieval.two_tower import ItemTower

def build_faiss_index(index_type="flat_ip", nlist=50):
    """
    Computes item embeddings for all catalog items and builds a FAISS index.
    """
    os.makedirs(FAISS_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(DATA_DIR, "vocab_movie2idx.json")) as f:
        movie2idx = json.load(f)
    with open(os.path.join(DATA_DIR, "vocab_idx2movie.json")) as f:
        idx2movie = json.load(f)

    movies_df = pd.read_parquet(os.path.join(DATA_DIR, "movies_features.parquet"))
    genres = meta["genres"]
    genre_cols = [f"genre_{g.lower()}" for g in genres]

    # Load Item Tower
    item_tower = ItemTower(
        num_items=meta["num_items"] + 1,
        num_genres=meta["num_genres"],
        embedding_dim=64
    )
    item_weights_path = os.path.join(MODEL_DIR, "retrieval", "item_tower.pt")
    if os.path.exists(item_weights_path):
        item_tower.load_state_dict(torch.load(item_weights_path, map_location="cpu", weights_only=True))
    item_tower.eval()

    # Prepare item tensors
    all_movie_ids = sorted([int(k) for k in movie2idx.keys()])
    m_indices = [movie2idx[str(m)] if str(m) in movie2idx else movie2idx.get(m, 0) for m in all_movie_ids]

    movie_dict = {row["movieId"]: row for _, row in movies_df.iterrows()}

    genre_vectors = []
    stat_vectors = []
    valid_movie_ids = []
    valid_indices = []

    for m_id, m_idx in zip(all_movie_ids, m_indices):
        if m_id in movie_dict and m_idx > 0:
            row = movie_dict[m_id]
            g_vec = row[genre_cols].values.astype(np.float32)
            norm_pop = float(np.log1p(row.get("item_popularity", 0.0)))
            norm_rating = float((row.get("item_avg_rating", 3.0) - 1.0) / 4.0)
            norm_year = float((row.get("year", 2000) - 1920) / 100.0)

            genre_vectors.append(g_vec)
            stat_vectors.append([norm_pop, norm_rating, norm_year])
            valid_movie_ids.append(m_id)
            valid_indices.append(m_idx)

    item_id_t = torch.tensor(valid_indices, dtype=torch.long)
    genre_t = torch.tensor(np.array(genre_vectors), dtype=torch.float32)
    stats_t = torch.tensor(np.array(stat_vectors), dtype=torch.float32)

    print(f"Extracting item embeddings for {len(valid_movie_ids)} catalog items...")
    with torch.no_grad():
        item_embeddings = item_tower(item_id_t, genre_t, stats_t).cpu().numpy().astype(np.float32)

    dim = item_embeddings.shape[1]

    if index_type == "ivf_flat":
        # Trainable IVF index for large catalogs
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, len(item_embeddings) // 10), faiss.METRIC_INNER_PRODUCT)
        index.train(item_embeddings)
        index.add(item_embeddings)
        index.nprobe = 10
    else:
        # Exact MIPS index
        index = faiss.IndexFlatIP(dim)
        index.add(item_embeddings)

    # Save FAISS index and metadata
    index_file = os.path.join(FAISS_DIR, "item_index.faiss")
    faiss.write_index(index, index_file)
    np.save(os.path.join(FAISS_DIR, "item_ids.npy"), np.array(valid_movie_ids))
    np.save(os.path.join(FAISS_DIR, "item_embeddings.npy"), item_embeddings)

    print(f"FAISS index saved successfully at {index_file} ({index.ntotal} items indexed, dim={dim})")
    return index

class FAISSCandidateRetriever:
    def __init__(self, index_dir=FAISS_DIR):
        index_file = os.path.join(index_dir, "item_index.faiss")
        id_file = os.path.join(index_dir, "item_ids.npy")

        if not os.path.exists(index_file) or not os.path.exists(id_file):
            raise FileNotFoundError(f"FAISS index or item_ids missing in {index_dir}")

        self.index = faiss.read_index(index_file)
        self.item_ids = np.load(id_file)

    def retrieve(self, user_embedding_np, top_k=100):
        """
        user_embedding_np: 1D array or 2D array [batch, 64] float32
        Returns: list of (movie_id, score), search_latency_ms
        """
        if user_embedding_np.ndim == 1:
            user_embedding_np = user_embedding_np.reshape(1, -1)

        # Ensure L2-normalized
        faiss.normalize_L2(user_embedding_np)

        t0 = time.perf_counter()
        scores, indices = self.index.search(user_embedding_np, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if 0 <= idx < len(self.item_ids):
                results.append({
                    "movieId": int(self.item_ids[idx]),
                    "retrieval_score": float(score),
                    "retrieval_rank": rank + 1
                })

        return results, latency_ms

if __name__ == "__main__":
    build_faiss_index()

