"""Two-stage FAISS-accelerated retrieval engine."""

from pathlib import Path

import faiss
import numpy as np
import torch
from torch import Tensor

from ..config import PipelineConfig
from ..hyperbolic.poincare import hyperbolic_distance, log_map_zero


class RetrievalEngine:
    """
    Two-stage retrieval: FAISS coarse search on roots, then exact
    hyperbolic distance re-ranking on children of top candidates.
    """

    def __init__(self, database: dict, config: PipelineConfig):
        self.root_descriptors = database["root_descriptors"]  # (N, d)
        self.child_descriptors = database["child_descriptors"]  # (N, K, d)
        self.view_metadata = database["view_metadata"]
        self.pano_metadata = database["pano_metadata"]
        self.config = config

        # Build FAISS index from log-mapped root descriptors
        if "root_descriptors_euclidean" in database:
            roots_eucl = database["root_descriptors_euclidean"]
        else:
            roots_eucl = log_map_zero(self.root_descriptors)

        roots_np = roots_eucl.cpu().numpy().astype(np.float32)
        roots_np = np.ascontiguousarray(roots_np)
        self.coarse_index = faiss.IndexFlatL2(roots_np.shape[1])
        self.coarse_index.add(roots_np)

    def retrieve(self, query_hyp: Tensor, top_k_coarse: int | None = None) -> list[dict]:
        """
        Two-stage retrieval for a single query.

        Args:
            query_hyp: (d,) query descriptor in the Poincare ball.
            top_k_coarse: Number of coarse candidates (default from config).

        Returns:
            List of result dicts sorted by fine distance, containing:
              pano_idx, view_idx, yaw, pitch, coarse_distance, fine_distance, pano_metadata.
        """
        top_k = top_k_coarse or self.config.top_k_coarse

        # Stage 1: Coarse FAISS search on root descriptors
        query_eucl = log_map_zero(query_hyp.unsqueeze(0))  # (1, d)
        query_np = query_eucl.cpu().numpy().astype(np.float32)
        query_np = np.ascontiguousarray(query_np)
        distances, indices = self.coarse_index.search(query_np, top_k)  # (1, top_k)

        candidate_pano_idxs = indices[0]
        coarse_dists = distances[0]

        # Stage 2: Fine re-ranking — exact hyperbolic distance on children
        results = []
        for rank, pano_idx in enumerate(candidate_pano_idxs):
            if pano_idx < 0:  # FAISS returns -1 for empty slots
                continue
            children = self.child_descriptors[pano_idx]  # (K, d)
            fine_dists = hyperbolic_distance(
                query_hyp.unsqueeze(0).expand_as(children), children
            )  # (K,)

            for view_idx in range(children.shape[0]):
                view_meta = self.view_metadata[pano_idx][view_idx]
                results.append({
                    "pano_idx": int(pano_idx),
                    "view_idx": int(view_idx),
                    "yaw": view_meta["yaw"],
                    "pitch": view_meta["pitch"],
                    "coarse_distance": float(coarse_dists[rank]),
                    "fine_distance": float(fine_dists[view_idx]),
                    "pano_metadata": self.pano_metadata[pano_idx],
                })

        results.sort(key=lambda x: x["fine_distance"])
        return results

    def retrieve_batch(
        self, queries_hyp: Tensor, top_k_coarse: int | None = None
    ) -> list[list[int]]:
        """
        Batch retrieval returning ranked panorama indices per query.

        Optimized for evaluation: returns only the ranked pano indices
        (deduplicated) rather than full result dicts.

        Args:
            queries_hyp: (Q, d) query descriptors in the Poincare ball.
            top_k_coarse: Number of coarse candidates.

        Returns:
            List of Q lists, each containing ranked panorama indices.
        """
        top_k = top_k_coarse or self.config.top_k_coarse
        max_recall_k = max(self.config.recall_ks) if self.config.recall_ks else 10

        # Stage 1: Batch FAISS search
        queries_eucl = log_map_zero(queries_hyp)  # (Q, d)
        queries_np = queries_eucl.cpu().numpy().astype(np.float32)
        queries_np = np.ascontiguousarray(queries_np)
        _, coarse_indices = self.coarse_index.search(queries_np, top_k)  # (Q, top_k)

        all_predictions = []
        for qi in range(queries_hyp.shape[0]):
            query = queries_hyp[qi]  # (d,)
            candidates = coarse_indices[qi]

            # Fine re-ranking
            pano_best_dist: dict[int, float] = {}
            for pano_idx in candidates:
                if pano_idx < 0:
                    continue
                children = self.child_descriptors[pano_idx]  # (K, d)
                fine_dists = hyperbolic_distance(
                    query.unsqueeze(0).expand_as(children), children
                )
                best_dist = float(fine_dists.min())
                pano_best_dist[int(pano_idx)] = best_dist

            # Sort panoramas by best child distance
            ranked = sorted(pano_best_dist.items(), key=lambda x: x[1])
            ranked_idxs = [idx for idx, _ in ranked][:max_recall_k]
            all_predictions.append(ranked_idxs)

        return all_predictions
