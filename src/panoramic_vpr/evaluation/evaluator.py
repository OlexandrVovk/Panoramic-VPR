"""End-to-end evaluation orchestrator."""

import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from ..backbone.megaloc import MegaLocBackbone
from ..config import PipelineConfig
from ..database.store import load_database
from ..datasets.base import BaseDataset
from ..hyperbolic.poincare import exp_map_zero
from ..retrieval.engine import RetrievalEngine
from .metrics import compute_recall_at_k


class Evaluator:
    """Runs full evaluation: encode queries, retrieve, compute recall."""

    def __init__(self, config: PipelineConfig, backbone: MegaLocBackbone):
        self.config = config
        self.backbone = backbone

    def evaluate(self, dataset: BaseDataset, database_path: Path) -> dict:
        """
        Full evaluation pipeline.

        1. Load pre-built database
        2. Build ground truth
        3. Encode all queries (batched)
        4. Map to Poincare ball
        5. Run batch retrieval
        6. Compute recall metrics

        Returns:
            Dict with recall metrics and timing info.
        """
        # Load database
        print("Loading database...")
        database = load_database(database_path)

        # Build ground truth
        print("Building ground truth...")
        ground_truth = dataset.build_ground_truth()
        print(f"  {len(ground_truth)} queries with ground truth")

        # Encode queries
        print("Encoding queries...")
        query_paths = dataset.get_query_paths()
        query_descriptors = self._encode_queries(query_paths)

        # Map to Poincare ball
        query_hyp = exp_map_zero(query_descriptors)  # (Q, d)

        # Retrieval
        print("Running retrieval...")
        engine = RetrievalEngine(database, self.config)

        t_start = time.perf_counter()
        predictions = engine.retrieve_batch(query_hyp, self.config.top_k_coarse)
        t_elapsed = time.perf_counter() - t_start

        # Compute recall
        results = compute_recall_at_k(predictions, ground_truth, self.config.recall_ks)
        results["retrieval_time_s"] = t_elapsed
        results["avg_time_per_query_ms"] = (t_elapsed / len(query_paths)) * 1000

        return results

    def _encode_queries(self, query_paths: list[Path]) -> torch.Tensor:
        """Encode all query images in batches."""
        all_descs = []
        batch_imgs = []

        for path in tqdm(query_paths, desc="Encoding queries"):
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Failed to load query: {path}")
            batch_imgs.append(img)

            if len(batch_imgs) >= self.config.batch_size:
                descs = self.backbone.extract(batch_imgs)
                all_descs.append(descs.cpu())
                batch_imgs.clear()

        if batch_imgs:
            descs = self.backbone.extract(batch_imgs)
            all_descs.append(descs.cpu())

        return torch.cat(all_descs, dim=0)  # (Q, d)
