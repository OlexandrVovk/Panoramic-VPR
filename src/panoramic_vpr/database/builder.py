"""Full 4-stage database building pipeline."""

from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from ..backbone.megaloc import MegaLocBackbone
from ..config import PipelineConfig
from ..datasets.base import BaseDataset
from ..hyperbolic.aggregation import einstein_midpoint, weighted_einstein_midpoint
from ..hyperbolic.poincare import exp_map_zero, log_map_zero
from ..projection.equirect_to_persp import generate_perspective_views


class DatabaseBuilder:
    """Orchestrates the panorama descriptor database construction."""

    def __init__(self, config: PipelineConfig, backbone: MegaLocBackbone):
        self.config = config
        self.backbone = backbone
        self.aggregate_fn = (
            weighted_einstein_midpoint if config.aggregation == "weighted" else einstein_midpoint
        )

    def build(self, dataset: BaseDataset) -> dict:
        """
        Build the complete hyperbolic descriptor database.

        Pipeline per panorama:
          1. Load equirectangular image
          2. Generate K perspective views
          3. Extract Euclidean features (MegaLoc, batched)
          4. Map to Poincare ball (exp_map_zero)
          5. Aggregate into root descriptor (Einstein midpoint)

        Returns:
            dict with:
              - root_descriptors: Tensor (N, d)
              - child_descriptors: Tensor (N, K, d)
              - root_descriptors_euclidean: Tensor (N, d) — log-mapped roots for FAISS
              - view_metadata: list of list of dicts
              - pano_metadata: list of dicts
        """
        pano_paths = dataset.get_database_paths()
        cfg = self.config

        all_roots = []
        all_children = []
        all_view_meta = []
        all_pano_meta = []

        # Collect views in batches for efficient backbone inference
        pending_views: list[tuple[int, int, dict]] = []  # (pano_idx, view_idx, meta)
        pending_images: list = []
        pano_euclid_descs: dict[int, list] = {}  # pano_idx -> list of (view_idx, descriptor)
        pano_view_metas: dict[int, list] = {}

        for pano_idx, pano_path in enumerate(tqdm(pano_paths, desc="Processing panoramas")):
            equirect = cv2.imread(str(pano_path))
            if equirect is None:
                raise RuntimeError(f"Failed to load panorama: {pano_path}")

            views = generate_perspective_views(
                equirect,
                num_views=cfg.num_views,
                fov=cfg.fov,
                pitch=cfg.pitch,
                width=cfg.view_width,
                height=cfg.view_height,
            )

            pano_meta = {"path": str(pano_path), "index": pano_idx}
            if hasattr(dataset, "parse_pano_filename"):
                pano_meta.update(dataset.parse_pano_filename(pano_path))

            pano_euclid_descs[pano_idx] = []
            pano_view_metas[pano_idx] = []

            for view_idx, (view_img, view_meta) in enumerate(views):
                pending_views.append((pano_idx, view_idx, view_meta))
                pending_images.append(view_img)
                pano_view_metas[pano_idx].append(view_meta)

            # Flush batch when full or at last panorama
            if len(pending_images) >= cfg.batch_size or pano_idx == len(pano_paths) - 1:
                if pending_images:
                    descs = self.backbone.extract(pending_images)  # (B, 8448)
                    for i, (pi, vi, _) in enumerate(pending_views):
                        pano_euclid_descs[pi].append((vi, descs[i].cpu()))
                    pending_views.clear()
                    pending_images.clear()

            # Once all views of a panorama have descriptors, aggregate
            if pano_idx in pano_euclid_descs and len(pano_euclid_descs[pano_idx]) == cfg.num_views:
                # Sort by view index
                sorted_descs = sorted(pano_euclid_descs[pano_idx], key=lambda x: x[0])
                euclid = torch.stack([d for _, d in sorted_descs])  # (K, d)

                # Stage 3: Exponential map to Poincare ball
                children_hyp = exp_map_zero(euclid)  # (K, d)

                # Stage 4: Aggregate into root
                root_hyp = self.aggregate_fn(children_hyp)  # (d,)

                all_roots.append(root_hyp)
                all_children.append(children_hyp)
                all_view_meta.append(pano_view_metas[pano_idx])
                all_pano_meta.append(pano_meta)

                del pano_euclid_descs[pano_idx]
                del pano_view_metas[pano_idx]

        # Handle any remaining panoramas with pending descriptors
        if pending_images:
            descs = self.backbone.extract(pending_images)
            for i, (pi, vi, _) in enumerate(pending_views):
                pano_euclid_descs[pi].append((vi, descs[i].cpu()))

        for pi in sorted(pano_euclid_descs.keys()):
            if len(pano_euclid_descs[pi]) == cfg.num_views:
                sorted_descs = sorted(pano_euclid_descs[pi], key=lambda x: x[0])
                euclid = torch.stack([d for _, d in sorted_descs])
                children_hyp = exp_map_zero(euclid)
                root_hyp = self.aggregate_fn(children_hyp)

                all_roots.append(root_hyp)
                all_children.append(children_hyp)
                all_view_meta.append(pano_view_metas[pi])
                # Find the pano_meta for this index
                pano_path = pano_paths[pi]
                pano_meta = {"path": str(pano_path), "index": pi}
                if hasattr(dataset, "parse_pano_filename"):
                    pano_meta.update(dataset.parse_pano_filename(pano_path))
                all_pano_meta.append(pano_meta)

        root_descriptors = torch.stack(all_roots)  # (N, d)
        child_descriptors = torch.stack(all_children)  # (N, K, d)

        # Log-map roots back to Euclidean for FAISS indexing
        root_descriptors_euclidean = log_map_zero(root_descriptors)

        return {
            "root_descriptors": root_descriptors,
            "child_descriptors": child_descriptors,
            "root_descriptors_euclidean": root_descriptors_euclidean,
            "view_metadata": all_view_meta,
            "pano_metadata": all_pano_meta,
            "config": {
                "num_views": cfg.num_views,
                "fov": cfg.fov,
                "pitch": cfg.pitch,
                "aggregation": cfg.aggregation,
                "descriptor_dim": cfg.descriptor_dim,
            },
        }
