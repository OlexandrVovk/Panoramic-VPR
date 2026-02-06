"""Pipeline configuration."""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # Perspective view generation
    num_views: int = 8
    fov: float = 90.0
    pitch: float = 0.0
    view_width: int = 640
    view_height: int = 480

    # Backbone
    backbone_name: str = "megaloc"
    backbone_input_size: int = 322
    descriptor_dim: int = 8448

    # Hyperbolic space
    curvature: float = 1.0
    eps: float = 1e-5
    max_norm: float = 0.99

    # Retrieval
    top_k_coarse: int = 5
    aggregation: str = "standard"  # "standard" or "weighted"

    # Evaluation
    recall_ks: list[int] = field(default_factory=lambda: [1, 5, 10])

    # Ground truth (pitts250k)
    utm_threshold: float = 25.0

    # Processing
    batch_size: int = 32
    device: str = "cuda"

    @property
    def yaw_angles(self) -> list[float]:
        return [i * (360.0 / self.num_views) for i in range(self.num_views)]
