"""Dataset factory."""

from pathlib import Path

from .base import BaseDataset
from .pitts250k import Pitts250kDataset
from .yq360 import YQ360Dataset

DATASETS: dict[str, type[BaseDataset]] = {
    "pitts250k": Pitts250kDataset,
    "yq360": YQ360Dataset,
}


def get_dataset(name: str, root_dir: str | Path, split: str = "test", **kwargs) -> BaseDataset:
    """Create a dataset instance by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name](Path(root_dir), split, **kwargs)
