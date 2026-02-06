"""Abstract base class for dataset handling."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDataset(ABC):
    """Base class defining the interface for VPR datasets."""

    def __init__(self, root_dir: Path, split: str):
        self.root_dir = Path(root_dir)
        self.split = split
        self._db_paths: list[Path] | None = None
        self._query_paths: list[Path] | None = None

    @abstractmethod
    def get_database_paths(self) -> list[Path]:
        """Return sorted list of panorama image paths."""

    @abstractmethod
    def get_query_paths(self) -> list[Path]:
        """Return sorted list of query image paths."""

    @abstractmethod
    def parse_pano_filename(self, path: Path) -> dict:
        """Extract metadata from panorama filename."""

    @abstractmethod
    def parse_query_filename(self, path: Path) -> dict:
        """Extract metadata from query filename."""

    @abstractmethod
    def build_ground_truth(self) -> dict[int, list[int]]:
        """Return mapping: query_index -> list of correct database indices."""
