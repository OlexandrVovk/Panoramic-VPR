"""Pittsburgh 250k dataset handler."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .base import BaseDataset


class Pitts250kDataset(BaseDataset):
    """
    Pittsburgh 250k panoramic VPR dataset.

    Pano filenames:  @UTM_E@UTM_N@zone@band@lat@lon@id@full@.jpg
    Query filenames: @UTM_E@UTM_N@zone@band@lat@lon@id@idx@@@@@@pitch_yaw@.jpg
    """

    def __init__(self, root_dir: Path, split: str = "test", utm_threshold: float = 25.0):
        super().__init__(root_dir, split)
        self.utm_threshold = utm_threshold

    def get_database_paths(self) -> list[Path]:
        if self._db_paths is None:
            db_dir = self.root_dir / "images" / self.split / "database_pano_clean"
            self._db_paths = sorted(db_dir.glob("*.jpg"))
        return self._db_paths

    def get_query_paths(self) -> list[Path]:
        if self._query_paths is None:
            q_dir = self.root_dir / "images" / self.split / "queries_split"
            self._query_paths = sorted(q_dir.glob("*.jpg"))
        return self._query_paths

    def parse_pano_filename(self, path: Path) -> dict:
        # @UTM_E@UTM_N@zone@band@lat@lon@id@full@.jpg
        parts = path.stem.split("@")
        # parts[0] is empty (leading @)
        return {
            "utm_e": float(parts[1]),
            "utm_n": float(parts[2]),
            "zone": int(parts[3]),
            "band": parts[4],
            "lat": float(parts[5]),
            "lon": float(parts[6]),
            "id": parts[7],
        }

    def parse_query_filename(self, path: Path) -> dict:
        # @UTM_E@UTM_N@zone@band@lat@lon@id@idx@@@@@@pitch_yaw@.jpg
        parts = path.stem.split("@")
        return {
            "utm_e": float(parts[1]),
            "utm_n": float(parts[2]),
            "zone": int(parts[3]),
            "band": parts[4],
            "lat": float(parts[5]),
            "lon": float(parts[6]),
            "id": parts[7],
            "idx": parts[8],
            "pitch_yaw": parts[14] if len(parts) > 14 else None,
        }

    def build_ground_truth(self) -> dict[int, list[int]]:
        db_paths = self.get_database_paths()
        q_paths = self.get_query_paths()

        db_coords = np.array(
            [[self.parse_pano_filename(p)["utm_e"], self.parse_pano_filename(p)["utm_n"]] for p in db_paths]
        )
        q_coords = np.array(
            [[self.parse_query_filename(p)["utm_e"], self.parse_query_filename(p)["utm_n"]] for p in q_paths]
        )

        dist_matrix = cdist(q_coords, db_coords)  # (N_q, N_db)

        ground_truth = {}
        for i in range(len(q_paths)):
            positives = np.where(dist_matrix[i] < self.utm_threshold)[0].tolist()
            if positives:
                ground_truth[i] = positives

        return ground_truth
