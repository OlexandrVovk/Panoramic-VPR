"""YQ360 dataset handler."""

from pathlib import Path

from .base import BaseDataset


class YQ360Dataset(BaseDataset):
    """
    YQ360 panoramic VPR dataset.

    Pano filenames:  ER@id@UTM_E@UTM_N@zone@band@lat@lon.jpg
    Query filenames: Camera@id@UTM_E@UTM_N@zone@band@lat@lon@Direction.jpg
    """

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
        # ER@id@UTM_E@UTM_N@zone@band@lat@lon.jpg
        parts = path.stem.split("@")
        return {
            "prefix": parts[0],  # "ER"
            "id": parts[1],
            "utm_e": float(parts[2]),
            "utm_n": float(parts[3]),
            "zone": int(parts[4]),
            "band": parts[5],
            "lat": float(parts[6]),
            "lon": float(parts[7]),
        }

    def parse_query_filename(self, path: Path) -> dict:
        # Camera@id@UTM_E@UTM_N@zone@band@lat@lon@Direction.jpg
        parts = path.stem.split("@")
        return {
            "prefix": parts[0],  # "Camera"
            "id": parts[1],
            "utm_e": float(parts[2]),
            "utm_n": float(parts[3]),
            "zone": int(parts[4]),
            "band": parts[5],
            "lat": float(parts[6]),
            "lon": float(parts[7]),
            "direction": parts[8],
        }

    def build_ground_truth(self) -> dict[int, list[int]]:
        db_paths = self.get_database_paths()
        q_paths = self.get_query_paths()

        # Build pano ID -> database index mapping
        id_to_db_idx: dict[str, int] = {}
        for idx, p in enumerate(db_paths):
            meta = self.parse_pano_filename(p)
            id_to_db_idx[meta["id"]] = idx

        ground_truth = {}
        for qi, p in enumerate(q_paths):
            meta = self.parse_query_filename(p)
            db_idx = id_to_db_idx.get(meta["id"])
            if db_idx is not None:
                ground_truth[qi] = [db_idx]

        return ground_truth
