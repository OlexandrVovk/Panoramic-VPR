"""Database serialization and FAISS index management."""

from pathlib import Path

import faiss
import numpy as np
import torch


def save_database(database: dict, path: Path) -> None:
    """
    Save descriptor database to .pt file and build FAISS indices.

    Creates:
        - {path}.pt  — torch database with descriptors and metadata
        - {path}_coarse.faiss — FAISS index on root descriptors (log-mapped)
    """
    path = Path(path)
    torch.save(database, path)

    # Build and save coarse FAISS index from log-mapped root descriptors
    if "root_descriptors_euclidean" in database:
        roots_np = database["root_descriptors_euclidean"].cpu().numpy().astype(np.float32)
        dim = roots_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(roots_np)
        faiss_path = str(path).replace(".pt", "_coarse.faiss")
        faiss.write_index(index, faiss_path)


def load_database(path: Path) -> dict:
    """Load descriptor database from .pt file."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_faiss_index(path: Path) -> faiss.Index:
    """Load a FAISS index from file."""
    return faiss.read_index(str(path))


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS IndexFlatL2 from a matrix of vectors.

    Args:
        vectors: (N, d) float32 array.

    Returns:
        FAISS index with N vectors added.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index
