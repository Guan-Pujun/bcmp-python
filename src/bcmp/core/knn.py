"""KNN search helpers for BCMP core."""

from __future__ import annotations

from annoy import AnnoyIndex
import numpy as np

from bcmp.constants import DEFAULT_SEED
from bcmp.utils.validation import require_int32_cell_count

__all__ = ["get_knn_neighbor_once"]


def get_knn_neighbor_once(
    embeddings: np.ndarray, k_max: int, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Build a single Annoy index and query neighbors up to ``k_max`` once."""
    embedding_shape = np.shape(embeddings)
    n_cells = int(embedding_shape[0])
    if n_cells < 3:
        raise ValueError("At least 3 cells are required for KNN search")
    require_int32_cell_count(n_cells, context="BCMP KNN")
    with np.errstate(over="ignore"):
        embeddings32 = np.asarray(embeddings, dtype=np.float32)
    if not bool(np.isfinite(embeddings32).all()):
        raise ValueError("embedding must contain only finite float32 values")

    k_use = min(int(k_max), embeddings32.shape[0] - 1)
    if k_use < 1:
        raise ValueError("k_max must be at least 1")
    index = AnnoyIndex(int(embeddings32.shape[1]), "euclidean")
    index.set_seed(int(seed))
    for i, row in enumerate(embeddings32):
        index.add_item(i, row.tolist())
    index.build(50)

    out = np.empty((embeddings32.shape[0], int(k_use)), dtype=np.int32)
    for i, row in enumerate(embeddings32):
        nn = index.get_nns_by_vector(
            row.tolist(),
            int(k_use),
            search_k=-1,
            include_distances=False,
        )
        if len(nn) != int(k_use):
            raise RuntimeError(
                f"Annoy returned {len(nn)} neighbors for cell {i}, expected {int(k_use)}"
            )
        out[i, :] = np.asarray(nn, dtype=np.int32)
    return out
