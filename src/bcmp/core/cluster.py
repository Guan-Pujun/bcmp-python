"""Graph construction and clustering helpers for BCMP core."""

from __future__ import annotations

import gc

import igraph as ig
import leidenalg
import numpy as np
import scipy.sparse as sp

from bcmp.constants import DEFAULT_SEED
from bcmp.utils.validation import require_int32_cell_count

__all__ = ["cluster_labels_from_edge_matrix"]

ALL_SINGLETON_CLUSTER_ERROR = (
    "All clusters are singletons after initial clustering; BCMP cannot proceed."
)
IGRAPH_TARGET_EDGES_PER_CHUNK = 250_000_000
IGRAPH_MAX_EDGE_ADD_CHUNKS = 8
_LEIDEN_CLUSTER_RESOLUTION = 0.1


def _contiguous_chunk_bounds(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    n_items = int(n_items)
    n_chunks = max(1, int(n_chunks))
    if n_items <= 0:
        return []
    n_chunks = min(n_chunks, n_items)
    base, remainder = divmod(n_items, n_chunks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for chunk_idx in range(n_chunks):
        width = base + (1 if chunk_idx < remainder else 0)
        end = start + width
        if end > start:
            bounds.append((start, end))
        start = end
    return bounds


def _resolve_igraph_edge_chunk_count(
    upper_nnz: int,
    target_edges_per_chunk: int = IGRAPH_TARGET_EDGES_PER_CHUNK,
    max_chunks: int = IGRAPH_MAX_EDGE_ADD_CHUNKS,
) -> int:
    upper_nnz = int(upper_nnz)
    target_edges_per_chunk = max(1, int(target_edges_per_chunk))
    max_chunks = max(1, int(max_chunks))
    chunk_count = max(
        1, (upper_nnz + target_edges_per_chunk - 1) // target_edges_per_chunk
    )
    return min(chunk_count, max_chunks)


def _group_singletons(
    labels: np.ndarray, snn: sp.csr_matrix, seed: int = DEFAULT_SEED
) -> np.ndarray:
    labels = labels.astype(int, copy=True)
    uniq, counts = np.unique(labels, return_counts=True)
    singleton_labels = [
        int(u) for u, c in zip(uniq.tolist(), counts.tolist()) if int(c) == 1
    ]
    if not singleton_labels:
        return labels

    snn = snn.tocsr()
    cluster_labels = [
        int(u) for u in np.unique(labels).tolist() if int(u) not in singleton_labels
    ]
    if not cluster_labels:
        raise ValueError(ALL_SINGLETON_CLUSTER_ERROR)
    cluster_cells_by_label = {
        cluster_label: np.flatnonzero(labels == cluster_label).astype(int).tolist()
        for cluster_label in cluster_labels
    }
    rng = np.random.RandomState(int(seed))
    for singleton in singleton_labels:
        singleton_cells = np.flatnonzero(labels == singleton)
        if singleton_cells.size < 1:
            continue
        connectivity: dict[int, float] = {}
        for cluster_label in cluster_labels:
            cluster_cells = cluster_cells_by_label[cluster_label]
            sub_snn = snn[singleton_cells[:, None], cluster_cells]
            if sub_snn.shape[0] == 0 or sub_snn.shape[1] == 0:
                connectivity[cluster_label] = 0.0
            else:
                connectivity[cluster_label] = float(sub_snn.sum()) / float(
                    sub_snn.shape[0] * sub_snn.shape[1]
                )
        max_connectivity = max(connectivity.values())
        tied = sorted(
            [
                cluster
                for cluster, score in connectivity.items()
                if score == max_connectivity
            ]
        )
        chosen = int(rng.choice(tied, size=1)[0])
        labels[singleton_cells] = chosen
        cluster_cells_by_label[chosen].extend(
            int(cell) for cell in singleton_cells.tolist()
        )
    _, relabeled = np.unique(labels, return_inverse=True)
    return relabeled.astype(int)


def _graph_from_edge_matrix(
    n_cells: int,
    edges: np.ndarray,
) -> tuple[ig.Graph, np.ndarray]:
    upper_nnz = int(edges.shape[0])
    graph_edges = np.asarray(edges, dtype=np.int32)
    graph = ig.Graph(n=int(n_cells), directed=False)
    chunk_count = _resolve_igraph_edge_chunk_count(upper_nnz)
    chunk_bounds = _contiguous_chunk_bounds(upper_nnz, chunk_count)
    for start, end in chunk_bounds:
        graph.add_edges(graph_edges[start:end])
    return graph, graph_edges


def _csr_from_edge_matrix(
    n_cells: int,
    edges: np.ndarray,
) -> sp.csr_matrix:
    if edges.shape[0] == 0:
        return sp.identity(n_cells, dtype=np.float32, format="csr")
    src = np.asarray(edges[:, 0], dtype=np.int32)
    dst = np.asarray(edges[:, 1], dtype=np.int32)
    diag_idx = np.arange(n_cells, dtype=np.int32)
    rows = np.concatenate([diag_idx, src, dst]).astype(np.int32, copy=False)
    cols = np.concatenate([diag_idx, dst, src]).astype(np.int32, copy=False)
    edge_weights = np.ones(edges.shape[0], dtype=np.float32)
    data = np.concatenate(
        [np.ones(n_cells, dtype=np.float32), edge_weights, edge_weights]
    )
    snn = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32
    )
    snn.sort_indices()
    return snn


def cluster_labels_from_edge_matrix(
    n_cells: int,
    edges: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    require_int32_cell_count(int(n_cells), context="BCMP clustering")
    graph, edge_matrix = _graph_from_edge_matrix(
        n_cells=int(n_cells),
        edges=edges,
    )
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=float(_LEIDEN_CLUSTER_RESOLUTION),
        seed=int(seed),
        n_iterations=10,
    )
    labels = np.asarray(partition.membership, dtype=int)
    del partition
    del graph

    _, counts = np.unique(labels, return_counts=True)
    if bool(np.all(counts == 1)):
        raise ValueError(ALL_SINGLETON_CLUSTER_ERROR)
    if not bool(np.any(counts == 1)):
        return labels

    # Help igraph/leiden objects release before constructing the regrouping CSR on large graphs.
    gc.collect()
    snn = _csr_from_edge_matrix(
        n_cells=int(n_cells),
        edges=edge_matrix,
    )
    labels = _group_singletons(labels, snn, seed=seed)
    return labels
