"""SNN construction helpers for BCMP core."""

from __future__ import annotations

import math

import numba
import numpy as np

from bcmp.utils.validation import require_int32_cell_count

__all__ = ["snn_upper_edge_matrix_from_knn_ranked"]

_SEURAT_SNN_PRUNE = 1 / 15


def _min_shared_neighbors_for_prune(k: int, prune_snn: float) -> int:
    prune = float(prune_snn)
    if not np.isfinite(prune) or prune < 0:
        raise ValueError(
            f"prune_snn must be a finite non-negative value; got {prune_snn}"
        )
    if prune == 0.0:
        return 0
    threshold = ((2.0 * float(k)) * prune) / (1.0 + prune)
    min_shared = int(math.ceil(threshold - 1e-12))
    return max(0, min(int(k), min_shared))


@numba.njit
def _build_knn_postings_exact(nn_idx_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cells, k = nn_idx_k.shape
    counts = np.zeros(n_cells, dtype=np.int64)
    for i in range(n_cells):
        for t in range(k):
            counts[int(nn_idx_k[i, t])] += 1

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    for i in range(n_cells):
        indptr[i + 1] = indptr[i] + counts[i]

    post_rows = np.empty(indptr[-1], dtype=np.int32)
    write_ptr = indptr[:-1].copy()
    for i in range(n_cells):
        for t in range(k):
            neighbor = int(nn_idx_k[i, t])
            pos = write_ptr[neighbor]
            post_rows[pos] = i
            write_ptr[neighbor] = pos + 1
    return indptr, post_rows


@numba.njit(parallel=True)
def _count_snn_upper_row_nnz_exact(
    nn_idx_k: np.ndarray,
    post_indptr: np.ndarray,
    post_rows: np.ndarray,
    min_shared: int,
) -> np.ndarray:
    n_cells, k = nn_idx_k.shape
    row_nnz = np.zeros(n_cells, dtype=np.int64)
    n_threads = numba.get_num_threads()
    stamps = np.zeros((n_threads, n_cells), dtype=np.int32)
    counts = np.zeros((n_threads, n_cells), dtype=np.int32)
    active = np.empty((n_threads, n_cells), dtype=np.int32)

    for i in numba.prange(n_cells):
        tid = numba.get_thread_id()
        stamp = stamps[tid]
        count = counts[tid]
        active_rows = active[tid]
        active_len = 0
        stamp_value = i + 1
        for t in range(k):
            neighbor = int(nn_idx_k[i, t])
            start = post_indptr[neighbor]
            end = post_indptr[neighbor + 1]
            for pos in range(start, end):
                j = int(post_rows[pos])
                if j <= i:
                    continue
                if stamp[j] != stamp_value:
                    stamp[j] = stamp_value
                    count[j] = 1
                    active_rows[active_len] = j
                    active_len += 1
                else:
                    count[j] += 1

        row_total = 0
        for idx in range(active_len):
            j = int(active_rows[idx])
            if count[j] >= min_shared:
                row_total += 1
        row_nnz[i] = row_total
    return row_nnz


@numba.njit(parallel=True)
def _fill_snn_upper_edge_matrix_exact(
    nn_idx_k: np.ndarray,
    post_indptr: np.ndarray,
    post_rows: np.ndarray,
    min_shared: int,
    upper_indptr: np.ndarray,
) -> np.ndarray:
    n_cells, k = nn_idx_k.shape
    edges = np.empty((upper_indptr[-1], 2), dtype=np.int32)
    n_threads = numba.get_num_threads()
    stamps = np.zeros((n_threads, n_cells), dtype=np.int32)
    counts = np.zeros((n_threads, n_cells), dtype=np.int32)
    active = np.empty((n_threads, n_cells), dtype=np.int32)

    for i in numba.prange(n_cells):
        tid = numba.get_thread_id()
        stamp = stamps[tid]
        count = counts[tid]
        active_rows = active[tid]
        active_len = 0
        stamp_value = i + 1
        for t in range(k):
            neighbor = int(nn_idx_k[i, t])
            start = post_indptr[neighbor]
            end = post_indptr[neighbor + 1]
            for pos in range(start, end):
                j = int(post_rows[pos])
                if j <= i:
                    continue
                if stamp[j] != stamp_value:
                    stamp[j] = stamp_value
                    count[j] = 1
                    active_rows[active_len] = j
                    active_len += 1
                else:
                    count[j] += 1

        active_sorted = active_rows[:active_len]
        active_sorted.sort()
        write_pos = upper_indptr[i]
        for idx in range(active_len):
            j = int(active_sorted[idx])
            shared = count[j]
            if shared >= min_shared:
                edges[write_pos, 0] = i
                edges[write_pos, 1] = j
                write_pos += 1

    return edges


def snn_upper_edge_matrix_from_knn_ranked(
    nn_idx_k: np.ndarray,
) -> np.ndarray:
    nn_shape = np.shape(nn_idx_k)
    n_cells, k = int(nn_shape[0]), int(nn_shape[1])
    require_int32_cell_count(n_cells, context="BCMP SNN")
    if not isinstance(nn_idx_k, np.ndarray) or nn_idx_k.dtype != np.int32:
        nn_idx_k = np.asarray(nn_idx_k, dtype=np.int32)
    else:
        nn_idx_k = np.asarray(nn_idx_k)
    min_shared = _min_shared_neighbors_for_prune(k=k, prune_snn=_SEURAT_SNN_PRUNE)
    post_indptr, post_rows = _build_knn_postings_exact(nn_idx_k)
    upper_row_nnz = _count_snn_upper_row_nnz_exact(
        nn_idx_k=nn_idx_k,
        post_indptr=post_indptr,
        post_rows=post_rows,
        min_shared=min_shared,
    )
    upper_indptr = np.empty(n_cells + 1, dtype=np.int64)
    upper_indptr[0] = 0
    np.cumsum(upper_row_nnz, out=upper_indptr[1:])
    edges = _fill_snn_upper_edge_matrix_exact(
        nn_idx_k=nn_idx_k,
        post_indptr=post_indptr,
        post_rows=post_rows,
        min_shared=min_shared,
        upper_indptr=upper_indptr,
    )
    return edges
