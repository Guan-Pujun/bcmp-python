"""AnnData-aware preprocessing and embedding extraction helpers."""

from __future__ import annotations

import warnings
from typing import Any

import anndata as ad
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

from bcmp.constants import (
    DEFAULT_BATCH_KEY,
    DEFAULT_PARTITION_N_HVG,
    DEFAULT_PARTITION_N_PCS,
    DEFAULT_SEED,
)
from bcmp.utils.sparse import as_csr_matrix
from bcmp.core.trace import emit_partition_warning
from ._glue import _require_batch_key, _var_names

__all__ = [
    "normalize_log1p_counts",
    "select_highly_variable_genes",
    "run_uncorrected_umap2d",
    "run_standard_bcmp_preprocess",
]

BATCH_HVG_MIN_CELLS = 3


def normalize_log1p_counts(
    counts: sp.csr_matrix, target_sum: float = 1e4
) -> sp.csr_matrix:
    counts = as_csr_matrix(counts)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(
        float(target_sum),
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals > 0,
    )
    normalized = counts.astype(np.float64, copy=True)
    indptr = normalized.indptr
    data = normalized.data
    for row_idx, factor in enumerate(scale):
        data[indptr[row_idx] : indptr[row_idx + 1]] *= factor
    np.log1p(data, out=data)
    return normalized


def _validate_nonnegative_finite_counts(counts: sp.csr_matrix) -> None:
    data = as_csr_matrix(counts).data
    try:
        finite = np.isfinite(data)
        has_negative = np.any(data < 0.0)
    except TypeError as exc:
        raise ValueError("adata.X must contain only finite count values") from exc
    if not bool(finite.all()):
        raise ValueError("adata.X must contain only finite count values")
    if bool(has_negative):
        raise ValueError("adata.X must contain non-negative count values")


def _sample_variances_from_moments(
    sums: np.ndarray,
    means: np.ndarray,
    sum_sq: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    variances = (
        sum_sq - (2.0 * means * sums) + (float(n_cells) * np.square(means))
    ) / float(n_cells - 1)
    return np.maximum(variances, 0.0)


def _csc_column_sum_squares(counts: sp.csc_matrix) -> np.ndarray:
    squared = sp.csc_matrix(
        (np.square(counts.data), counts.indices, counts.indptr),
        shape=counts.shape,
        copy=False,
    )
    return np.asarray(squared.sum(axis=0), dtype=np.float64).ravel()


def _stabilize_duplicate_loess_predictor(
    values: np.ndarray,
    jitter_width: float = 1e-12,
) -> np.ndarray:
    predictor = np.asarray(values, dtype=np.float64)
    if predictor.size < 2:
        return predictor.copy()
    order = np.argsort(predictor, kind="mergesort")
    sorted_values = predictor[order]
    stabilized = predictor.copy()
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            stabilized[order[start:end]] = sorted_values[start] + np.linspace(
                -float(jitter_width),
                float(jitter_width),
                num=end - start,
                dtype=np.float64,
            )
        start = end
    return stabilized


def _fit_seurat_v5_loess(
    log10_means: np.ndarray,
    log10_variances: np.ndarray,
    span: float,
):
    from skmisc.loess import loess

    x = np.asarray(log10_means, dtype=np.float64)
    y = np.asarray(log10_variances, dtype=np.float64)
    fit = loess(x, y, span=float(span))
    try:
        fit.fit()
        return fit
    except ValueError as exc:
        message = str(exc)
        if (
            "reciprocal condition number" not in message
            and "near singularities" not in message
        ):
            raise
        # Some batches produce a highly discrete log10(mean) predictor with many exact duplicates.
        # That can make the local quadratic loess system nearly singular in skmisc. Break only the
        # duplicate x-ties by a tiny, deterministic jitter and retry, leaving the normal path unchanged.
        x_stable = _stabilize_duplicate_loess_predictor(x)
        if np.array_equal(x_stable, x):
            raise
        fit = loess(x_stable, y, span=float(span))
        fit.fit()
        return fit


@numba.njit
def _variance_standardized_seurat_v5_vst_from_csc(
    indptr: np.ndarray,
    data: np.ndarray,
    means: np.ndarray,
    sd: np.ndarray,
    n_cells: int,
    vmax: float,
) -> np.ndarray:
    n_genes = means.shape[0]
    variance_standardized = np.zeros(n_genes, dtype=np.float64)
    denom = float(n_cells - 1)
    for gene_idx in range(n_genes):
        sd_i = sd[gene_idx]
        if sd_i == 0.0 or not np.isfinite(sd_i):
            continue
        start = indptr[gene_idx]
        end = indptr[gene_idx + 1]
        nnz = end - start
        n_zero = n_cells - nnz
        mean_i = means[gene_idx]
        col_sum = 0.0
        if nnz > 0:
            for pos in range(nnz):
                z_i = (data[start + pos] - mean_i) / sd_i
                if z_i > vmax:
                    z_i = vmax
                col_sum += z_i * z_i
        zero_term = ((0.0 - mean_i) / sd_i) ** 2
        variance_standardized[gene_idx] = (col_sum + zero_term * n_zero) / denom
    return variance_standardized


def _select_highly_variable_genes_seurat_v5_vst_dgcmatrix(
    raw_counts: sp.csr_matrix,
    var_names: pd.Index,
    n_top_genes: int,
    span: float = 0.3,
    clip: float | None = None,
) -> list[str]:
    n_top = min(int(n_top_genes), int(len(var_names)))
    if n_top < 1:
        raise ValueError("n_top_genes must be positive")

    counts = sp.csc_matrix(as_csr_matrix(raw_counts), dtype=np.float64)
    n_cells, n_genes = counts.shape
    if n_cells < 2:
        raise ValueError("Seurat v5 VST requires at least 2 cells")

    sums = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    means = sums / float(n_cells)

    sum_sq = _csc_column_sum_squares(counts)
    variances = _sample_variances_from_moments(
        sums=sums,
        means=means,
        sum_sq=sum_sq,
        n_cells=int(n_cells),
    )

    variance_expected = np.zeros(n_genes, dtype=np.float64)
    not_const = variances > 0
    if np.any(not_const):
        fit = _fit_seurat_v5_loess(
            np.log10(means[not_const]),
            np.log10(variances[not_const]),
            span=float(span),
        )
        variance_expected[not_const] = np.power(10.0, fit.outputs.fitted_values)

    sd = np.sqrt(variance_expected)
    vmax = float(clip) if clip is not None else float(np.sqrt(float(n_cells)))
    variance_standardized = _variance_standardized_seurat_v5_vst_from_csc(
        indptr=counts.indptr,
        data=counts.data,
        means=means,
        sd=sd,
        n_cells=int(n_cells),
        vmax=float(vmax),
    )

    keep = means != 0
    candidate_pos = np.flatnonzero(keep)
    order = np.argsort(-variance_standardized[candidate_pos], kind="mergesort")
    selected_pos = candidate_pos[order[:n_top]]
    genes = pd.Index(var_names.astype(str))[selected_pos].astype(str).tolist()
    if len(genes) < 1:
        raise RuntimeError(
            "No highly variable genes were selected by Seurat v5 dgCMatrix VST"
        )
    return genes


def seurat_v5_consensus_features(
    features_by_layer: dict[str, list[str]],
    common_features: list[str] | pd.Index,
    nfeatures: int | None = None,
) -> list[str]:
    if not features_by_layer:
        return []

    common_set = set(pd.Index(common_features).astype(str))
    if not common_set:
        return []

    positions: dict[str, list[int]] = {}
    for features in features_by_layer.values():
        for index_1based, feature in enumerate(features, start=1):
            feature = str(feature)
            if feature not in common_set:
                continue
            positions.setdefault(feature, []).append(int(index_1based))

    if not positions:
        return []

    consensus_df = pd.DataFrame(
        {
            "feature": sorted(positions.keys()),
            "frequency": [
                len(positions[feature]) for feature in sorted(positions.keys())
            ],
            "median_position": [
                float(np.median(positions[feature]))
                for feature in sorted(positions.keys())
            ],
        }
    )
    consensus_df = consensus_df.sort_values(
        ["frequency", "median_position"],
        ascending=[False, True],
        kind="mergesort",
    )
    consensus = consensus_df["feature"].astype(str).tolist()
    if nfeatures is not None:
        consensus = consensus[: int(nfeatures)]
    return consensus


def select_highly_variable_genes(
    raw_counts: sp.csr_matrix,
    obs: pd.DataFrame,
    var_names: pd.Index,
    n_top_genes: int,
    batch_key: str = DEFAULT_BATCH_KEY,
) -> list[str]:
    if batch_key not in obs.columns:
        raise KeyError(f"Missing batch column in metadata: {batch_key}")
    n_top = min(int(n_top_genes), int(len(var_names)))
    if n_top < 1:
        raise ValueError("n_top_genes must be positive")

    batch_values = obs[batch_key].astype(str)
    batch_sizes = batch_values.value_counts(sort=False)
    small_batches = batch_sizes[batch_sizes < BATCH_HVG_MIN_CELLS]
    if not small_batches.empty:
        preview = ", ".join(
            f"{str(batch)}={int(count)}"
            for batch, count in small_batches.iloc[:5].items()
        )
        suffix = "" if small_batches.shape[0] <= 5 else ", ..."
        raise ValueError(
            "Each batch must contain at least "
            f"{BATCH_HVG_MIN_CELLS} cells for Seurat v5 split-layer HVG selection; "
            f"small batches: {preview}{suffix}"
        )
    batch_order = pd.Index(pd.unique(batch_values), dtype="object")
    features_by_layer: dict[str, list[str]] = {}
    common_features = pd.Index(var_names.astype(str))
    for batch in batch_order:
        batch_mask = (batch_values == batch).to_numpy(dtype=bool, copy=False)
        batch_counts = as_csr_matrix(raw_counts[batch_mask, :])
        layer_name = f"counts.{batch}"
        features_by_layer[layer_name] = (
            _select_highly_variable_genes_seurat_v5_vst_dgcmatrix(
                raw_counts=batch_counts,
                var_names=var_names,
                n_top_genes=n_top,
            )
        )

    genes = seurat_v5_consensus_features(
        features_by_layer=features_by_layer,
        common_features=common_features,
        nfeatures=n_top,
    )
    if len(genes) < 1:
        raise RuntimeError(
            "No highly variable genes were selected by Seurat v5 split-layer consensus"
        )
    return genes


def seurat_style_scale_dense(
    matrix: sp.csr_matrix, scale_max: float = 10.0
) -> np.ndarray:
    dense = as_csr_matrix(matrix).toarray().astype(float, copy=False)
    mean = dense.mean(axis=0)
    std = dense.std(axis=0, ddof=1)
    std[~np.isfinite(std)] = 0.0
    std[std == 0] = 1.0
    dense -= mean
    dense /= std
    dense[dense > float(scale_max)] = float(scale_max)
    dense[~np.isfinite(dense)] = 0.0
    return dense


def run_uncorrected_umap2d(
    pca_embeddings: np.ndarray,
    dims_use: list[int] | np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> np.ndarray | None:
    import umap

    dims_max = int(pca_embeddings.shape[1])
    if dims_max < 2:
        emit_partition_warning(
            step="umap",
            details="Skipping UMAP because reduction 'Uncorrectedpca' has fewer than 2 valid dims for UMAP.",
        )
        return None

    if dims_use is None:
        dims_use = list(range(min(DEFAULT_PARTITION_N_PCS, dims_max)))
    dims_use = [int(x) for x in dims_use if 0 <= int(x) < dims_max]
    dims_use = list(dict.fromkeys(dims_use))
    if len(dims_use) < 2:
        emit_partition_warning(
            step="umap",
            details="Skipping UMAP because reduction 'Uncorrectedpca' has fewer than 2 valid dims for UMAP.",
        )
        return None

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(30, pca_embeddings.shape[0] - 1)),
        metric="cosine",
        min_dist=0.3,
        random_state=int(seed),
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*force_all_finite.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*n_jobs value 1 overridden to 1 by setting random_state.*",
            category=UserWarning,
        )
        return np.asarray(
            reducer.fit_transform(pca_embeddings[:, dims_use]), dtype=float
        )


def run_standard_bcmp_preprocess(
    adata: ad.AnnData,
    batch_key: str = DEFAULT_BATCH_KEY,
    partition_n_pcs: int = DEFAULT_PARTITION_N_PCS,
    partition_n_hvg: int = DEFAULT_PARTITION_N_HVG,
    seed: int = DEFAULT_SEED,
    compute_umap: bool = True,
) -> dict[str, Any]:
    _require_batch_key(adata, batch_key)
    if adata.n_obs < 3:
        raise ValueError("run_standard_bcmp_preprocess requires at least 3 cells")

    raw_counts = as_csr_matrix(adata.X)
    _validate_nonnegative_finite_counts(raw_counts)
    var_names = _var_names(adata)
    log_norm = normalize_log1p_counts(raw_counts)
    hvg_genes = select_highly_variable_genes(
        raw_counts=raw_counts,
        obs=adata.obs,
        var_names=var_names,
        n_top_genes=int(partition_n_hvg),
        batch_key=batch_key,
    )

    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
    hvg_idx = np.array(
        [gene_to_idx[gene] for gene in hvg_genes if gene in gene_to_idx], dtype=int
    )
    if len(hvg_idx) < int(partition_n_pcs):
        raise ValueError(
            "Requested "
            f"partition_n_pcs={int(partition_n_pcs)} but only {len(hvg_idx)} HVGs are available "
            "after feature selection"
        )

    hvg_matrix = log_norm[:, hvg_idx]
    del log_norm
    hvg_scaled = seurat_style_scale_dense(hvg_matrix)
    del hvg_matrix

    feasible_pcs = min(hvg_scaled.shape[0] - 1, hvg_scaled.shape[1])
    if feasible_pcs < int(partition_n_pcs):
        raise ValueError(
            "Requested "
            f"partition_n_pcs={int(partition_n_pcs)} but only {feasible_pcs} PCs are feasible for the current object"
        )

    pca = PCA(n_components=int(partition_n_pcs), svd_solver="full", copy=False)
    pca_embeddings = np.asarray(pca.fit_transform(hvg_scaled), dtype=float)
    del hvg_scaled
    umap_embeddings = None
    if compute_umap:
        umap_embeddings = run_uncorrected_umap2d(
            pca_embeddings=pca_embeddings,
            dims_use=list(range(min(DEFAULT_PARTITION_N_PCS, int(partition_n_pcs)))),
            seed=int(seed),
        )

    return {
        "hvg_genes": hvg_genes,
        "pca_embeddings": pca_embeddings,
        "umap_embeddings": umap_embeddings,
    }
