"""AnnData result assembly for the public BCMP workflow."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from bcmp.constants import (
    BCMP_PCA_EMBEDDING_KEY,
    BCMP_UMAP_EMBEDDING_KEY,
    CANONICAL_LABEL_COLUMN,
)
from bcmp.models import (
    AnnDataDebugResult,
    AnnDataResult,
    BCMPAnnDataDebug,
    EmbeddingDebugResult,
    EmbeddingResult,
)

__all__ = ["write_bcmp_result_to_adata"]


def _embed_active_into_full(
    all_cells: pd.Index,
    include_cells: pd.Index,
    active_matrix: np.ndarray,
) -> np.ndarray:
    if active_matrix.ndim != 2:
        raise ValueError("active_matrix must be two-dimensional")
    if active_matrix.shape[0] != len(include_cells):
        raise ValueError("active_matrix row count must match include_cells")
    full = np.full((len(all_cells), active_matrix.shape[1]), np.nan, dtype=float)
    take = all_cells.get_indexer(include_cells)
    if bool(np.any(take < 0)):
        raise ValueError("include_cells must be contained in all_cells")
    full[take, :] = np.asarray(active_matrix, dtype=float)
    return full


def write_bcmp_result_to_adata(
    *,
    adata_full: ad.AnnData,
    all_cells: pd.Index,
    active_cells: pd.Index,
    full_labels: pd.Series,
    preprocess: dict[str, Any],
    write_debug: bool,
    core: EmbeddingResult | EmbeddingDebugResult,
    debug_parameters: dict[str, Any] | None,
    workflow_debug: dict[str, Any] | None,
) -> AnnDataResult | AnnDataDebugResult:
    labels = pd.Series(full_labels, index=all_cells, dtype="string")
    if bool(labels.isna().any()):
        raise ValueError("BCMP result labels do not cover all cells")

    for key in [
        "bcmp",
        *[key for key in adata_full.uns if str(key).startswith("bcmp_")],
    ]:
        adata_full.uns.pop(key, None)
    for key in (BCMP_PCA_EMBEDDING_KEY, BCMP_UMAP_EMBEDDING_KEY):
        if key in adata_full.obsm:
            del adata_full.obsm[key]
    adata_full.obs[CANONICAL_LABEL_COLUMN] = pd.Categorical(labels.astype(str))

    if write_debug:
        if (
            not isinstance(core, EmbeddingDebugResult)
            or debug_parameters is None
            or workflow_debug is None
        ):
            raise RuntimeError(
                "debug output requires complete core and workflow diagnostics"
            )
        adata_full.obsm[BCMP_PCA_EMBEDDING_KEY] = _embed_active_into_full(
            all_cells=all_cells,
            include_cells=active_cells,
            active_matrix=np.asarray(preprocess["pca_embeddings"]),
        )
        if preprocess["umap_embeddings"] is not None:
            adata_full.obsm[BCMP_UMAP_EMBEDDING_KEY] = _embed_active_into_full(
                all_cells=all_cells,
                include_cells=active_cells,
                active_matrix=np.asarray(preprocess["umap_embeddings"]),
            )
        debug = BCMPAnnDataDebug(
            parameters=debug_parameters,
            search=core.debug.search,
            candidates=core.debug.candidates,
            workflow=workflow_debug,
        )
        return AnnDataDebugResult(
            selection=core.selection,
            search_trace=core.search_trace,
            debug=debug,
            adata=adata_full,
        )

    return AnnDataResult(
        selection=core.selection,
        search_trace=core.search_trace,
        adata=adata_full,
    )
