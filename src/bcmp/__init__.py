"""Top-level BCMP public API."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING

from .constants import (
    DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
    DEFAULT_BATCH_KEY,
    DEFAULT_K_MIN,
    DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    DEFAULT_PARTITION_N_HVG,
    DEFAULT_PARTITION_N_PCS,
    DEFAULT_SEED,
    OutputLevel,
)
from .settings import cache_info, set_cache_dir

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import pandas as pd

    from .models import (
        AnnDataDebugResult,
        AnnDataResult,
        EmbeddingDebugResult,
        EmbeddingResult,
    )

__all__ = ["bcmp_embedding", "bcmp", "set_cache_dir", "cache_info"]


def bcmp_embedding(
    embedding: np.ndarray,
    batch_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    cell_ids: Sequence[str] | pd.Index | None = None,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = None,
    batch_frac_threshold_for_k_max: float = DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
    min_batch_coverage: int | None = None,
    min_cells_in_domain_per_batch: int = DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    max_underrepresentation_fold: float
    | np.integer
    | np.floating = DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    max_residual_cell_frac: float = DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    seed: int = DEFAULT_SEED,
    progress_logger: logging.Logger | None = None,
    output_level: OutputLevel = "standard",
    verbose: bool = True,
) -> EmbeddingResult | EmbeddingDebugResult:
    """Run BCMP from an unintegrated embedding matrix and batch labels."""
    from .core import bcmp_embedding as _bcmp_embedding

    return _bcmp_embedding(
        embedding=embedding,
        batch_labels=batch_labels,
        cell_ids=cell_ids,
        k_min=k_min,
        k_max=k_max,
        batch_frac_threshold_for_k_max=batch_frac_threshold_for_k_max,
        min_batch_coverage=min_batch_coverage,
        min_cells_in_domain_per_batch=min_cells_in_domain_per_batch,
        max_underrepresentation_fold=max_underrepresentation_fold,
        max_residual_cell_frac=max_residual_cell_frac,
        seed=seed,
        progress_logger=progress_logger,
        output_level=output_level,
        verbose=verbose,
    )


def bcmp(
    adata: ad.AnnData,
    *,
    batch_key: str = DEFAULT_BATCH_KEY,
    partition_n_pcs: int = DEFAULT_PARTITION_N_PCS,
    partition_n_hvg: int = DEFAULT_PARTITION_N_HVG,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = None,
    batch_frac_threshold_for_k_max: float = DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
    min_batch_coverage: int | None = None,
    min_cells_in_domain_per_batch: int = DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    max_underrepresentation_fold: float
    | np.integer
    | np.floating = DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    max_residual_cell_frac: float = DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    exclude_cells: list[str]
    | tuple[str, ...]
    | pd.Index
    | pd.Series
    | np.ndarray
    | None = None,
    seed: int = DEFAULT_SEED,
    progress_logger: logging.Logger | None = None,
    output_level: OutputLevel = "standard",
    verbose: bool = True,
) -> AnnDataResult | AnnDataDebugResult:
    """Run BCMP from a raw-count AnnData object."""
    from .adata import bcmp as _bcmp

    return _bcmp(
        adata=adata,
        batch_key=batch_key,
        partition_n_pcs=partition_n_pcs,
        partition_n_hvg=partition_n_hvg,
        k_min=k_min,
        k_max=k_max,
        batch_frac_threshold_for_k_max=batch_frac_threshold_for_k_max,
        min_batch_coverage=min_batch_coverage,
        min_cells_in_domain_per_batch=min_cells_in_domain_per_batch,
        max_underrepresentation_fold=max_underrepresentation_fold,
        max_residual_cell_frac=max_residual_cell_frac,
        exclude_cells=exclude_cells,
        seed=seed,
        progress_logger=progress_logger,
        output_level=output_level,
        verbose=verbose,
    )
