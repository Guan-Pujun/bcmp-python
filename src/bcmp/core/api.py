"""Public embedding-facing API for BCMP core."""

from __future__ import annotations

from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd

from bcmp.constants import (
    DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    DEFAULT_SEED,
    OutputLevel,
)
from bcmp.models import EmbeddingDebugResult, EmbeddingResult
from bcmp.utils.validation import (
    coerce_non_missing_string_index,
    require_int32_cell_count,
    require_unique_values,
    validate_output_level,
    validate_max_underrepresentation_fold,
    validate_search_integer_parameters,
    validate_fraction,
)

from .search import run_bcmp_partition_search

__all__ = ["bcmp_embedding"]


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
    level = validate_output_level(output_level)
    validate_search_integer_parameters(k_min, k_max, seed)
    batch_frac_threshold_for_k_max = validate_fraction(
        batch_frac_threshold_for_k_max, name="batch_frac_threshold_for_k_max"
    )
    max_residual_cell_frac = validate_fraction(
        max_residual_cell_frac, name="max_residual_cell_frac"
    )
    max_underrepresentation_fold = validate_max_underrepresentation_fold(
        max_underrepresentation_fold
    )
    embedding_shape = np.shape(embedding)
    if len(embedding_shape) != 2:
        raise ValueError("embedding must be a 2D array")
    if embedding_shape[1] < 1:
        raise ValueError("embedding must contain at least one dimension")
    require_int32_cell_count(int(embedding_shape[0]), context="BCMP embedding")
    embedding_arr = np.asarray(embedding)
    if embedding_arr.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if not np.issubdtype(embedding_arr.dtype, np.number) or np.issubdtype(
        embedding_arr.dtype, np.complexfloating
    ):
        raise ValueError("embedding must contain real numeric values")
    if not bool(np.isfinite(embedding_arr).all()):
        raise ValueError("embedding must contain only finite values")

    batch_index = coerce_non_missing_string_index(batch_labels, name="batch_labels")
    batch_series = pd.Series(batch_index.astype(str), dtype="string")
    if len(batch_series) != embedding_arr.shape[0]:
        raise ValueError("batch_labels length must match embedding row count")

    if cell_ids is None:
        names = pd.Index(
            [str(i) for i in range(embedding_arr.shape[0])], name="cell_id"
        )
    else:
        names = coerce_non_missing_string_index(cell_ids, name="cell_ids")
        names = pd.Index(names.astype(str), name="cell_id")
        if len(names) != embedding_arr.shape[0]:
            raise ValueError("cell_ids length must match embedding row count")
    names = require_unique_values(names, name="cell_ids")

    return run_bcmp_partition_search(
        pca_embeddings=embedding_arr,
        batch_labels=batch_series,
        cell_ids=names,
        min_batch_coverage=min_batch_coverage,
        min_cells_in_domain_per_batch=min_cells_in_domain_per_batch,
        max_underrepresentation_fold=max_underrepresentation_fold,
        max_residual_cell_frac=max_residual_cell_frac,
        k_min=k_min,
        k_max=k_max,
        batch_frac_threshold_for_k_max=batch_frac_threshold_for_k_max,
        seed=seed,
        progress_logger=progress_logger,
        output_level=level,
        verbose=verbose,
    )
