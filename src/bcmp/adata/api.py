"""Public AnnData-facing API for BCMP."""

from __future__ import annotations

import importlib
import logging
from numbers import Integral
from typing import TYPE_CHECKING, Any

from bcmp.constants import (
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
from bcmp.utils.validation import (
    validate_output_level,
    validate_max_underrepresentation_fold,
    validate_search_integer_parameters,
    validate_fraction,
)

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import pandas as pd

    from bcmp.models import AnnDataDebugResult, AnnDataResult

__all__ = ["bcmp"]

_ACTIVE_ADATA_MIN_CELLS = 100


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
    """Run BCMP from raw counts and return a copied AnnData with proxy domains."""
    from ._glue import (
        _obs_names,
        _require_batch_key,
        compose_full_partition_labels,
        prepare_active_adata,
    )

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
    write_debug = level == "debug"

    for name, value in (
        ("partition_n_pcs", partition_n_pcs),
        ("partition_n_hvg", partition_n_hvg),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if partition_n_pcs < 1:
        raise ValueError("partition_n_pcs must be at least 1")

    _require_batch_key(adata, batch_key)

    adata_active, excluded_info = prepare_active_adata(
        adata,
        batch_key=batch_key,
        exclude_cells=exclude_cells,
        min_cells_after_exclusion=_ACTIVE_ADATA_MIN_CELLS,
    )

    core_api = importlib.import_module("bcmp.core.api")
    pipeline_module = importlib.import_module("bcmp.adata.pipeline")
    preprocess_module = importlib.import_module("bcmp.adata.preprocess")

    preprocess: dict[str, Any] | None
    preprocess = preprocess_module.run_standard_bcmp_preprocess(
        adata=adata_active,
        batch_key=batch_key,
        partition_n_pcs=int(partition_n_pcs),
        partition_n_hvg=int(partition_n_hvg),
        seed=int(seed),
        compute_umap=write_debug,
    )
    embedding = preprocess["pca_embeddings"]

    core = core_api.bcmp_embedding(
        embedding=embedding,
        batch_labels=adata_active.obs[batch_key],
        cell_ids=_obs_names(adata_active),
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

    debug_parameters = None
    workflow_debug = None
    if write_debug:
        debug_parameters = {
            "batch_key": str(batch_key),
            "partition_n_pcs": int(partition_n_pcs),
            "partition_n_hvg": int(partition_n_hvg),
            "k_min": int(k_min),
            "k_max": None if k_max is None else int(k_max),
            "batch_frac_threshold_for_k_max": float(batch_frac_threshold_for_k_max),
            "min_batch_coverage": None
            if min_batch_coverage is None
            else int(min_batch_coverage),
            "min_cells_in_domain_per_batch": int(min_cells_in_domain_per_batch),
            "max_underrepresentation_fold": float(max_underrepresentation_fold),
            "max_residual_cell_frac": float(max_residual_cell_frac),
            "exclude_cells": None if exclude_cells is None else list(exclude_cells),
            "seed": int(seed),
        }
        workflow_debug = {
            "hvg_genes": list(preprocess["hvg_genes"]),
            "excluded_cell_ids": excluded_info["exclude_cells"].astype(str).tolist(),
        }

    all_cells = _obs_names(adata)
    full_labels = compose_full_partition_labels(
        all_cells=all_cells,
        active_labels=core.labels,
        excluded_labels=excluded_info["excluded_labels"],
    )
    adata_full = adata.copy()
    adata_full.obs_names = all_cells

    return pipeline_module.write_bcmp_result_to_adata(
        adata_full=adata_full,
        all_cells=all_cells,
        active_cells=_obs_names(adata_active),
        full_labels=full_labels,
        preprocess=preprocess,
        write_debug=write_debug,
        core=core,
        debug_parameters=debug_parameters,
        workflow_debug=workflow_debug,
    )
