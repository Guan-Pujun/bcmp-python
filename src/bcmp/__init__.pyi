from __future__ import annotations

from collections.abc import Sequence
from logging import Logger
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .constants import OutputLevel
from .models import (
    AnnDataDebugResult,
    AnnDataResult,
    EmbeddingDebugResult,
    EmbeddingResult,
)

def bcmp_embedding(
    embedding: np.ndarray,
    batch_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    cell_ids: Sequence[str] | pd.Index | None = None,
    k_min: int = 3,
    k_max: int | None = None,
    batch_frac_threshold_for_k_max: float = 0.01,
    min_batch_coverage: int | None = None,
    min_cells_in_domain_per_batch: int = 5,
    max_underrepresentation_fold: float | np.integer | np.floating = 10,
    max_residual_cell_frac: float = 0.05,
    seed: int = 236,
    progress_logger: Logger | None = None,
    output_level: OutputLevel = "standard",
    verbose: bool = True,
) -> EmbeddingResult | EmbeddingDebugResult: ...
def bcmp(
    adata: ad.AnnData,
    *,
    batch_key: str = "batch",
    partition_n_pcs: int = 30,
    partition_n_hvg: int = 2000,
    k_min: int = 3,
    k_max: int | None = None,
    batch_frac_threshold_for_k_max: float = 0.01,
    min_batch_coverage: int | None = None,
    min_cells_in_domain_per_batch: int = 5,
    max_underrepresentation_fold: float | np.integer | np.floating = 10,
    max_residual_cell_frac: float = 0.05,
    exclude_cells: list[str]
    | tuple[str, ...]
    | pd.Index
    | pd.Series
    | np.ndarray
    | None = None,
    seed: int = 236,
    progress_logger: Logger | None = None,
    output_level: OutputLevel = "standard",
    verbose: bool = True,
) -> AnnDataResult | AnnDataDebugResult: ...
def set_cache_dir(cache_dir: str | Path, *, force: bool = False) -> dict[str, Any]: ...
def cache_info() -> dict[str, Any]: ...
