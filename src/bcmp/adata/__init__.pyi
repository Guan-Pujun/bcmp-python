from __future__ import annotations

from logging import Logger

import anndata as ad
import numpy as np
import pandas as pd

from bcmp.constants import OutputLevel
from bcmp.models import AnnDataDebugResult, AnnDataResult

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
