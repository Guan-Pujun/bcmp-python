"""Lightweight AnnData glue helpers that avoid heavy preprocess imports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from bcmp.constants import DEFAULT_BATCH_KEY
from bcmp.utils.sparse import as_csr_matrix
from bcmp.utils.validation import (
    coerce_non_missing_string_index,
    require_int32_cell_count,
    require_unique_index,
)

__all__ = [
    "_obs_names",
    "_var_names",
    "_require_batch_key",
    "_require_obs_key_values",
    "rebuild_counts_only_adata",
    "resolve_excluded_cells",
    "prepare_active_adata",
    "compose_full_partition_labels",
]


def _obs_names(adata: ad.AnnData) -> pd.Index:
    names = pd.Index(adata.obs_names.astype(str), name="cell_id")
    return require_unique_index(
        names, name="adata.obs_names", duplicate_label="cell IDs"
    )


def _var_names(adata: ad.AnnData) -> pd.Index:
    names = coerce_non_missing_string_index(adata.var_names, name="adata.var_names")
    names = pd.Index(names.astype(str), name="gene_id")
    return require_unique_index(
        names, name="adata.var_names", duplicate_label="feature IDs"
    )


def _require_obs_key_values(adata: ad.AnnData, key: str, *, label: str) -> None:
    if key not in adata.obs.columns:
        raise KeyError(f"Missing {label} column in metadata: {key}")
    coerce_non_missing_string_index(adata.obs[key], name=label)


def _require_batch_key(adata: ad.AnnData, batch_key: str) -> None:
    _require_obs_key_values(adata, batch_key, label="batch")


def _coerce_excluded_cell_values(
    exclude_cells: list[str]
    | tuple[str, ...]
    | pd.Index
    | pd.Series
    | np.ndarray
    | None,
    *,
    all_cells: pd.Index,
) -> Sequence[object]:
    if exclude_cells is None:
        return []
    if isinstance(exclude_cells, pd.Series) and pd.api.types.is_bool_dtype(
        exclude_cells.dtype
    ):
        if bool(exclude_cells.isna().any()):
            raise ValueError(
                "exclude_cells boolean Series must not contain missing values"
            )
        series_index = pd.Index(exclude_cells.index.astype(str), name="cell_id")
        if not series_index.equals(all_cells):
            raise ValueError(
                "exclude_cells boolean Series index must match adata.obs_names"
            )
        mask = exclude_cells.to_numpy(dtype=bool, copy=False)
        return all_cells[mask].astype(str).tolist()
    if isinstance(exclude_cells, (list, tuple, pd.Index, pd.Series, np.ndarray)):
        values = (
            exclude_cells.flat
            if isinstance(exclude_cells, np.ndarray)
            else exclude_cells
        )
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(
                "exclude_cells boolean masks must be a boolean Series indexed like adata.obs_names; "
                "cell ID sequences must not contain boolean values"
            )
        return list(exclude_cells)
    raise TypeError(
        "exclude_cells must be one of: None, list[str], tuple[str, ...], pd.Index, pd.Series, np.ndarray"
    )


def rebuild_counts_only_adata(adata: ad.AnnData) -> ad.AnnData:
    out = ad.AnnData(
        X=as_csr_matrix(adata.X),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    out.obs_names = _obs_names(adata)
    out.var_names = _var_names(adata)
    return out


def resolve_excluded_cells(
    adata: ad.AnnData,
    batch_key: str = DEFAULT_BATCH_KEY,
    exclude_cells: list[str]
    | tuple[str, ...]
    | pd.Index
    | pd.Series
    | np.ndarray
    | None = None,
) -> dict[str, Any]:
    _require_batch_key(adata, batch_key)
    all_cells = _obs_names(adata)
    exclude_values = _coerce_excluded_cell_values(
        exclude_cells,
        all_cells=all_cells,
    )
    exclude_idx = coerce_non_missing_string_index(exclude_values, name="exclude_cells")
    all_cell_set = set(all_cells)
    unknown_cells = [
        cell for cell in pd.unique(exclude_idx) if cell not in all_cell_set
    ]
    if unknown_cells:
        preview = ", ".join([str(cell) for cell in unknown_cells[:5]])
        suffix = "" if len(unknown_cells) <= 5 else ", ..."
        raise ValueError(
            "exclude_cells contains cells not present in adata.obs_names; "
            f"{len(unknown_cells)} unknown cells: {preview}{suffix}"
        )
    exclude_set = {str(x) for x in exclude_idx}
    exclude_cells_idx = pd.Index(
        [cell for cell in all_cells if cell in exclude_set], name="cell_id"
    )
    include_cells_idx = all_cells.difference(exclude_cells_idx, sort=False)

    batch_vals = adata.obs.loc[all_cells, batch_key].astype(str)
    excluded_labels = pd.Series(dtype="string")
    if len(exclude_cells_idx) > 0:
        excluded_labels = pd.Series(
            data=[
                "Excluded_" + str(batch_vals.loc[cell]) for cell in exclude_cells_idx
            ],
            index=exclude_cells_idx,
            dtype="string",
        )

    return {
        "all_cells": all_cells,
        "include_cells": include_cells_idx,
        "exclude_cells": exclude_cells_idx,
        "excluded_labels": excluded_labels,
    }


def prepare_active_adata(
    adata: ad.AnnData,
    batch_key: str = DEFAULT_BATCH_KEY,
    exclude_cells: list[str]
    | tuple[str, ...]
    | pd.Index
    | pd.Series
    | np.ndarray
    | None = None,
    *,
    min_cells_after_exclusion: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    excluded_info = resolve_excluded_cells(
        adata, batch_key=batch_key, exclude_cells=exclude_cells
    )
    if len(excluded_info["exclude_cells"]) > 0:
        active_view = adata[excluded_info["include_cells"], :]
    else:
        active_view = adata
    if active_view.n_obs < int(min_cells_after_exclusion):
        raise ValueError(
            "Need at least "
            f"{int(min_cells_after_exclusion)} active cells after applying exclude_cells; got {active_view.n_obs}"
        )
    require_int32_cell_count(
        int(active_view.n_obs), context="BCMP AnnData active cells"
    )
    obj_active = rebuild_counts_only_adata(active_view)
    return obj_active, excluded_info


def compose_full_partition_labels(
    all_cells: pd.Index,
    active_labels: pd.Series,
    excluded_labels: pd.Series | None = None,
) -> pd.Series:
    out = pd.Series(index=all_cells, data=pd.NA, dtype="string")
    if active_labels is not None and len(active_labels) > 0:
        out.loc[active_labels.index.astype(str)] = active_labels.astype(
            "string"
        ).to_numpy()
    if excluded_labels is not None and len(excluded_labels) > 0:
        out.loc[excluded_labels.index.astype(str)] = excluded_labels.astype(
            "string"
        ).to_numpy()
    return out
