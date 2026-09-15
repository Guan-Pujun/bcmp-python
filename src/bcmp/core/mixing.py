"""Batch-mixing helpers for the core layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bcmp.constants import (
    DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    DEFAULT_MIN_BATCH_COVERAGE_FRAC,
    DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
)
from bcmp.utils.validation import (
    coerce_non_missing_string_index,
    validate_max_underrepresentation_fold,
)

__all__ = [
    "DEFAULT_MIN_BATCH_COVERAGE_FRAC",
    "DEFAULT_MAX_RESIDUAL_CELL_FRAC",
    "required_batch_coverage",
    "prepare_bcmp_batch_reference",
    "evaluate_bcmp_batch_mixing_labels",
]


def _validate_default_batch_coverage_fraction(
    default_batch_coverage_fraction: float,
) -> float:
    value = float(default_batch_coverage_fraction)
    if not np.isfinite(value) or pd.isna(value) or value <= 0.0 or value > 1.0:
        raise ValueError(
            "default_batch_coverage_fraction must be a finite numeric fraction in (0, 1]"
        )
    return value


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value_int = int(value)
    if value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _validate_max_residual_cell_frac(max_residual_cell_frac: float) -> float:
    value = float(max_residual_cell_frac)
    if not np.isfinite(value) or pd.isna(value) or value < 0.0 or value >= 1.0:
        raise ValueError(
            "max_residual_cell_frac must be a finite numeric fraction in [0, 1)"
        )
    return value


def _validate_explicit_min_batch_coverage(
    min_batch_coverage: int, n_batches: int
) -> int:
    if isinstance(min_batch_coverage, bool) or not isinstance(
        min_batch_coverage, (int, np.integer)
    ):
        raise ValueError("min_batch_coverage must be an integer batch count or None")
    value = int(min_batch_coverage)
    if value < 2:
        raise ValueError(
            "min_batch_coverage must be at least 2 when supplied as a batch count"
        )
    if value > int(n_batches):
        raise ValueError("min_batch_coverage cannot exceed the number of batches")
    return value


def required_batch_coverage(
    n_batches: int,
    min_batch_coverage: int | None = None,
    *,
    default_batch_coverage_fraction: float = DEFAULT_MIN_BATCH_COVERAGE_FRAC,
) -> int:
    n_batches = int(n_batches)
    if n_batches < 2:
        raise ValueError("Batch mixing requires at least two batches")
    if min_batch_coverage is not None:
        return _validate_explicit_min_batch_coverage(
            min_batch_coverage, n_batches=n_batches
        )
    default_frac = _validate_default_batch_coverage_fraction(
        default_batch_coverage_fraction
    )
    return int(max(2, np.ceil(float(n_batches) * default_frac)))


def prepare_bcmp_batch_reference(
    batch_labels: list[str] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    labels = coerce_non_missing_string_index(batch_labels, name="batch_labels")
    if len(labels) < 1:
        raise ValueError(
            "prepare_bcmp_batch_reference requires at least one batch label"
        )
    batch_levels = pd.Index(sorted(labels.unique().tolist()))
    if len(batch_levels) < 2:
        raise ValueError("Batch mixing requires at least two batches")
    batch_idx = batch_levels.get_indexer(labels)
    if np.any(batch_idx < 0):
        raise ValueError("Failed to encode batch labels")
    batch_counts = np.bincount(batch_idx, minlength=len(batch_levels)).astype(int)
    return {
        "batch_levels": batch_levels.tolist(),
        "batch_idx": batch_idx.astype(int),
        "batch_counts": batch_counts,
        "p_global": batch_counts.astype(float) / max(1, len(labels)),
        "S": int(len(batch_levels)),
        "n": int(len(labels)),
    }


def _resolve_required_batch_coverage(
    batch_reference: dict[str, Any],
    min_batch_coverage: int | None,
) -> tuple[int, str]:
    if min_batch_coverage is not None:
        return (
            _validate_explicit_min_batch_coverage(
                min_batch_coverage,
                n_batches=int(batch_reference["S"]),
            ),
            "explicit",
        )
    return required_batch_coverage(int(batch_reference["S"])), "default_fraction"


def _domain_first_seen_positions(
    domain_arr: np.ndarray, domain_levels: pd.Index
) -> np.ndarray:
    first_seen: dict[str, int] = {}
    for pos, label in enumerate(domain_arr.astype(str)):
        first_seen.setdefault(str(label), int(pos))
    return np.asarray(
        [first_seen[str(label)] for label in domain_levels.astype(str)], dtype=int
    )


def _domain_residual_selection(
    *,
    domain_sizes: np.ndarray,
    domain_first_seen_pos: np.ndarray,
    max_residual_cell_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual_frac = _validate_max_residual_cell_frac(max_residual_cell_frac)
    n_cells = int(np.asarray(domain_sizes, dtype=int).sum())
    if n_cells < 1:
        raise ValueError("Need at least one cell for residual domain selection")

    order = np.lexsort((domain_first_seen_pos.astype(int), -domain_sizes.astype(int)))
    target_evaluated_cells = int(np.ceil((1.0 - residual_frac) * float(n_cells)))
    target_evaluated_cells = min(max(target_evaluated_cells, 1), n_cells)

    evaluated = np.zeros(domain_sizes.shape[0], dtype=bool)
    evaluated_cells = 0
    for domain_pos in order:
        if evaluated_cells >= target_evaluated_cells:
            break
        evaluated[int(domain_pos)] = True
        evaluated_cells += int(domain_sizes[int(domain_pos)])
    return evaluated, order.astype(int)


def evaluate_bcmp_batch_mixing_labels(
    domain_labels: list[str] | np.ndarray | pd.Series,
    batch_labels: list[str] | np.ndarray | pd.Series | None = None,
    batch_reference: dict[str, Any] | None = None,
    min_batch_coverage: int | None = None,
    min_cells_in_domain_per_batch: int = DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    max_underrepresentation_fold: float
    | np.integer
    | np.floating = DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    max_residual_cell_frac: float = DEFAULT_MAX_RESIDUAL_CELL_FRAC,
) -> dict[str, Any]:
    domain_arr = coerce_non_missing_string_index(
        domain_labels, name="domain_labels"
    ).to_numpy()
    if batch_reference is None:
        if batch_labels is None:
            raise ValueError("Either batch_labels or batch_reference must be supplied")
        batch_reference = prepare_bcmp_batch_reference(batch_labels)

    if len(domain_arr) != int(batch_reference["n"]):
        raise ValueError(
            "domain_labels length must match the encoded batch label length"
        )
    if len(domain_arr) < 3:
        raise ValueError("Too few cells remaining for batch-mixing evaluation")

    domain_levels = pd.Index(pd.unique(domain_arr).tolist())
    domain_idx = domain_levels.get_indexer(domain_arr)
    if np.any(domain_idx < 0):
        raise ValueError("Failed to encode domain labels")

    c = int(len(domain_levels))
    s = int(batch_reference["S"])
    if c < 1 or s < 1:
        raise ValueError(
            "Need at least one domain and one batch for batch-mixing evaluation"
        )

    required_coverage, required_coverage_source = _resolve_required_batch_coverage(
        batch_reference=batch_reference,
        min_batch_coverage=min_batch_coverage,
    )
    min_effective_cells = _validate_positive_integer(
        min_cells_in_domain_per_batch,
        name="min_cells_in_domain_per_batch",
    )
    underrepresentation_fold = validate_max_underrepresentation_fold(
        max_underrepresentation_fold
    )
    n_cells = int(len(domain_arr))
    linear_idx = domain_idx + batch_reference["batch_idx"] * c
    counts_vec = np.bincount(linear_idx, minlength=c * s)
    count_mat = counts_vec.reshape(s, c).T.astype(int)
    domain_sizes = count_mat.sum(axis=1).astype(int)
    domain_size_frac = domain_sizes.astype(float) / float(n_cells)
    domain_first_seen_pos = _domain_first_seen_positions(domain_arr, domain_levels)
    domain_is_evaluated, domain_order = _domain_residual_selection(
        domain_sizes=domain_sizes,
        domain_first_seen_pos=domain_first_seen_pos,
        max_residual_cell_frac=max_residual_cell_frac,
    )
    n_residual_cells = int(domain_sizes[~domain_is_evaluated].sum())
    residual_cell_frac = float(n_residual_cells) / float(n_cells)
    n_evaluated_cells = int(domain_sizes[domain_is_evaluated].sum())
    evaluated_cell_frac = float(n_evaluated_cells) / float(n_cells)

    batch_prop_threshold = np.asarray(batch_reference["p_global"], dtype=float) / float(
        underrepresentation_fold
    )
    threshold_counts = (
        domain_sizes[:, None].astype(float) * batch_prop_threshold[None, :]
    )
    effective_mat = (count_mat >= int(min_effective_cells)) & (
        count_mat >= threshold_counts
    )
    n_effective = effective_mat.sum(axis=1).astype(int)
    pass_domain_bool = n_effective >= int(required_coverage)
    evaluated_fail = domain_is_evaluated & (~pass_domain_bool)
    n_evaluated_domains = int(domain_is_evaluated.sum())
    fail_domain_count = int(evaluated_fail.sum())
    n_residual_domains = int((~domain_is_evaluated).sum())
    if fail_domain_count == 0:
        mixing_status = "pass"
        mixing_failure_reason = "none"
        search_pass = True
    else:
        mixing_status = "fail"
        mixing_failure_reason = "domain_mixing_fail"
        search_pass = False
    domain_pass = np.where(
        domain_is_evaluated,
        np.where(pass_domain_bool, "pass", "fail"),
        "residual",
    )

    per_domain = (
        pd.DataFrame(
            {
                "domain": domain_levels.astype(str),
                "domain_size": domain_sizes,
                "domain_size_frac": domain_size_frac,
                "domain_first_seen_pos": domain_first_seen_pos,
                "effective_batch_count": n_effective,
                "domain_pass": domain_pass,
            }
        )
        .iloc[domain_order]
        .reset_index(drop=True)
    )
    return {
        "pass": bool(search_pass),
        "mixing_status": str(mixing_status),
        "fail_domain_count": fail_domain_count,
        "n_evaluated_domains": n_evaluated_domains,
        "n_residual_domains": n_residual_domains,
        "n_residual_cells": n_residual_cells,
        "evaluated_cell_frac": evaluated_cell_frac,
        "residual_cell_frac": residual_cell_frac,
        "mixing_failure_reason": mixing_failure_reason,
        "S": int(s),
        "required_batch_coverage": int(required_coverage),
        "required_batch_coverage_source": str(required_coverage_source),
        "default_batch_coverage_fraction": float(DEFAULT_MIN_BATCH_COVERAGE_FRAC),
        "max_residual_cell_frac": float(
            _validate_max_residual_cell_frac(max_residual_cell_frac)
        ),
        "per_domain": per_domain,
    }
