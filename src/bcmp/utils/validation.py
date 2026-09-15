"""Shared validation helpers for lightweight core and AnnData boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd

from bcmp.constants import INT32_INDEX_MAX, OUTPUT_LEVEL_VALUES, OutputLevel

type StringIndexInput = pd.Index | pd.Series | np.ndarray | Sequence[object]

__all__ = [
    "StringIndexInput",
    "coerce_non_missing_string_index",
    "require_int32_cell_count",
    "require_unique_index",
    "require_unique_values",
    "validate_output_level",
    "validate_max_underrepresentation_fold",
    "validate_search_integer_parameters",
    "validate_fraction",
]


def validate_search_integer_parameters(
    k_min: int, k_max: int | None, seed: int
) -> None:
    """Validate positive integer k bounds and an integer seed."""
    for name, value in (("k_min", k_min), ("k_max", k_max), ("seed", seed)):
        if name == "k_max" and value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
        if name != "seed" and value < 1:
            raise ValueError(f"{name} must be at least 1")


def validate_fraction(value: float, *, name: str) -> float:
    """Validate a real scalar in [0, 1) before converting to float."""
    message = f"{name} must be a finite numeric fraction in [0, 1)"
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or value < 0
        or not value < 1
    ):
        raise ValueError(message)
    fraction = float(value)
    if not 0 <= fraction < 1:
        raise ValueError(message)
    return fraction


def validate_max_underrepresentation_fold(
    value: float | np.integer | np.floating,
) -> float:
    """Validate a real fold threshold >= 1 and return it as a finite float."""
    message = "max_underrepresentation_fold must be a finite real scalar >= 1"
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or value < 1
        or not value < np.inf
    ):
        raise ValueError(message)
    range_message = "max_underrepresentation_fold must fit in a finite Python float"
    try:
        fold = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(range_message) from exc
    if not np.isfinite(fold):
        raise ValueError(range_message)
    return fold


def coerce_non_missing_string_index(values: StringIndexInput, *, name: str) -> pd.Index:
    series = pd.Series(values, dtype="string")
    if bool(series.isna().any()):
        raise ValueError(f"{name} must not contain missing values")
    text = series.astype(str)
    if bool(text.str.strip().eq("").any()):
        raise ValueError(f"{name} must not contain empty values")
    return pd.Index(text, dtype="object")


def require_int32_cell_count(n_cells: int, *, context: str) -> None:
    """Validate BCMP's int32 endpoint representation before large allocations."""
    if int(n_cells) > INT32_INDEX_MAX:
        raise ValueError(
            f"{context} currently requires n_cells <= {INT32_INDEX_MAX:,} "
            "because BCMP stores neighbor and graph endpoints as int32"
        )


def validate_output_level(output_level: str) -> OutputLevel:
    """Validate the requested public output level."""
    if output_level not in OUTPUT_LEVEL_VALUES:
        raise ValueError("output_level must be one of: standard, debug")
    return cast(OutputLevel, output_level)


def require_unique_index(
    values: pd.Index,
    *,
    name: str,
    duplicate_label: str = "values",
) -> pd.Index:
    if values.is_unique:
        return values
    duplicates = values[values.duplicated()].astype(str).unique().tolist()
    preview = ", ".join(duplicates[:5])
    suffix = "" if len(duplicates) <= 5 else ", ..."
    raise ValueError(
        f"{name} must be unique after string conversion; duplicate {duplicate_label}: {preview}{suffix}"
    )


def require_unique_values(
    values: pd.Index,
    *,
    name: str,
    duplicate_label: str = "values",
) -> pd.Index:
    if values.is_unique:
        return values
    duplicates = values[values.duplicated()].astype(str).unique().tolist()
    preview = ", ".join(duplicates[:5])
    suffix = "" if len(duplicates) <= 5 else ", ..."
    raise ValueError(
        f"{name} must contain unique values; duplicate {duplicate_label}: {preview}{suffix}"
    )
