"""Runtime warning helpers."""

from __future__ import annotations

import warnings

_ALGORITHM_NAME = "BCMP"

__all__ = ["format_partition_runtime_notice", "emit_partition_warning"]


def format_partition_runtime_notice(kind: str, step: str, details: str) -> str:
    return f"[{_ALGORITHM_NAME} {kind}][{step}] {details}"


def emit_partition_warning(step: str, details: str) -> None:
    warnings.warn(
        format_partition_runtime_notice("warning", step, details),
        RuntimeWarning,
        stacklevel=2,
    )
