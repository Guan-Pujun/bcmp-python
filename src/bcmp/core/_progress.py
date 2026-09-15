"""Progress-line helpers for BCMP core orchestration."""

from __future__ import annotations

import logging
import sys
from typing import Any

__all__ = [
    "_emit_progress_notice",
    "_close_progress_logger",
    "_emit_progress_line",
    "_format_k_search_notice",
    "_make_progress_logger",
]

ProgressLogger = logging.Logger | logging.LoggerAdapter


def _format_k_search_notice(record: dict[str, Any]) -> str:
    return (
        f"[iter {int(record['iter_id'])}][{record['phase']}] k={int(record['k'])} | "
        f"mixing_status={record['mixing_status']} | n_domains={int(record['n_domains'])} | "
        f"n_failed_domains={int(record['n_failed_domains'])} | "
        f"n_residual_domains={int(record['n_residual_domains'])} | "
        f"actual_residual_cell_frac={float(record['actual_residual_cell_frac']):.4f}"
    )


def _make_progress_logger(
    *,
    verbose: bool,
    progress_logger: ProgressLogger | None = None,
) -> tuple[ProgressLogger | None, bool]:
    if not verbose:
        return None, False
    if progress_logger is not None:
        return progress_logger, False
    logger = logging.Logger("bcmp.progress", level=logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, True


def _close_progress_logger(
    progress_logger: ProgressLogger | None, *, owned: bool
) -> None:
    if progress_logger is None or not owned:
        return
    if not isinstance(progress_logger, logging.Logger):
        return
    for handler in progress_logger.handlers[:]:
        handler.flush()
        handler.close()
        progress_logger.removeHandler(handler)


def _emit_progress_notice(
    *,
    progress_logger: ProgressLogger | None,
    message: str,
) -> None:
    if progress_logger is None:
        return
    progress_logger.info(str(message))


def _emit_progress_line(
    *,
    progress_logger: ProgressLogger | None,
    record: dict[str, Any],
) -> None:
    if progress_logger is None:
        return
    progress_logger.info(_format_k_search_notice(record=record))
