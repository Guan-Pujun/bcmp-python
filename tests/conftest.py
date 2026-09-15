"""Shared test runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = _ROOT / ".runtime-cache"


def _configure_runtime_cache(variable: str, directory: str) -> None:
    """Provide a writable test cache unless the shell already configured one."""
    cache_dir = Path(os.environ.setdefault(variable, str(_CACHE_ROOT / directory)))
    cache_dir.mkdir(parents=True, exist_ok=True)


_configure_runtime_cache("NUMBA_CACHE_DIR", "numba")
_configure_runtime_cache("MPLCONFIGDIR", "matplotlib")


@pytest.fixture
def assert_golden_labels():
    """Compare cell-aligned golden labels with strictly less than 1% drift."""

    def check(observed: pd.Series, expected: pd.Series) -> None:
        pd.testing.assert_index_equal(observed.index, expected.index, check_names=False)
        assert len(expected) > 0, "Golden labels must contain cells"
        assert observed.notna().all() and expected.notna().all(), (
            "Labels must be complete"
        )
        mismatches = int(observed.ne(expected).sum())
        total = len(expected)
        assert mismatches * 100 < total, (
            f"Golden label mismatch: {mismatches}/{total} cells "
            f"({mismatches / total:.4%}); required < 1%"
        )

    return check
