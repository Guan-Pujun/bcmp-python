"""AnnData-facing BCMP public wrapper."""

from __future__ import annotations

from typing import Any

__all__ = ["bcmp"]


def bcmp(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the AnnData API to avoid hard import-time dependency leaks."""
    from .api import bcmp as _bcmp

    return _bcmp(*args, **kwargs)
