"""BCMP core layer.

This package hosts algorithmic logic that remains independent from AnnData,
scanpy, scikit-learn, and umap-learn.
"""

from __future__ import annotations

from typing import Any

__all__ = ["bcmp_embedding"]


def bcmp_embedding(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the embedding API to keep core package import lightweight."""
    from .api import bcmp_embedding as _bcmp_embedding

    return _bcmp_embedding(*args, **kwargs)
