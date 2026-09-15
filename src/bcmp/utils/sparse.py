"""Sparse matrix conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

__all__ = ["as_csr_matrix"]


def as_csr_matrix(x: Any) -> sp.csr_matrix:
    if sp.isspmatrix_csr(x):
        return x
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))
