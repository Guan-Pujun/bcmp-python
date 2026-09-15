"""Runtime result models for BCMP public APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd


@dataclass(frozen=True, slots=True)
class BCMPSelection:
    """Final BCMP partition selected by the k-search."""

    selected_k: int
    n_domains: int
    mixing_status: str
    n_evaluated_domains: int
    n_residual_domains: int
    n_residual_cells: int
    residual_cell_frac: float


@dataclass(slots=True)
class BCMPDebugCandidate:
    """Detailed diagnostics for one evaluated representation scale."""

    partition: pd.Series
    domain_summary: pd.DataFrame


@dataclass(slots=True)
class BCMPDebug:
    """Debug information shared by both BCMP entry points."""

    parameters: dict[str, Any]
    search: dict[str, Any]
    candidates: dict[int, BCMPDebugCandidate]


@dataclass(slots=True)
class BCMPAnnDataDebug(BCMPDebug):
    """Debug information produced only by the AnnData workflow."""

    workflow: dict[str, Any]


@dataclass(slots=True)
class _Result:
    """Fields shared by standard BCMP results."""

    selection: BCMPSelection
    search_trace: pd.DataFrame


@dataclass(slots=True)
class EmbeddingResult(_Result):
    """Standard result returned by :func:`bcmp_embedding`."""

    labels: pd.Series


@dataclass(slots=True)
class EmbeddingDebugResult(EmbeddingResult):
    """Debug result returned by :func:`bcmp_embedding`."""

    debug: BCMPDebug


@dataclass(slots=True)
class AnnDataResult(_Result):
    """Standard result returned by :func:`bcmp`."""

    adata: ad.AnnData


@dataclass(slots=True)
class AnnDataDebugResult(AnnDataResult):
    """Debug result returned by :func:`bcmp`."""

    debug: BCMPAnnDataDebug
