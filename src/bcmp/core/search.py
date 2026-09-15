"""K-search orchestration for BCMP core."""

from __future__ import annotations

import logging
from typing import Any

import numba
import numpy as np
import pandas as pd

from bcmp.constants import (
    DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    DEFAULT_SEED,
    OutputLevel,
)
from bcmp.models import (
    BCMPDebug,
    BCMPDebugCandidate,
    BCMPSelection,
    EmbeddingDebugResult,
    EmbeddingResult,
)
from bcmp.utils.validation import coerce_non_missing_string_index

from ._progress import (
    _close_progress_logger,
    _emit_progress_line,
    _emit_progress_notice,
    _make_progress_logger,
)
from .cluster import cluster_labels_from_edge_matrix
from .knn import get_knn_neighbor_once
from .mixing import evaluate_bcmp_batch_mixing_labels, prepare_bcmp_batch_reference
from .snn import snn_upper_edge_matrix_from_knn_ranked

__all__ = ["derive_k_search_bounds", "run_bcmp_partition_search"]


def derive_k_search_bounds(
    batch_labels: pd.Series | list[str] | np.ndarray,
    n_total: int,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = None,
    batch_frac_threshold_for_k_max: float = DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
) -> dict[str, int]:
    batch_series = pd.Series(
        coerce_non_missing_string_index(batch_labels, name="batch_labels"),
        dtype="string",
    )
    batch_sizes = batch_series.value_counts(sort=False)
    if batch_sizes.empty:
        raise ValueError("Cannot derive k bounds: no non-missing batch counts found")
    batch_frac_threshold_for_k_max = float(batch_frac_threshold_for_k_max)
    if (
        not np.isfinite(batch_frac_threshold_for_k_max)
        or pd.isna(batch_frac_threshold_for_k_max)
        or batch_frac_threshold_for_k_max < 0.0
        or batch_frac_threshold_for_k_max >= 1.0
    ):
        raise ValueError(
            "batch_frac_threshold_for_k_max must be a finite numeric fraction in [0, 1)"
        )
    if k_max is None:
        eligible_batch_sizes = batch_sizes[
            batch_sizes > (float(n_total) * batch_frac_threshold_for_k_max)
        ]
        if eligible_batch_sizes.empty:
            raise ValueError(
                f"Cannot derive k bounds: no batches exceed {100.0 * batch_frac_threshold_for_k_max:.2f}% of total cells"
            )
        min_batch_n = max(int(eligible_batch_sizes.min()), 1)
        k_high = min(int(min_batch_n - 1), int(n_total - 1))
    else:
        k_high = min(int(k_max), int(n_total - 1))
    k_low = max(int(k_min), 2)
    return {
        "S": int(batch_series.nunique()),
        "k_low": int(k_low),
        "k_high": int(k_high),
    }


def labels_at_k(
    nn_idx_max: np.ndarray,
    k: int,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    nn_idx_k = nn_idx_max[:, : int(k)]
    edges = snn_upper_edge_matrix_from_knn_ranked(nn_idx_k)
    labels = cluster_labels_from_edge_matrix(
        n_cells=int(nn_idx_k.shape[0]),
        edges=edges,
        seed=seed,
    )
    return labels


def _candidate_from_eval(
    k: int, labels: np.ndarray, mix: dict[str, Any]
) -> dict[str, Any]:
    return {
        "k": int(k),
        "pass": bool(mix["pass"]),
        "n_domains": int(mix["per_domain"].shape[0]),
        "labels": labels,
        "mix": mix,
    }


def run_bcmp_partition_search(
    pca_embeddings: np.ndarray,
    batch_labels: pd.Series,
    cell_ids: pd.Index,
    min_batch_coverage: int | None = None,
    min_cells_in_domain_per_batch: int = DEFAULT_MIN_EFFECTIVE_CELLS_IN_DOMAIN_PER_BATCH,
    max_underrepresentation_fold: float
    | np.integer
    | np.floating = DEFAULT_MAX_UNDERREPRESENTATION_FOLD,
    max_residual_cell_frac: float = DEFAULT_MAX_RESIDUAL_CELL_FRAC,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = None,
    batch_frac_threshold_for_k_max: float = DEFAULT_BATCH_FRAC_THRESHOLD_FOR_K_MAX,
    seed: int = DEFAULT_SEED,
    *,
    output_level: OutputLevel,
    progress_logger: logging.Logger | None = None,
    verbose: bool = True,
) -> EmbeddingResult | EmbeddingDebugResult:
    write_debug = output_level == "debug"
    if pca_embeddings.ndim != 2:
        raise ValueError("pca_embeddings must be a 2D array")
    if pca_embeddings.shape[0] != len(cell_ids):
        raise ValueError("pca_embeddings row count must match cell_ids")
    if pca_embeddings.shape[0] != len(batch_labels):
        raise ValueError("batch_labels length must match pca_embeddings row count")

    kb = derive_k_search_bounds(
        batch_labels=batch_labels,
        n_total=pca_embeddings.shape[0],
        k_min=k_min,
        k_max=k_max,
        batch_frac_threshold_for_k_max=batch_frac_threshold_for_k_max,
    )
    k_low = int(kb["k_low"])
    k_high = int(kb["k_high"])
    if k_low > k_high:
        raise ValueError("Invalid k range")

    mix_batch_reference = prepare_bcmp_batch_reference(batch_labels)
    nn_idx_max = get_knn_neighbor_once(pca_embeddings, k_max=k_high, seed=seed)
    trace_rows: list[dict[str, Any]] = []
    debug_candidates: dict[int, BCMPDebugCandidate] = {}
    resolved_progress_logger, owns_progress_logger = _make_progress_logger(
        verbose=verbose,
        progress_logger=progress_logger,
    )
    _emit_progress_notice(
        progress_logger=resolved_progress_logger,
        message=f"[runtime] numba_threads={int(numba.get_num_threads())}",
    )

    try:

        def eval_mix(k: int) -> tuple[np.ndarray, dict[str, Any]]:
            labels_num = labels_at_k(
                nn_idx_max=nn_idx_max,
                k=int(k),
                seed=seed,
            )
            label_strings = np.asarray([str(x) for x in labels_num], dtype=object)
            mix = evaluate_bcmp_batch_mixing_labels(
                domain_labels=label_strings,
                batch_reference=mix_batch_reference,
                min_batch_coverage=min_batch_coverage,
                min_cells_in_domain_per_batch=min_cells_in_domain_per_batch,
                max_underrepresentation_fold=max_underrepresentation_fold,
                max_residual_cell_frac=max_residual_cell_frac,
            )
            return label_strings, mix

        def record_trace(
            iter_id: int,
            k: int,
            labels: np.ndarray,
            mix: dict[str, Any],
            phase: str,
        ) -> None:
            record = {
                "iter_id": int(iter_id),
                "phase": str(phase),
                "k": int(k),
                "mixing_status": str(mix["mixing_status"]),
                "n_domains": int(mix["per_domain"].shape[0]),
                "n_failed_domains": int(mix["fail_domain_count"]),
                "n_residual_domains": int(mix["n_residual_domains"]),
                "actual_residual_cell_frac": float(mix["residual_cell_frac"]),
            }
            trace_rows.append(record)
            if write_debug:
                per_domain = mix["per_domain"].copy()
                per_domain = per_domain[
                    [
                        "domain",
                        "domain_size",
                        "domain_size_frac",
                        "effective_batch_count",
                        "domain_pass",
                    ]
                ].rename(
                    columns={
                        "domain_size_frac": "domain_cell_frac",
                        "domain_pass": "domain_mixing_status",
                    }
                )
                debug_candidates[int(k)] = BCMPDebugCandidate(
                    partition=pd.Series(
                        labels, index=cell_ids.astype(str), dtype="string"
                    ),
                    domain_summary=per_domain,
                )
            _emit_progress_line(
                progress_logger=resolved_progress_logger,
                record=record,
            )

        iter_id = 1
        k_search_status: str | None = None

        lo_labels, lo_mix = eval_mix(k_low)
        record_trace(
            iter_id=iter_id, k=k_low, labels=lo_labels, mix=lo_mix, phase="k_low"
        )
        lo = _candidate_from_eval(k_low, lo_labels, lo_mix)

        if k_low == k_high:
            chosen = lo
            if lo["pass"]:
                k_search_status = "k_low_pass"
            else:
                k_search_status = "diagnostic_no_feasible_k"
        elif lo["pass"]:
            chosen = lo
            k_search_status = "k_low_pass"
        else:
            iter_id += 1
            hi_labels, hi_mix = eval_mix(k_high)
            record_trace(
                iter_id=iter_id, k=k_high, labels=hi_labels, mix=hi_mix, phase="k_high"
            )
            hi = _candidate_from_eval(k_high, hi_labels, hi_mix)

            if not hi["pass"]:
                chosen = hi
                k_search_status = "diagnostic_no_feasible_k"
            else:
                best_pass_candidate = hi
                low = k_low
                high = k_high
                while high - low > 1:
                    mid = int(np.floor((low + high) / 2))
                    iter_id += 1
                    mid_labels, mid_mix = eval_mix(mid)
                    record_trace(
                        iter_id=iter_id,
                        k=mid,
                        labels=mid_labels,
                        mix=mid_mix,
                        phase="k_mid",
                    )
                    mid_candidate = _candidate_from_eval(mid, mid_labels, mid_mix)
                    if mid_candidate["pass"]:
                        best_pass_candidate = mid_candidate
                        high = mid
                    else:
                        low = mid
                chosen = best_pass_candidate
                k_search_status = "binary_search_best_pass"

        selected_record = next(
            record for record in trace_rows if record["k"] == int(chosen["k"])
        )
        _emit_progress_notice(
            progress_logger=resolved_progress_logger,
            message=(
                f"[selected] k={int(selected_record['k'])} | "
                f"mixing_status={selected_record['mixing_status']} | "
                f"n_domains={int(selected_record['n_domains'])} | "
                f"n_failed_domains={int(selected_record['n_failed_domains'])} | "
                f"n_residual_domains={int(selected_record['n_residual_domains'])} | "
                f"actual_residual_cell_frac={float(selected_record['actual_residual_cell_frac']):.4f}"
            ),
        )

        trace_df = pd.DataFrame(
            trace_rows,
            columns=[
                "iter_id",
                "phase",
                "k",
                "mixing_status",
                "n_domains",
                "n_failed_domains",
                "n_residual_domains",
                "actual_residual_cell_frac",
            ],
        )
        labels_series = pd.Series(
            chosen["labels"], index=cell_ids.astype(str), dtype="string"
        )
        selection = BCMPSelection(
            selected_k=int(chosen["k"]),
            n_domains=int(chosen["n_domains"]),
            mixing_status=str(chosen["mix"]["mixing_status"]),
            n_evaluated_domains=int(chosen["mix"]["n_evaluated_domains"]),
            n_residual_domains=int(chosen["mix"]["n_residual_domains"]),
            n_residual_cells=int(chosen["mix"]["n_residual_cells"]),
            residual_cell_frac=float(chosen["mix"]["residual_cell_frac"]),
        )
        if write_debug:
            debug = BCMPDebug(
                parameters={
                    "k_min": int(k_min),
                    "k_max": None if k_max is None else int(k_max),
                    "batch_frac_threshold_for_k_max": float(
                        batch_frac_threshold_for_k_max
                    ),
                    "min_batch_coverage": None
                    if min_batch_coverage is None
                    else int(min_batch_coverage),
                    "min_cells_in_domain_per_batch": int(min_cells_in_domain_per_batch),
                    "max_underrepresentation_fold": float(max_underrepresentation_fold),
                    "max_residual_cell_frac": float(max_residual_cell_frac),
                    "seed": int(seed),
                },
                search={
                    "k_low": int(k_low),
                    "k_high": int(k_high),
                    "status": str(k_search_status),
                    "required_batch_coverage": int(
                        chosen["mix"]["required_batch_coverage"]
                    ),
                    "required_batch_coverage_source": str(
                        chosen["mix"]["required_batch_coverage_source"]
                    ),
                },
                candidates=debug_candidates,
            )
            return EmbeddingDebugResult(
                selection=selection,
                labels=labels_series,
                search_trace=trace_df,
                debug=debug,
            )
        return EmbeddingResult(
            selection=selection,
            labels=labels_series,
            search_trace=trace_df,
        )
    finally:
        _close_progress_logger(resolved_progress_logger, owned=owns_progress_logger)
