"""Deterministic core-algorithm behavior checks."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

from bcmp.core.cluster import _group_singletons, cluster_labels_from_edge_matrix
from bcmp.core.knn import get_knn_neighbor_once
from bcmp.core.mixing import (
    evaluate_bcmp_batch_mixing_labels,
    prepare_bcmp_batch_reference,
    required_batch_coverage,
)
from bcmp.core.snn import (
    _build_knn_postings_exact,
    _count_snn_upper_row_nnz_exact,
    _fill_snn_upper_edge_matrix_exact,
    snn_upper_edge_matrix_from_knn_ranked,
)
from bcmp.utils.sparse import as_csr_matrix


def test_knn_returns_ranked_neighbors_for_a_small_embedding() -> None:
    embedding = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    neighbors = get_knn_neighbor_once(embedding, k_max=3, seed=236)

    assert neighbors.shape == (4, 3)
    assert neighbors.dtype == np.int32
    for cell_id, row in enumerate(neighbors):
        assert row[0] == cell_id
        assert len(set(row)) == 3


def test_batch_reference_and_mixing_summary_are_deterministic() -> None:
    reference = prepare_bcmp_batch_reference(["b", "a", "b", "c"])
    summary = evaluate_bcmp_batch_mixing_labels(
        domain_labels=["x", "x", "x", "y", "y", "y"],
        batch_labels=["b1", "b1", "b2", "b2", "b3", "b3"],
        min_cells_in_domain_per_batch=1,
        max_underrepresentation_fold=100,
    )

    assert reference["batch_levels"] == ["a", "b", "c"]
    np.testing.assert_array_equal(reference["batch_counts"], [1, 2, 1])
    assert required_batch_coverage(5) == 3
    assert summary["mixing_status"] == "pass"
    assert summary["n_evaluated_domains"] == 2
    assert summary["per_domain"]["domain_pass"].tolist() == ["pass", "pass"]


def test_mixing_summary_keeps_small_tail_as_a_residual_domain() -> None:
    summary = evaluate_bcmp_batch_mixing_labels(
        domain_labels=["large"] * 11 + ["tail"],
        batch_labels=["b1"] * 6 + ["b2"] * 6,
        min_cells_in_domain_per_batch=5,
        max_underrepresentation_fold=100,
        max_residual_cell_frac=0.1,
    )

    assert summary["mixing_status"] == "pass"
    assert summary["n_residual_domains"] == 1
    assert summary["n_residual_cells"] == 1
    assert (
        summary["per_domain"].set_index("domain").loc["tail", "domain_pass"]
        == "residual"
    )


def test_snn_exact_helpers_and_public_wrapper_agree() -> None:
    ranked_neighbors = np.array([[0, 1], [1, 0], [2, 0]], dtype=np.int32)
    postings, posting_rows = _build_knn_postings_exact.py_func(ranked_neighbors)
    row_counts = _count_snn_upper_row_nnz_exact.py_func(
        ranked_neighbors, postings, posting_rows, 1
    )
    offsets = np.empty(ranked_neighbors.shape[0] + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(row_counts, out=offsets[1:])
    direct_edges = _fill_snn_upper_edge_matrix_exact.py_func(
        ranked_neighbors, postings, posting_rows, 1, offsets
    )
    expected = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)

    np.testing.assert_array_equal(direct_edges, expected)
    np.testing.assert_array_equal(
        snn_upper_edge_matrix_from_knn_ranked(ranked_neighbors), expected
    )


def test_clustering_and_singleton_grouping_preserve_graph_structure() -> None:
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    labels = cluster_labels_from_edge_matrix(n_cells=4, edges=edges, seed=236)
    grouped = _group_singletons(
        np.array([0, 1, 1]),
        sp.csr_matrix([[1.0, 0.9, 0.8], [0.9, 1.0, 0.6], [0.8, 0.6, 1.0]]),
        seed=236,
    )

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    np.testing.assert_array_equal(grouped, np.array([0, 0, 0]))


def test_mixing_rejects_invalid_user_configuration() -> None:
    with pytest.raises(ValueError, match="at least two batches"):
        required_batch_coverage(1)
    with pytest.raises(ValueError, match="finite numeric fraction"):
        required_batch_coverage(3, default_batch_coverage_fraction=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        required_batch_coverage(3, min_batch_coverage=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        required_batch_coverage(3, min_batch_coverage=4)
    with pytest.raises(ValueError, match="at least one batch label"):
        prepare_bcmp_batch_reference([])
    with pytest.raises(ValueError, match="at least two batches"):
        prepare_bcmp_batch_reference(["one", "one"])


def test_mixing_rejects_invalid_evaluation_inputs() -> None:
    valid_domains = ["a", "a", "b", "b"]
    valid_batches = ["x", "x", "y", "y"]
    with pytest.raises(ValueError, match="Either batch_labels or batch_reference"):
        evaluate_bcmp_batch_mixing_labels(valid_domains)
    with pytest.raises(ValueError, match="length must match"):
        evaluate_bcmp_batch_mixing_labels(valid_domains, valid_batches + ["x"])
    with pytest.raises(ValueError, match="Too few cells"):
        evaluate_bcmp_batch_mixing_labels(["a", "b"], ["x", "y"])
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_bcmp_batch_mixing_labels(
            valid_domains, valid_batches, min_cells_in_domain_per_batch=0
        )
    with pytest.raises(ValueError, match="finite numeric fraction"):
        evaluate_bcmp_batch_mixing_labels(
            valid_domains, valid_batches, max_residual_cell_frac=1.0
        )


def test_knn_rejects_too_small_or_invalid_search_inputs() -> None:
    with pytest.raises(ValueError, match="At least 3 cells"):
        get_knn_neighbor_once(np.zeros((2, 2)), k_max=1)
    with pytest.raises(ValueError, match="k_max must be at least 1"):
        get_knn_neighbor_once(np.zeros((3, 2)), k_max=0)


def test_sparse_conversion_preserves_values() -> None:
    matrix = np.array([[0.0, 1.0], [2.0, 0.0]])

    converted = as_csr_matrix(matrix)

    assert sp.isspmatrix_csr(converted)
    np.testing.assert_array_equal(converted.toarray(), matrix)


def test_edge_matrix_builds_a_symmetric_sparse_graph() -> None:
    from bcmp.core.cluster import _csr_from_edge_matrix

    graph = _csr_from_edge_matrix(3, np.array([[0, 1], [1, 2]], dtype=np.int32))
    unchanged = _group_singletons(
        np.array([0, 0, 1, 1]), sp.identity(4, format="csr"), seed=236
    )

    np.testing.assert_array_equal(
        graph.toarray(),
        np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    np.testing.assert_array_equal(unchanged, np.array([0, 0, 1, 1]))


def test_mixing_uses_an_explicit_required_batch_coverage() -> None:
    reference = prepare_bcmp_batch_reference(["a", "a", "b", "b", "c", "c"])
    summary = evaluate_bcmp_batch_mixing_labels(
        domain_labels=["one", "one", "one", "two", "two", "two"],
        batch_reference=reference,
        min_batch_coverage=2,
        min_cells_in_domain_per_batch=1,
        max_underrepresentation_fold=100,
    )

    assert summary["required_batch_coverage"] == 2
    assert summary["required_batch_coverage_source"] == "explicit"
    assert summary["mixing_status"] == "pass"


def test_empty_graph_helpers_preserve_isolated_cells() -> None:
    from bcmp.core.cluster import _contiguous_chunk_bounds, _csr_from_edge_matrix

    graph = _csr_from_edge_matrix(3, np.empty((0, 2), dtype=np.int32))

    assert _contiguous_chunk_bounds(0, 2) == []
    np.testing.assert_array_equal(graph.toarray(), np.eye(3))


def test_clustering_reports_a_fully_disconnected_graph() -> None:
    with pytest.raises(ValueError, match="All clusters are singletons"):
        cluster_labels_from_edge_matrix(
            n_cells=3,
            edges=np.empty((0, 2), dtype=np.int32),
            seed=236,
        )


def test_sparse_conversion_keeps_existing_csr_matrices() -> None:
    matrix = sp.identity(3, format="csr")

    assert as_csr_matrix(matrix) is matrix


def test_sparse_conversion_accepts_csc_matrices() -> None:
    matrix = sp.csc_matrix([[0.0, 1.0], [2.0, 0.0]])

    converted = as_csr_matrix(matrix)

    assert sp.isspmatrix_csr(converted)
    np.testing.assert_array_equal(converted.toarray(), matrix.toarray())
