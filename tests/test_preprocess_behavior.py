"""Deterministic preprocessing behavior checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import bcmp.adata.preprocess as preprocess


def test_normalization_and_scaling_follow_expected_sparse_math() -> None:
    counts = sp.csr_matrix([[1, 0, 3], [0, 0, 0], [2, 4, 0]], dtype=float)
    normalized = preprocess.normalize_log1p_counts(counts, target_sum=100.0)
    scaled = preprocess.seurat_style_scale_dense(normalized, scale_max=0.5)

    np.testing.assert_allclose(
        normalized.toarray()[0], [np.log1p(25), 0.0, np.log1p(75)]
    )
    np.testing.assert_array_equal(normalized.toarray()[1], [0.0, 0.0, 0.0])
    assert scaled.shape == (3, 3)
    assert np.isfinite(scaled).all()
    assert float(scaled.max()) <= 0.5


def test_consensus_and_duplicate_predictor_stabilization_are_deterministic() -> None:
    consensus = preprocess.seurat_v5_consensus_features(
        {"counts.a": ["g2", "g1", "g3"], "counts.b": ["g1", "g2", "g3"]},
        ["g1", "g2", "g3"],
        nfeatures=2,
    )
    predictor = preprocess._stabilize_duplicate_loess_predictor(
        np.array([0.0, 0.0, 1.0, 2.0, 2.0, 3.0])
    )

    assert consensus == ["g1", "g2"]
    assert predictor[2] == 1.0
    assert predictor[5] == 3.0
    assert np.all(np.diff(np.sort(predictor)) > 0)


def test_vst_variance_kernel_matches_a_direct_reference() -> None:
    counts = sp.csc_matrix([[0.0, 1.0], [3.0, 0.0], [1.0, 2.0]], dtype=float)
    means = np.asarray(counts.mean(axis=0)).ravel()
    sd = np.array([1.0, 1.0])
    observed = preprocess._variance_standardized_seurat_v5_vst_from_csc.py_func(
        counts.indptr, counts.data, means, sd, 3, 2.0
    )

    expected = []
    for column in counts.toarray().T:
        z = np.minimum((column - means[len(expected)]) / sd[len(expected)], 2.0)
        expected.append(float(np.dot(z, z) / 2.0))
    np.testing.assert_allclose(observed, expected)


def test_hvg_selection_combines_each_batch_in_first_seen_order(monkeypatch) -> None:
    calls: list[int] = []

    def select_per_batch(*, raw_counts, var_names, n_top_genes):
        calls.append(raw_counts.shape[0])
        return ["g1", "g0"]

    monkeypatch.setattr(
        preprocess,
        "_select_highly_variable_genes_seurat_v5_vst_dgcmatrix",
        select_per_batch,
    )
    selected = preprocess.select_highly_variable_genes(
        sp.csr_matrix(np.arange(18).reshape(6, 3)),
        pd.DataFrame({"batch": ["a", "a", "a", "b", "b", "b"]}),
        pd.Index(["g0", "g1", "g2"]),
        n_top_genes=2,
    )

    assert calls == [3, 3]
    assert selected == ["g1", "g0"]


def test_real_hvg_selection_and_diagnostic_umap_return_expected_shapes() -> None:
    rng = np.random.default_rng(236)
    counts = sp.csr_matrix(rng.poisson(3, size=(20, 40)))
    genes = pd.Index([f"g{index}" for index in range(counts.shape[1])])

    selected = preprocess._select_highly_variable_genes_seurat_v5_vst_dgcmatrix(
        counts, genes, n_top_genes=5
    )
    coordinates = preprocess.run_uncorrected_umap2d(rng.normal(size=(20, 4)), seed=236)

    assert len(selected) == 5
    assert set(selected).issubset(set(genes))
    assert coordinates is not None
    assert coordinates.shape == (20, 2)


def test_diagnostic_umap_skips_a_one_dimension_reduction() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        coordinates = preprocess.run_uncorrected_umap2d(np.array([[0.0], [1.0], [2.0]]))

    assert coordinates is None
    assert "fewer than 2 valid dims" in str(captured[0].message)


def test_consensus_handles_valid_empty_feature_overlap() -> None:
    assert preprocess.seurat_v5_consensus_features({"counts.a": ["g1"]}, []) == []
    assert preprocess.seurat_v5_consensus_features({"counts.a": ["g1"]}, ["g2"]) == []
    np.testing.assert_array_equal(
        preprocess._stabilize_duplicate_loess_predictor(np.array([1.0])),
        np.array([1.0]),
    )


def test_diagnostic_umap_skips_when_fewer_than_two_dimensions_are_selected() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        coordinates = preprocess.run_uncorrected_umap2d(
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), dims_use=[0]
        )

    assert coordinates is None
    assert "fewer than 2 valid dims" in str(captured[0].message)


def test_consensus_handles_an_empty_layer_collection() -> None:
    assert preprocess.seurat_v5_consensus_features({}, ["g1"]) == []
