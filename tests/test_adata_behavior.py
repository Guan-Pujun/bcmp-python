"""AnnData workflow helper behavior checks."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

from bcmp import bcmp, bcmp_embedding
from bcmp.constants import INT32_INDEX_MAX
from bcmp.utils.validation import require_int32_cell_count

from bcmp.adata._glue import (
    compose_full_partition_labels,
    prepare_active_adata,
    rebuild_counts_only_adata,
    resolve_excluded_cells,
)


def _example_adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.arange(18).reshape(6, 3),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "a", "b", "b", "b"]}, index=[f"c{i}" for i in range(6)]
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )


def test_counts_only_rebuild_preserves_the_analysis_axes() -> None:
    rebuilt = rebuild_counts_only_adata(_example_adata())

    assert rebuilt.shape == (6, 3)
    assert rebuilt.obs_names.tolist() == [f"c{i}" for i in range(6)]
    assert rebuilt.var_names.tolist() == ["g0", "g1", "g2"]
    assert rebuilt.obs["batch"].tolist() == ["a", "a", "a", "b", "b", "b"]


def test_excluded_cells_are_kept_with_batch_specific_labels() -> None:
    adata = _example_adata()
    active, exclusion = prepare_active_adata(
        adata, exclude_cells=["c1", "c4"], min_cells_after_exclusion=3
    )
    labels = compose_full_partition_labels(
        exclusion["all_cells"],
        pd.Series(
            ["0", "1", "0", "1"], index=exclusion["include_cells"], dtype="string"
        ),
        exclusion["excluded_labels"],
    )

    assert active.obs_names.tolist() == ["c0", "c2", "c3", "c5"]
    assert exclusion["excluded_labels"].to_dict() == {
        "c1": "Excluded_a",
        "c4": "Excluded_b",
    }
    assert labels.to_dict() == {
        "c0": "0",
        "c1": "Excluded_a",
        "c2": "1",
        "c3": "0",
        "c4": "Excluded_b",
        "c5": "1",
    }
    assert resolve_excluded_cells(adata, exclude_cells=None)["exclude_cells"].empty


def test_exclusion_resolves_supported_cell_id_containers() -> None:
    import anndata as ad
    import pandas as pd

    from bcmp.adata._glue import resolve_excluded_cells

    adata = ad.AnnData(
        X=sp.identity(4, format="csr"),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "b", "b"]}, index=["c0", "c1", "c2", "c3"]
        ),
    )
    boolean_mask = pd.Series([False, True, False, False], index=adata.obs_names)
    inputs = [
        ("c1",),
        pd.Index(["c1"]),
        pd.Series(["c1"]),
        np.array(["c1"]),
        boolean_mask,
    ]

    for value in inputs:
        resolved = resolve_excluded_cells(adata, exclude_cells=value)
        assert resolved["exclude_cells"].tolist() == ["c1"]
        assert resolved["excluded_labels"].tolist() == ["Excluded_a"]


def test_embedding_requires_complete_batch_labels() -> None:
    embedding = np.arange(6, dtype=float).reshape(-1, 1)

    with pytest.raises(
        ValueError, match="batch_labels must not contain missing values"
    ):
        bcmp_embedding(embedding, ["a", "a", None, "b", "b", "b"])
    with pytest.raises(ValueError, match="batch_labels must not contain empty values"):
        bcmp_embedding(embedding, ["a", "a", " ", "b", "b", "b"])


def test_embedding_requires_unique_cell_ids_and_a_public_output_level() -> None:
    embedding = np.arange(6, dtype=float).reshape(-1, 1)
    batches = ["a", "a", "a", "b", "b", "b"]

    with pytest.raises(ValueError, match="cell_ids must contain unique values"):
        bcmp_embedding(
            embedding, batches, cell_ids=["c0", "c1", "c1", "c3", "c4", "c5"]
        )
    with pytest.raises(
        ValueError, match="output_level must be one of: standard, debug"
    ):
        bcmp_embedding(embedding, batches, output_level="full")
    with pytest.raises(
        ValueError, match="cell_ids length must match embedding row count"
    ):
        bcmp_embedding(embedding, batches, cell_ids=["c0", "c1"])


def test_anndata_requires_unique_observation_and_feature_ids() -> None:
    adata = ad.AnnData(
        X=np.ones((6, 3)),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "a", "b", "b", "b"]},
            index=["c0", "c1", "c1", "c3", "c4", "c5"],
        ),
        var=pd.DataFrame(index=["g0", "g1", "g1"]),
    )

    with pytest.raises(ValueError, match="adata.obs_names must be unique"):
        rebuild_counts_only_adata(adata)

    adata.obs_names = ["c0", "c1", "c2", "c3", "c4", "c5"]
    with pytest.raises(ValueError, match="adata.var_names must be unique"):
        rebuild_counts_only_adata(adata)


def test_cell_count_limit_is_checked_before_large_allocations() -> None:
    with pytest.raises(ValueError, match="requires n_cells <="):
        require_int32_cell_count(INT32_INDEX_MAX + 1, context="BCMP test")


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        (np.arange(6, dtype=float), "embedding must be a 2D array"),
        (
            np.array([[0.0], [1.0], [np.nan], [3.0], [4.0], [5.0]]),
            "finite values",
        ),
        (
            np.array([["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]),
            "real numeric",
        ),
        (np.ones((6, 1), dtype=np.complex128), "real numeric"),
    ],
)
def test_embedding_requires_a_finite_numeric_matrix(
    embedding: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bcmp_embedding(embedding, ["a", "a", "a", "b", "b", "b"])


def test_embedding_requires_one_batch_label_per_cell() -> None:
    embedding = np.arange(12, dtype=float).reshape(6, 2)

    with pytest.raises(
        ValueError, match="batch_labels length must match embedding row count"
    ):
        bcmp_embedding(embedding, ["a", "a", "a", "b", "b"])


def test_embedding_rejects_invalid_search_bounds() -> None:
    embedding = np.arange(12, dtype=float).reshape(6, 2)
    batches = ["a", "a", "a", "b", "b", "b"]

    with pytest.raises(ValueError, match="Invalid k range"):
        bcmp_embedding(embedding, batches, k_min=3, k_max=2)
    with pytest.raises(ValueError, match="finite numeric fraction"):
        bcmp_embedding(embedding, batches, batch_frac_threshold_for_k_max=1.0)


@pytest.mark.parametrize(
    "parameter, value",
    [
        (name, value)
        for name in ("k_min", "k_max", "seed")
        for value in (3.9, 2.0, np.float64(2), True, np.bool_(False), "3", None, [3])
        if not (name == "k_max" and value is None)
    ],
)
def test_public_entries_reject_noninteger_search_parameters(parameter, value) -> None:
    adata = _example_adata()
    kwargs = {parameter: value, "verbose": False}
    with pytest.raises(ValueError, match=f"{parameter} must be an integer"):
        bcmp(adata, **kwargs)
    with pytest.raises(ValueError, match=f"{parameter} must be an integer"):
        bcmp_embedding(adata.X, adata.obs["batch"], **kwargs)


def test_embedding_uses_public_default_cell_ids() -> None:
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
        ]
    )

    result = bcmp_embedding(
        embedding,
        ["a", "a", "a", "b", "b", "b"],
        k_min=3,
        k_max=3,
        verbose=False,
    )

    assert result.labels.index.tolist() == ["0", "1", "2", "3", "4", "5"]


@pytest.mark.parametrize("parameter", ["k_min", "k_max"])
@pytest.mark.parametrize("value", [0, -1])
def test_public_entries_require_positive_k_bounds(parameter, value) -> None:
    adata = _example_adata()
    with pytest.raises(ValueError, match=f"{parameter} must be at least 1"):
        bcmp(adata, **{parameter: value}, verbose=False)
    with pytest.raises(ValueError, match=f"{parameter} must be at least 1"):
        bcmp_embedding(adata.X, adata.obs["batch"], **{parameter: value}, verbose=False)


@pytest.mark.parametrize(
    "parameter", ["batch_frac_threshold_for_k_max", "max_residual_cell_frac"]
)
@pytest.mark.parametrize(
    "value",
    [
        "0.01",
        "0.05",
        False,
        np.bool_(True),
        None,
        [0.1],
        np.array(0.1),
        0.1 + 0j,
        np.nan,
        np.inf,
        -np.inf,
        -0.01,
        1.0,
    ],
)
def test_public_entries_require_real_fractions(parameter, value) -> None:
    adata = _example_adata()
    # An explicit upper bound must not bypass fraction validation.
    kwargs = {parameter: value, "k_max": 3, "verbose": False}
    with pytest.raises(
        ValueError, match=f"{parameter} must be a finite numeric fraction"
    ):
        bcmp(adata, **kwargs)
    with pytest.raises(
        ValueError, match=f"{parameter} must be a finite numeric fraction"
    ):
        bcmp_embedding(adata.X, adata.obs["batch"], **kwargs)


@pytest.mark.parametrize(
    "fraction", [0, np.int64(0), np.float32(0.5), np.nextafter(1.0, 0.0)]
)
def test_fraction_boundaries_and_effective_k_bounds(fraction) -> None:
    from bcmp.utils.validation import validate_fraction
    from bcmp.core.search import derive_k_search_bounds

    for name in ("batch_frac_threshold_for_k_max", "max_residual_cell_frac"):
        assert validate_fraction(fraction, name=name) == float(fraction)
    bounds = derive_k_search_bounds(
        ["a"] * 3 + ["b"] * 3,
        n_total=6,
        k_min=1,
        k_max=100,
        batch_frac_threshold_for_k_max=fraction,
    )
    assert bounds["k_low"] == 2
    assert bounds["k_high"] == 5


@pytest.mark.parametrize(
    "value",
    [
        [True, False],
        (True, False),
        np.array([True, False]),
        ["c1", True],
        ("c1", np.bool_(False)),
        np.array(["c1", True], dtype=object),
        pd.Index(["c1", True]),
        pd.Series(["c1", True]),
    ],
)
def test_exclusions_reject_boolean_id_sequences(value) -> None:
    adata = _example_adata()
    with pytest.raises(ValueError, match="boolean masks must be a boolean Series"):
        bcmp(adata, exclude_cells=value, verbose=False)


def test_exclusions_preserve_empty_and_boolean_named_ids() -> None:
    adata = _example_adata()
    adata.obs_names = ["True", "False", "c2", "c3", "c4", "c5"]
    assert resolve_excluded_cells(adata, exclude_cells=[])["exclude_cells"].empty
    result = resolve_excluded_cells(adata, exclude_cells=["True", "False"])
    assert result["exclude_cells"].tolist() == ["True", "False"]


def test_exclusions_reject_unknown_cells_and_misaligned_boolean_masks() -> None:
    adata = _example_adata()

    with pytest.raises(ValueError, match="cells not present"):
        resolve_excluded_cells(adata, exclude_cells=["not-a-cell"])
    for index in (
        pd.Index([f"other-{i}" for i in range(adata.n_obs)]),
        adata.obs_names[::-1],
        adata.obs_names[:-1],
    ):
        mask = pd.Series(False, index=index)
        with pytest.raises(ValueError, match="index must match adata.obs_names"):
            resolve_excluded_cells(adata, exclude_cells=mask)


def test_exclusions_reject_missing_boolean_values_and_unsupported_containers() -> None:
    adata = _example_adata()

    with pytest.raises(ValueError, match="must not contain missing values"):
        resolve_excluded_cells(
            adata,
            exclude_cells=pd.Series(
                [False, pd.NA, False, False, False, False],
                index=adata.obs_names,
                dtype="boolean",
            ),
        )
    with pytest.raises(TypeError, match="exclude_cells must be one of"):
        resolve_excluded_cells(adata, exclude_cells={"c0"})


def test_anndata_requires_batch_labels_and_preserves_user_metadata() -> None:
    from bcmp.datasets import diabetic_kidney_lite

    missing_batch = ad.AnnData(
        X=np.ones((100, 3)),
        obs=pd.DataFrame(index=[f"c{index}" for index in range(100)]),
    )
    with pytest.raises(KeyError, match="Missing batch column"):
        bcmp(missing_batch, verbose=False)

    adata = diabetic_kidney_lite()
    adata.obs["user_group"] = "kept"
    adata.var["user_feature"] = "kept"
    adata.layers["user_counts"] = adata.X.copy()
    adata.obsm["X_user"] = np.zeros((adata.n_obs, 2))
    adata.uns["user_metadata"] = {"source": "kept"}
    adata.uns["bcmp"] = {"old": True}
    adata.uns["bcmp_previous"] = {"old": True}
    adata.obsm["X_bcmp_pca"] = np.full((adata.n_obs, 2), -1.0)
    adata.obsm["X_bcmp_umap"] = np.full((adata.n_obs, 2), -1.0)

    result = bcmp(adata, verbose=False)

    assert "bcmp_domain" not in adata.obs
    assert result.adata.obs["user_group"].eq("kept").all()
    assert result.adata.var["user_feature"].eq("kept").all()
    assert "user_counts" in result.adata.layers
    assert "X_user" in result.adata.obsm
    assert result.adata.uns["user_metadata"] == {"source": "kept"}
    assert "bcmp" not in result.adata.uns
    assert "bcmp_previous" not in result.adata.uns
    assert "X_bcmp_pca" not in result.adata.obsm
    assert "X_bcmp_umap" not in result.adata.obsm
    assert "bcmp" in adata.uns
    assert "X_bcmp_pca" in adata.obsm


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1.0, "non-negative count values"),
        (np.nan, "finite count values"),
    ],
)
def test_anndata_requires_finite_non_negative_raw_counts(
    value: float, message: str
) -> None:
    counts = np.ones((100, 3))
    counts[0, 0] = value
    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"batch": ["a"] * 50 + ["b"] * 50},
            index=[f"c{index}" for index in range(100)],
        ),
    )

    with pytest.raises(ValueError, match=message):
        bcmp(adata, verbose=False)


@pytest.mark.parametrize("parameter", ["partition_n_hvg", "partition_n_pcs"])
@pytest.mark.parametrize(
    "value",
    [16.9, 8.9, 8.0, True, np.bool_(True), "8", None, np.nan, np.inf, [8], np.array(8)],
)
def test_anndata_rejects_noninteger_partition_counts(parameter, value) -> None:
    adata = _example_adata()
    with pytest.raises(ValueError, match=f"{parameter} must be an integer"):
        bcmp(adata, **{parameter: value}, verbose=False)


def test_anndata_rejects_a_nonpositive_hvg_count() -> None:
    adata = ad.AnnData(
        X=np.ones((100, 3)),
        obs=pd.DataFrame(
            {"batch": ["a"] * 50 + ["b"] * 50},
            index=[f"c{index}" for index in range(100)],
        ),
    )

    with pytest.raises(ValueError, match="n_top_genes must be positive"):
        bcmp(adata, partition_n_hvg=0, verbose=False)


def test_anndata_rejects_more_pcs_than_selected_hvgs() -> None:
    rng = np.random.default_rng(236)
    adata = ad.AnnData(
        X=rng.poisson(3, size=(100, 30)),
        obs=pd.DataFrame(
            {"batch": ["a"] * 50 + ["b"] * 50},
            index=[f"c{index}" for index in range(100)],
        ),
        var=pd.DataFrame(index=[f"g{index}" for index in range(30)]),
    )

    with pytest.raises(ValueError, match="only 3 HVGs are available"):
        bcmp(
            adata,
            partition_n_hvg=3,
            partition_n_pcs=4,
            verbose=False,
        )


def test_anndata_requires_at_least_three_cells_per_batch_for_hvg_selection() -> None:
    adata = ad.AnnData(
        X=sp.csr_matrix(np.ones((100, 3))),
        obs=pd.DataFrame(
            {"batch": ["a"] * 98 + ["b"] * 2},
            index=[f"c{index}" for index in range(100)],
        ),
    )

    with pytest.raises(ValueError, match="Each batch must contain at least 3 cells"):
        bcmp(adata, verbose=False)


def test_anndata_requires_enough_active_cells_after_exclusion() -> None:
    adata = ad.AnnData(
        X=sp.csr_matrix(np.ones((100, 3))),
        obs=pd.DataFrame(
            {"batch": ["a"] * 50 + ["b"] * 50},
            index=[f"c{index}" for index in range(100)],
        ),
    )

    with pytest.raises(ValueError, match="at least 100 active cells"):
        bcmp(adata, exclude_cells=["c0"], verbose=False)
