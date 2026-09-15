"""Real-valued fold thresholds across the public Python workflows."""

import json
from dataclasses import asdict
from fractions import Fraction

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from bcmp import bcmp, bcmp_embedding
from bcmp.core.mixing import evaluate_bcmp_batch_mixing_labels
from bcmp.utils.validation import validate_max_underrepresentation_fold


@pytest.mark.parametrize(
    "value",
    [
        1,
        2.5,
        3,
        3.33,
        10,
        np.int32(1),
        np.int64(10),
        np.uint64(3),
        np.float16(2.5),
        np.float32(3.33),
        np.float64(3.33),
        np.longdouble(3.33),
        np.finfo(float).max,
    ],
)
def test_fold_accepts_finite_real_scalars(value):
    assert validate_max_underrepresentation_fold(value) == float(value)
    result = evaluate_bcmp_batch_mixing_labels(
        ["a"] * 10,
        ["x", "y"] * 5,
        min_cells_in_domain_per_batch=1,
        max_underrepresentation_fold=value,
    )
    assert result["pass"] is True


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        0.99,
        np.nextafter(1.0, 0.0),
        Fraction(2**54 - 1, 2**54),
        np.nextafter(np.longdouble(1), np.longdouble(0)),
        np.nan,
        np.inf,
        -np.inf,
        np.float32(np.nan),
        True,
        False,
        np.bool_(True),
        "3.33",
        3.33 + 0j,
        np.complex128(3.33),
        None,
        [3.33],
        (3.33,),
        np.array(3.33),
        np.array([3.33]),
        pd.Series([3.33]),
        {"fold": 3.33},
    ],
)
def test_invalid_fold_is_rejected_at_both_public_boundaries(value):
    # Deliberately unusable data confirms validation precedes preprocessing/KNN.
    for call in (
        lambda: bcmp(None, max_underrepresentation_fold=value),
        lambda: bcmp_embedding(None, None, max_underrepresentation_fold=value),
        lambda: evaluate_bcmp_batch_mixing_labels(
            ["a"] * 10,
            ["x", "y"] * 5,
            max_underrepresentation_fold=value,
        ),
    ):
        with pytest.raises(
            ValueError, match="max_underrepresentation_fold.*finite real scalar"
        ):
            call()


@pytest.mark.parametrize(
    "minority_count,fold,expected",
    [
        (4, np.nextafter(2.5, 0.0), False),
        (4, 2.5, True),
        (4, np.nextafter(2.5, np.inf), True),
        (3, 3.33, False),
        (3, 3.34, True),
    ],
)
def test_fold_threshold_is_inclusive_and_fractional(minority_count, fold, expected):
    # 100 cells, 50 per batch; the first domain contains 20 cells.
    batches = (
        ["x"] * minority_count
        + ["y"] * (20 - minority_count)
        + ["x"] * (50 - minority_count)
        + ["y"] * (30 + minority_count)
    )
    result = evaluate_bcmp_batch_mixing_labels(
        ["a"] * 20 + ["b"] * 80,
        batches,
        min_cells_in_domain_per_batch=1,
        max_underrepresentation_fold=fold,
        max_residual_cell_frac=0,
    )
    assert result["pass"] is expected


def test_fold_rejects_values_outside_finite_float_range():
    for value in (10**400, Fraction(10**400)):
        for call in (
            lambda: bcmp(None, max_underrepresentation_fold=value),
            lambda: bcmp_embedding(None, None, max_underrepresentation_fold=value),
        ):
            with pytest.raises(ValueError, match="fit in a finite Python float"):
                call()


@pytest.mark.parametrize("fold", [2.5, np.float64(3.33)])
def test_public_results_preserve_fractional_fold_in_debug_and_json(fold):
    rng = np.random.default_rng(236)
    counts = rng.poisson(rng.uniform(0.5, 20, size=80), size=(108, 80))
    adata = ad.AnnData(
        counts.astype(float),
        obs=pd.DataFrame(
            {"batch": ["x", "y", "z"] * 36}, index=[f"cell_{i}" for i in range(108)]
        ),
    )
    common = dict(
        k_min=10,
        k_max=10,
        max_underrepresentation_fold=fold,
        min_cells_in_domain_per_batch=1,
        verbose=False,
    )
    debug = bcmp(
        adata, partition_n_hvg=40, partition_n_pcs=4, output_level="debug", **common
    )
    standard = bcmp(adata, partition_n_hvg=40, partition_n_pcs=4, **common)
    embedding = debug.adata.obsm["X_bcmp_pca"]
    emb_debug = bcmp_embedding(
        embedding,
        adata.obs["batch"],
        cell_ids=adata.obs_names,
        output_level="debug",
        **common,
    )
    emb_standard = bcmp_embedding(
        embedding, adata.obs["batch"], cell_ids=adata.obs_names, **common
    )
    for result in (debug, emb_debug):
        assert result.debug.parameters["max_underrepresentation_fold"] == float(fold)
        serialized = json.loads(json.dumps(result.debug.parameters, allow_nan=False))
        assert serialized["max_underrepresentation_fold"] == float(fold)
    for result in (standard, emb_debug, emb_standard):
        assert asdict(result.selection) == asdict(debug.selection)
        pd.testing.assert_frame_equal(result.search_trace, debug.search_trace)
    np.testing.assert_array_equal(
        standard.adata.obs["bcmp_domain"].astype(str), emb_debug.labels.astype(str)
    )


@pytest.mark.parametrize(
    "name", ["min_cells_in_domain_per_batch", "min_batch_coverage"]
)
def test_other_integer_thresholds_still_reject_fractions(name):
    with pytest.raises(ValueError, match="integer"):
        evaluate_bcmp_batch_mixing_labels(["a"] * 10, ["x", "y"] * 5, **{name: 2.5})
