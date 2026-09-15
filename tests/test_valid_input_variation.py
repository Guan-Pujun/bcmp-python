"""Small deterministic variations of valid public inputs."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from bcmp import bcmp, bcmp_embedding


def _valid_case(seed: int) -> tuple[ad.AnnData, np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    batches = np.repeat(["batch_a", "batch_b", "batch_c"], 36)
    populations = np.tile(np.repeat(np.arange(3), 12), 3)

    rates = np.full((batches.size, 24), 1.0)
    for population in range(3):
        rates[populations == population, population * 8 : (population + 1) * 8] += 5.0
    counts = rng.poisson(rates).astype(float)

    centers = np.eye(3, 6)[populations] * 4.0
    embedding = centers + rng.normal(scale=0.2, size=centers.shape)
    cell_ids = pd.Index([f"cell_{index}" for index in range(batches.size)])
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"batch": batches}, index=cell_ids),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(counts.shape[1])]),
    )
    return adata, embedding, pd.Series(batches, index=cell_ids)


@pytest.mark.parametrize(
    "seed, integer_type", [(11, int), (29, np.int32), (47, np.int64)]
)
def test_valid_input_variations_complete_for_both_entry_points(
    seed: int, integer_type
) -> None:
    adata, embedding, batch_labels = _valid_case(seed)
    common = {
        "k_min": integer_type(3),
        "k_max": integer_type(3),
        "seed": integer_type(236),
        # No batch exceeds this threshold; explicit k_max bypasses auto derivation.
        "batch_frac_threshold_for_k_max": np.float64(0.5),
        "max_residual_cell_frac": np.float64(0.05),
        "min_batch_coverage": 2,
        "min_cells_in_domain_per_batch": 1,
        "max_underrepresentation_fold": 100,
        "verbose": False,
    }

    anndata_result = bcmp(
        adata,
        batch_key="batch",
        partition_n_hvg=integer_type(16),
        partition_n_pcs=integer_type(8),
        **common,
    )
    embedding_result = bcmp_embedding(
        embedding, batch_labels, cell_ids=adata.obs_names, **common
    )

    assert anndata_result.adata.obs_names.equals(adata.obs_names)
    assert anndata_result.adata.obs["bcmp_domain"].notna().all()
    assert anndata_result.selection.selected_k == 3
    assert anndata_result.search_trace["k"].tolist() == [3]
    assert embedding_result.labels.index.equals(adata.obs_names)
    assert embedding_result.labels.notna().all()
    assert embedding_result.selection.selected_k == 3
    assert embedding_result.search_trace["k"].tolist() == [3]
