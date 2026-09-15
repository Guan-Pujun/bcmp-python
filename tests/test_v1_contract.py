"""Positive v1 public-contract regression tests."""

from __future__ import annotations

import json
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd

from bcmp import bcmp, bcmp_embedding
from bcmp.datasets import diabetic_kidney_lite
import bcmp.settings as settings


FIXTURE_ROOT = Path(__file__).parent / "data"
GOLDEN_ROOT = FIXTURE_ROOT / "golden"


def test_cache_api_configures_the_current_process(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(settings, "_CONFIG", None)
    monkeypatch.setattr(settings, "_CONFIG_REQUESTED_BASE", None)
    monkeypatch.setattr(
        settings,
        "_SENSITIVE_MODULES",
        {key: () for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME")},
    )

    info = settings.set_cache_dir(cache_dir)

    assert info == settings.cache_info()
    assert info["configured"] is True
    assert info["source"] == "api"
    assert Path(info["base_dir"]) == cache_dir.resolve()
    assert Path(info["numba_cache_dir"]).is_dir()
    assert Path(info["matplotlib_config_dir"]).is_dir()
    assert Path(info["xdg_cache_home"]).is_dir()


def _expected_selection() -> dict[str, object]:
    return json.loads((GOLDEN_ROOT / "selection.json").read_text(encoding="utf-8"))


def _expected_labels() -> pd.Series:
    labels = pd.read_csv(
        GOLDEN_ROOT / "partition_labels.tsv", sep="\t", index_col="cell_id"
    ).iloc[:, 0]
    labels.index = labels.index.astype(str)
    return labels.astype(str)


def _expected_trace() -> pd.DataFrame:
    return pd.read_csv(GOLDEN_ROOT / "search_trace.tsv", sep="\t")


def _assert_standard_result(result, assert_golden_labels) -> None:
    assert asdict(result.selection) == _expected_selection()
    observed_labels = result.adata.obs["bcmp_domain"].astype(str)
    observed_labels.index = observed_labels.index.astype(str)
    assert_golden_labels(observed_labels, _expected_labels())
    pd.testing.assert_frame_equal(
        result.search_trace,
        _expected_trace(),
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_anndata_standard_matches_v1_golden(assert_golden_labels) -> None:
    adata = ad.read_h5ad(FIXTURE_ROOT / "base_adata.h5ad")
    _assert_standard_result(bcmp(adata, verbose=False), assert_golden_labels)


def test_anndata_debug_and_embedding_match_v1_golden(assert_golden_labels) -> None:
    adata = ad.read_h5ad(FIXTURE_ROOT / "base_adata.h5ad")
    adata.uns["bcmp"] = {"old": True}
    adata.obsm["X_bcmp_pca"] = np.full((adata.n_obs, 2), -1.0)
    adata.obsm["X_bcmp_umap"] = np.full((adata.n_obs, 2), -1.0)
    result = bcmp(adata, output_level="debug", verbose=False)
    _assert_standard_result(result, assert_golden_labels)
    assert "bcmp" not in result.adata.uns
    assert result.adata.obsm["X_bcmp_pca"].shape == (result.adata.n_obs, 30)
    assert result.adata.obsm["X_bcmp_umap"].shape == (result.adata.n_obs, 2)
    assert not np.all(result.adata.obsm["X_bcmp_pca"] == -1.0)

    debug_root = GOLDEN_ROOT / "debug"
    assert result.debug.parameters == json.loads(
        (debug_root / "parameters.json").read_text(encoding="utf-8")
    )
    assert result.debug.search == json.loads(
        (debug_root / "search.json").read_text(encoding="utf-8")
    )
    expected_workflow = json.loads(
        (debug_root / "workflow.json").read_text(encoding="utf-8")
    )
    assert (
        result.debug.workflow["excluded_cell_ids"]
        == expected_workflow["excluded_cell_ids"]
    )
    assert set(result.debug.workflow["hvg_genes"]) == set(
        expected_workflow["hvg_genes"]
    )
    assert set(result.debug.candidates) == set(result.search_trace["k"])
    for k, candidate in result.debug.candidates.items():
        candidate_root = debug_root / "candidates" / f"k_{k}"
        expected_partition = pd.read_csv(
            candidate_root / "partition_labels.tsv", sep="\t", index_col="cell_id"
        ).iloc[:, 0]
        expected_partition.index = expected_partition.index.astype(str)
        assert_golden_labels(
            candidate.partition.astype(str),
            expected_partition.astype(str),
        )
        expected_summary = pd.read_csv(
            candidate_root / "domain_summary.tsv", sep="\t", dtype={"domain": str}
        )
        pd.testing.assert_frame_equal(
            candidate.domain_summary,
            expected_summary,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )

    embedding_result = bcmp_embedding(
        result.adata.obsm["X_bcmp_pca"],
        result.adata.obs["batch"],
        cell_ids=result.adata.obs_names,
        verbose=False,
    )
    assert asdict(embedding_result.selection) == _expected_selection()
    observed_labels = embedding_result.labels.astype(str)
    observed_labels.index = observed_labels.index.astype(str)
    assert_golden_labels(observed_labels, _expected_labels())
    pd.testing.assert_frame_equal(
        embedding_result.search_trace,
        _expected_trace(),
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_bundled_dataset_supports_vignette() -> None:
    adata = diabetic_kidney_lite()
    benchmark = adata.uns["integration_benchmark"]
    embedding_keys = benchmark["embedding_keys"].tolist()

    assert adata.shape == (1500, 27980)
    assert "batch" in adata.obs
    assert adata.obsm["X_demo_umap"].shape == (adata.n_obs, 2)
    assert all(key in adata.obsm for key in embedding_keys)
    assert set(benchmark["method_names"]) == set(embedding_keys)


def test_cache_api_configures_a_clean_process(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    environment = os.environ.copy()
    for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        environment.pop(key, None)

    script = (
        "import json\n"
        "from bcmp import cache_info, set_cache_dir\n"
        f"set_cache_dir({str(cache_dir)!r})\n"
        "print(json.dumps(cache_info()))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "configured": True,
        "source": "api",
        "base_dir": str(cache_dir.resolve()),
        "numba_cache_dir": str((cache_dir / "bcmp_numba_cache").resolve()),
        "matplotlib_config_dir": str((cache_dir / "bcmp_mpl_config").resolve()),
        "xdg_cache_home": str((cache_dir / "bcmp_xdg_cache").resolve()),
    }


def test_anndata_exclusion_keeps_cells_with_excluded_batch_labels() -> None:
    adata = ad.read_h5ad(FIXTURE_ROOT / "base_adata.h5ad")
    excluded = adata.obs_names[:4].astype(str).tolist()

    result = bcmp(adata, exclude_cells=excluded, verbose=False)

    assert "bcmp_domain" not in adata.obs
    assert result.adata.n_obs == adata.n_obs
    assert (
        result.adata.obs.loc[excluded, "bcmp_domain"]
        .astype(str)
        .str.startswith("Excluded_")
        .all()
    )


def test_embedding_accepts_a_fixed_representation_scale() -> None:
    adata = ad.read_h5ad(FIXTURE_ROOT / "base_adata.h5ad")
    cells = adata.obs.groupby("batch", observed=True).head(60).index
    result = bcmp_embedding(
        adata[cells, :15].X.toarray(),
        adata.obs.loc[cells, "batch"],
        cell_ids=cells,
        k_min=3,
        k_max=3,
        verbose=False,
    )

    assert result.selection.selected_k == 3
    assert result.search_trace["phase"].tolist() == ["k_low"]
    assert result.labels.index.equals(cells)


def test_embedding_selects_the_first_passing_representation_scale() -> None:
    adata = ad.read_h5ad(FIXTURE_ROOT / "base_adata.h5ad")
    cells = adata.obs.groupby("batch", observed=True).head(60).index
    result = bcmp_embedding(
        adata[cells, :15].X.toarray(),
        adata.obs.loc[cells, "batch"],
        cell_ids=cells,
        k_min=3,
        k_max=4,
        min_batch_coverage=2,
        min_cells_in_domain_per_batch=1,
        max_underrepresentation_fold=100,
        verbose=False,
    )

    assert result.selection.selected_k == 3
    assert result.selection.mixing_status == "pass"
    assert result.search_trace["phase"].tolist() == ["k_low"]


def test_embedding_reports_when_no_candidate_scale_passes_mixing() -> None:
    adata = ad.read_h5ad(FIXTURE_ROOT / "base_adata.h5ad")
    cells = adata.obs.groupby("batch", observed=True).head(60).index
    result = bcmp_embedding(
        adata[cells, :15].X.toarray(),
        adata.obs.loc[cells, "batch"],
        cell_ids=cells,
        k_min=3,
        k_max=4,
        min_batch_coverage=3,
        min_cells_in_domain_per_batch=100,
        max_underrepresentation_fold=100,
        verbose=False,
    )

    assert result.selection.mixing_status == "fail"
    assert result.search_trace["phase"].tolist() == ["k_low", "k_high"]
