"""Runtime configuration and progress behavior checks."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import warnings

import numpy as np
import pytest

import bcmp
import bcmp.settings as settings
from bcmp.core._progress import (
    _close_progress_logger,
    _emit_progress_line,
    _make_progress_logger,
)
from bcmp.core.search import derive_k_search_bounds
from bcmp.core.trace import emit_partition_warning, format_partition_runtime_notice


def _reset_cache(monkeypatch) -> None:
    for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(settings, "_CONFIG", None)
    monkeypatch.setattr(settings, "_CONFIG_REQUESTED_BASE", None)
    monkeypatch.setattr(
        settings, "_SENSITIVE_MODULES", {key: () for key in settings._ENV_KEYS}
    )


def test_cache_info_and_api_configuration_report_the_current_environment(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_cache(monkeypatch)
    assert bcmp.cache_info()["source"] == "unconfigured"

    info = bcmp.set_cache_dir(tmp_path / "runtime")

    assert bcmp.set_cache_dir(tmp_path / "runtime") == info
    assert info["source"] == "api"
    assert Path(os.environ["NUMBA_CACHE_DIR"]).is_dir()
    assert Path(os.environ["MPLCONFIGDIR"]).is_dir()
    assert Path(os.environ["XDG_CACHE_HOME"]).is_dir()


def test_cache_api_respects_complete_existing_environment(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_cache(monkeypatch)
    base_dir = tmp_path / "shell"
    paths = {
        "NUMBA_CACHE_DIR": base_dir / "bcmp_numba_cache",
        "MPLCONFIGDIR": base_dir / "bcmp_mpl_config",
        "XDG_CACHE_HOME": base_dir / "bcmp_xdg_cache",
    }
    for key, path in paths.items():
        monkeypatch.setenv(key, str(path))

    info = bcmp.set_cache_dir(base_dir)

    assert info["source"] == "environment"
    assert info["base_dir"] == str(base_dir.resolve())
    assert all(path.is_dir() for path in paths.values())


def test_cache_api_completes_a_partial_environment(tmp_path: Path, monkeypatch) -> None:
    _reset_cache(monkeypatch)
    external_numba = tmp_path / "external-numba"
    requested = tmp_path / "requested"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(external_numba))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        info = bcmp.set_cache_dir(requested)

    assert info["source"] == "mixed"
    assert info["numba_cache_dir"] == str(external_numba.resolve())
    assert info["matplotlib_config_dir"] == str(
        (requested / "bcmp_mpl_config").resolve()
    )
    assert len(captured) == 1


def test_progress_lines_and_runtime_warnings_are_human_readable() -> None:
    stream = io.StringIO()
    logger = logging.Logger("bcmp-test", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    progress_logger, owned = _make_progress_logger(verbose=True, progress_logger=logger)
    _emit_progress_line(
        progress_logger=progress_logger,
        record={
            "iter_id": 1,
            "phase": "k_low",
            "k": 3,
            "mixing_status": "pass",
            "n_domains": 4,
            "n_failed_domains": 0,
            "n_residual_domains": 1,
            "actual_residual_cell_frac": 0.0125,
        },
    )
    _close_progress_logger(progress_logger, owned=owned)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        emit_partition_warning("umap", "skipped")

    assert "[iter 1][k_low] k=3" in stream.getvalue()
    assert format_partition_runtime_notice("warning", "umap", "skipped") == str(
        captured[0].message
    )


def test_search_bounds_follow_batch_sizes() -> None:
    bounds = derive_k_search_bounds(
        batch_labels=["a"] * 8 + ["b"] * 5 + ["small"],
        n_total=14,
        k_min=3,
        batch_frac_threshold_for_k_max=0.1,
    )

    assert bounds == {"S": 3, "k_low": 3, "k_high": 4}


@pytest.mark.parametrize("threshold", [0.5, 0.75])
@pytest.mark.parametrize("k_max,expected", [(3, 3), (20, 9)])
def test_explicit_k_max_does_not_require_an_eligible_batch(threshold, k_max, expected):
    bounds = derive_k_search_bounds(
        ["a"] * 5 + ["b"] * 5,
        n_total=10,
        k_max=k_max,
        batch_frac_threshold_for_k_max=threshold,
    )
    assert bounds == {"S": 2, "k_low": 3, "k_high": expected}


def test_automatic_k_max_still_requires_a_batch_strictly_above_threshold():
    with pytest.raises(ValueError, match="no batches exceed"):
        derive_k_search_bounds(
            ["a"] * 5 + ["b"] * 5,
            n_total=10,
            batch_frac_threshold_for_k_max=0.5,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1, np.nan, np.inf, -np.inf])
def test_explicit_k_max_still_validates_the_fraction(threshold):
    with pytest.raises(ValueError, match="finite numeric fraction"):
        derive_k_search_bounds(
            ["a"] * 5 + ["b"] * 5,
            n_total=10,
            k_max=3,
            batch_frac_threshold_for_k_max=threshold,
        )


def test_cache_api_requires_setup_before_sensitive_imports(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_cache(monkeypatch)
    monkeypatch.setattr(
        settings,
        "_SENSITIVE_MODULES",
        {
            "NUMBA_CACHE_DIR": ("already_loaded_numba",),
            "MPLCONFIGDIR": (),
            "XDG_CACHE_HOME": (),
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "already_loaded_numba", object())

    with pytest.raises(RuntimeError, match="must be configured before importing"):
        bcmp.set_cache_dir(tmp_path / "late")


def test_cache_api_force_sets_all_runtime_paths(tmp_path, monkeypatch) -> None:
    _reset_cache(monkeypatch)
    for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(key, str(tmp_path / "old" / key))

    info = bcmp.set_cache_dir(tmp_path / "forced", force=True)

    assert info["source"] == "api"
    assert info["base_dir"] == str((tmp_path / "forced").resolve())


def test_cache_api_explains_conflicting_runtime_paths(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_cache(monkeypatch)
    bcmp.set_cache_dir(tmp_path / "first")

    with pytest.raises(
        RuntimeError, match="Conflict detected in BCMP runtime cache configuration"
    ):
        bcmp.set_cache_dir(tmp_path / "second")


def test_default_progress_logger_writes_and_closes(capsys) -> None:
    logger, owned = _make_progress_logger(verbose=True)
    _emit_progress_line(
        progress_logger=logger,
        record={
            "iter_id": 2,
            "phase": "k_mid",
            "k": 4,
            "mixing_status": "pass",
            "n_domains": 3,
            "n_failed_domains": 0,
            "n_residual_domains": 0,
            "actual_residual_cell_frac": 0.0,
        },
    )
    _close_progress_logger(logger, owned=owned)

    assert "[iter 2][k_mid] k=4" in capsys.readouterr().out
    assert logger is not None and not logger.handlers


def test_cache_info_describes_shell_provided_paths(tmp_path: Path, monkeypatch) -> None:
    _reset_cache(monkeypatch)
    base_dir = tmp_path / "provided"
    for key, name in settings._CACHE_SUBDIRS.items():
        monkeypatch.setenv(key, str(base_dir / name))

    info = bcmp.cache_info()

    assert info["source"] == "environment"
    assert info["base_dir"] == str(base_dir.resolve())


def test_cache_info_describes_a_partial_shell_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_cache(monkeypatch)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba"))

    info = bcmp.cache_info()

    assert info["source"] == "partial_environment"
    assert info["configured"] is False
