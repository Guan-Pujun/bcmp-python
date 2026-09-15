"""Process-level runtime cache settings for BCMP."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from threading import Lock
import warnings
from typing import Any, Literal

__all__ = ["set_cache_dir", "cache_info"]

CacheSource = Literal[
    "api", "environment", "mixed", "partial_environment", "unconfigured"
]

_NUMBA_ENV = "NUMBA_CACHE_DIR"
_MPL_ENV = "MPLCONFIGDIR"
_XDG_ENV = "XDG_CACHE_HOME"
_ENV_KEYS = (_NUMBA_ENV, _MPL_ENV, _XDG_ENV)
_PRIMARY_ENV_KEYS = (_NUMBA_ENV, _MPL_ENV)
_INFO_FIELDS = {
    _NUMBA_ENV: "numba_cache_dir",
    _MPL_ENV: "matplotlib_config_dir",
    _XDG_ENV: "xdg_cache_home",
}
_CACHE_SUBDIRS = {
    _NUMBA_ENV: "bcmp_numba_cache",
    _MPL_ENV: "bcmp_mpl_config",
    _XDG_ENV: "bcmp_xdg_cache",
}
_SENSITIVE_MODULES = {
    _NUMBA_ENV: ("numba", "umap"),
    _MPL_ENV: ("matplotlib",),
    _XDG_ENV: ("matplotlib",),
}

_CACHE_LOCK = Lock()
_CONFIG: dict[str, Any] | None = None
_CONFIG_REQUESTED_BASE: Path | None = None


def _info_matches_desired(info: dict[str, Any], desired_paths: dict[str, Path]) -> bool:
    for key in _ENV_KEYS:
        configured_path = info.get(_INFO_FIELDS[key])
        if (
            configured_path is None
            or _resolve_path(str(configured_path)) != desired_paths[key]
        ):
            return False
    return True


def _raise_conflicting_cache_dir(
    *,
    requested_base: Path,
    requested_paths: dict[str, Path],
    current_info: dict[str, Any],
) -> None:
    expected = ", ".join(f"{key}={str(requested_paths[key])!r}" for key in _ENV_KEYS)
    current = ", ".join(
        f"{key}={current_info.get(_INFO_FIELDS[key])!r}" for key in _ENV_KEYS
    )
    raise RuntimeError(
        "Conflict detected in BCMP runtime cache configuration. "
        f"set_cache_dir({str(requested_base)!r}) expects {expected}, "
        f"but the current process has {current}. "
        "To safely isolate BCMP, clear the existing shell environment first, "
        f"or call set_cache_dir({str(requested_base)!r}, force=True) before "
        "importing numba, umap, or matplotlib."
    )


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _path_from_env(key: str) -> Path | None:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return None
    return _resolve_path(raw)


def _cache_paths(base_dir: str | Path) -> dict[str, Path]:
    base = _resolve_path(base_dir)
    return {"base_dir": base, **{key: base / _CACHE_SUBDIRS[key] for key in _ENV_KEYS}}


def _infer_base_dir(cache_paths: dict[str, Path | None]) -> Path | None:
    if any(cache_paths[key] is None for key in _ENV_KEYS):
        return None
    parents: set[Path] = set()
    for key in _ENV_KEYS:
        path = cache_paths[key]
        if path is None or path.name != _CACHE_SUBDIRS[key]:
            return None
        parents.add(path.parent)
    return parents.pop() if len(parents) == 1 else None


def _loaded_sensitive_modules(key: str) -> list[str]:
    loaded: list[str] = []
    for module_name in _SENSITIVE_MODULES[key]:
        if module_name in sys.modules:
            loaded.append(module_name)
    return loaded


def _require_safe_to_write(keys: list[str]) -> None:
    blocked = {
        key: loaded for key in keys if (loaded := _loaded_sensitive_modules(key))
    }
    if not blocked:
        return
    details = "; ".join(
        f"{key} after modules already loaded: {', '.join(modules)}"
        for key, modules in blocked.items()
    )
    raise RuntimeError(
        "BCMP runtime cache environment must be configured before importing "
        f"cache-sensitive runtime modules ({details})"
    )


def _ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                "Unable to create BCMP runtime cache directory "
                f"{path}. Check path permissions, disk quota, and available space."
            ) from exc


def _warn_mixed_cache_config(
    *,
    requested_base: Path,
    final_paths: dict[str, Path | None],
    keys_to_write: list[str],
) -> None:
    external_parts: list[str] = []
    routed_parts: list[str] = []
    for key in _ENV_KEYS:
        path = final_paths[key]
        if path is None:
            continue
        if key in keys_to_write:
            routed_parts.append(f"{key}={str(path)!r}")
        else:
            external_parts.append(f"{key}={str(path)!r}")

    warnings.warn(
        "External runtime cache environment detected: "
        f"{'; '.join(external_parts)}. "
        "Only the missing cache variables were routed under requested "
        f"cache_dir {str(requested_base)!r}: {'; '.join(routed_parts)}. "
        "Use force=True to override all BCMP runtime cache paths.",
        UserWarning,
        stacklevel=3,
    )


def _build_info(
    *,
    source: CacheSource,
    cache_paths: dict[str, Path | None],
    base_dir: Path | None = None,
) -> dict[str, Any]:
    inferred_base = base_dir if base_dir is not None else _infer_base_dir(cache_paths)
    return {
        "configured": all(cache_paths[key] is not None for key in _ENV_KEYS),
        "source": source,
        "base_dir": None if inferred_base is None else str(inferred_base),
        **{
            _INFO_FIELDS[key]: None
            if cache_paths[key] is None
            else str(cache_paths[key])
            for key in _ENV_KEYS
        },
    }


def _environment_info() -> dict[str, Any]:
    cache_paths = {key: _path_from_env(key) for key in _ENV_KEYS}
    configured_count = sum(cache_paths[key] is not None for key in _ENV_KEYS)
    if configured_count == len(_ENV_KEYS):
        source: CacheSource = "environment"
    elif configured_count:
        source = "partial_environment"
    else:
        source = "unconfigured"
    return _build_info(source=source, cache_paths=cache_paths)


def set_cache_dir(cache_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Configure BCMP runtime caches for the current process.

    Existing shell-provided ``NUMBA_CACHE_DIR``, ``MPLCONFIGDIR``, and
    ``XDG_CACHE_HOME`` values are respected by default. Pass ``force=True`` to
    replace them.
    """
    global _CONFIG, _CONFIG_REQUESTED_BASE

    desired = _cache_paths(cache_dir)
    desired_base = desired["base_dir"]
    desired_paths = {key: desired[key] for key in _ENV_KEYS}

    with _CACHE_LOCK:
        env_paths = {key: _path_from_env(key) for key in _ENV_KEYS}

        if not force and _CONFIG is not None:
            env_matches_config = True
            for key in _ENV_KEYS:
                config_path = _CONFIG.get(_INFO_FIELDS[key])
                if config_path is None or env_paths[key] != _resolve_path(
                    str(config_path)
                ):
                    env_matches_config = False
                    break
            if env_matches_config:
                if _CONFIG_REQUESTED_BASE == desired_base:
                    return dict(_CONFIG)
                _raise_conflicting_cache_dir(
                    requested_base=desired_base,
                    requested_paths=desired_paths,
                    current_info=_CONFIG,
                )

        if not force and all(env_paths[key] is not None for key in _ENV_KEYS):
            info = _build_info(source="environment", cache_paths=env_paths)
            if not _info_matches_desired(info, desired_paths):
                _raise_conflicting_cache_dir(
                    requested_base=desired_base,
                    requested_paths=desired_paths,
                    current_info=info,
                )
            _ensure_dirs([path for path in env_paths.values() if path is not None])
            _CONFIG = dict(info)
            _CONFIG_REQUESTED_BASE = desired_base
            return dict(info)

        if not force and all(env_paths[key] is not None for key in _PRIMARY_ENV_KEYS):
            primary_paths_match_request = all(
                env_paths[key] == desired_paths[key] for key in _PRIMARY_ENV_KEYS
            )
            if not primary_paths_match_request:
                info = _build_info(source="partial_environment", cache_paths=env_paths)
                _raise_conflicting_cache_dir(
                    requested_base=desired_base,
                    requested_paths=desired_paths,
                    current_info=info,
                )

        keys_to_write: list[str] = []
        final_paths = dict(env_paths)
        source: CacheSource

        if force:
            final_paths = dict(desired_paths)
            keys_to_write = list(_ENV_KEYS)
            source = "api"
        else:
            for key in _ENV_KEYS:
                if final_paths[key] is None:
                    final_paths[key] = desired_paths[key]
                    keys_to_write.append(key)
            source = (
                "api"
                if all(final_paths[key] == desired_paths[key] for key in _ENV_KEYS)
                else "mixed"
            )

        _require_safe_to_write(keys_to_write)
        if source == "mixed":
            _warn_mixed_cache_config(
                requested_base=desired_base,
                final_paths=final_paths,
                keys_to_write=keys_to_write,
            )
        _ensure_dirs([path for path in final_paths.values() if path is not None])

        for key in keys_to_write:
            os.environ[key] = str(final_paths[key])

        base = desired_base if source == "api" else None
        info = _build_info(source=source, cache_paths=final_paths, base_dir=base)
        _CONFIG = dict(info)
        _CONFIG_REQUESTED_BASE = desired_base
        return dict(info)


def cache_info() -> dict[str, Any]:
    """Return the currently observed BCMP runtime cache configuration."""
    with _CACHE_LOCK:
        if _CONFIG is not None:
            env_paths = {key: _path_from_env(key) for key in _ENV_KEYS}
            env_matches_config = True
            for key in _ENV_KEYS:
                config_path = _CONFIG.get(_INFO_FIELDS[key])
                if config_path is None or env_paths[key] != _resolve_path(
                    str(config_path)
                ):
                    env_matches_config = False
                    break
            if env_matches_config:
                return dict(_CONFIG)
        return _environment_info()
