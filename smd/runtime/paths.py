from __future__ import annotations

import os
from pathlib import Path
from typing import Any


RUNTIME_ENV_VARS = {
    "project_name": "SMD_PROJECT_NAME",
    "project_root": "SMD_PROJECT_ROOT",
    "data_root": "SMD_DATA_ROOT",
    "runs_root": "SMD_RUNS_ROOT",
    "env_root": "SMD_ENV_ROOT",
    "trajectories_root": "SMD_TRAJECTORIES_ROOT",
    "trained_models_root": "SMD_TRAINED_MODELS_ROOT",
    "instances_root": "SMD_INSTANCES_ROOT",
    "init4proj_root": "SMD_INIT4PROJ_ROOT",
    "experiments_root": "SMD_EXPERIMENTS_ROOT",
}

DERIVED_RUNTIME_ROOTS = {
    "trajectories_root": ("data_root", "data_trajectories"),
    "trained_models_root": ("data_root", "data_trained_models"),
    "instances_root": ("data_root", "instances_data"),
    "init4proj_root": ("data_root", "init4proj_data"),
    "experiments_root": ("runs_root", "experiments"),
}


def discover_project_root(start: str | Path | None = None) -> Path:
    anchor = Path(start).resolve() if start is not None else Path(__file__).resolve()
    current = anchor if anchor.is_dir() else anchor.parent
    for directory in (current, *current.parents):
        if (directory / "setup.py").exists():
            return directory
    raise ValueError("Could not discover the Diffusion-MRMP project root.")


def _current_user() -> str:
    return os.getenv("USER") or os.getenv("LOGNAME") or "unknown"


def default_runtime_roots(project_name: str = "Diffusion_MRMP_DFM") -> dict[str, str]:
    current_user = _current_user()
    roots = {
        "project_name": project_name,
        "project_root": f"/home/{current_user}/{project_name}",
        "data_root": f"/scratch/{current_user}/{project_name}/data",
        "runs_root": f"/scratch/{current_user}/{project_name}/runs",
        "env_root": f"/home/{current_user}/envs/{project_name}",
    }
    for key, (parent_key, leaf_name) in DERIVED_RUNTIME_ROOTS.items():
        roots[key] = str(Path(roots[parent_key]) / leaf_name)
    return roots


def _runtime_value(runtime: dict[str, Any], key: str, fallback: str | None = None) -> str | None:
    env_var = RUNTIME_ENV_VARS.get(key)
    return runtime.get(key) or (os.getenv(env_var) if env_var is not None else None) or fallback


def resolve_runtime_config(runtime: Any | None = None) -> dict[str, str]:
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, dict):
        runtime = {key: getattr(runtime, key, None) for key in RUNTIME_ENV_VARS}

    project_name = _runtime_value(runtime, "project_name", "Diffusion_MRMP_DFM")
    roots = default_runtime_roots(project_name=project_name)

    for key in ("project_root", "data_root", "runs_root", "env_root"):
        roots[key] = _runtime_value(runtime, key, roots[key]) or roots[key]

    for key, (parent_key, leaf_name) in DERIVED_RUNTIME_ROOTS.items():
        derived_default = str(Path(roots[parent_key]) / leaf_name)
        roots[key] = _runtime_value(runtime, key, derived_default) or derived_default

    return roots
