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
    return {
        "project_name": project_name,
        "project_root": f"/home/{current_user}/{project_name}",
        "data_root": f"/scratch/{current_user}/{project_name}/data",
        "runs_root": f"/scratch/{current_user}/{project_name}/runs",
        "env_root": f"/home/{current_user}/envs/{project_name}",
    }


def resolve_runtime_config(runtime: Any | None = None) -> dict[str, str]:
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, dict):
        runtime = {"project_name": getattr(runtime, "project_name", None)}
    project_name = runtime.get("project_name") or os.getenv(RUNTIME_ENV_VARS["project_name"]) or "Diffusion_MRMP_DFM"
    return default_runtime_roots(project_name=project_name)
