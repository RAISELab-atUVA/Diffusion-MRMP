import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEP_ROOTS = [
    REPO_ROOT,
    REPO_ROOT / "deps" / "torch_robotics",
    REPO_ROOT / "deps" / "experiment_launcher",
    REPO_ROOT / "deps" / "motion_planning_baselines",
]

for dep_root in DEP_ROOTS:
    dep_root_str = str(dep_root)
    if dep_root_str not in sys.path:
        sys.path.insert(0, dep_root_str)

os.environ.setdefault("MPLBACKEND", "Agg")
