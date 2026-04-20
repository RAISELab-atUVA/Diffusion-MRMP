import importlib.util
import os
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_paths_module():
    spec = importlib.util.spec_from_file_location("smd_runtime_paths", REPO_ROOT / "smd/runtime/paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UVAAssetsTests(unittest.TestCase):
    def test_default_runtime_roots_follow_uva_contract(self):
        paths = load_paths_module()
        os.environ["USER"] = "tester"
        roots = paths.default_runtime_roots(project_name="Diffusion_MRMP_DFM")

        self.assertEqual(roots["project_root"], "/home/tester/Diffusion_MRMP_DFM")
        self.assertEqual(roots["data_root"], "/scratch/tester/Diffusion_MRMP_DFM/data")
        self.assertEqual(roots["runs_root"], "/scratch/tester/Diffusion_MRMP_DFM/runs")
        self.assertEqual(roots["env_root"], "/home/tester/envs/Diffusion_MRMP_DFM")

    def test_runtime_and_slurm_assets_reference_canonical_env_vars(self):
        runtime_env = (REPO_ROOT / "scripts/uva/runtime_env.sh").read_text(encoding="utf-8")
        bootstrap = (REPO_ROOT / "scripts/uva/bootstrap_home_checkout.sh").read_text(encoding="utf-8")
        train_job = (REPO_ROOT / "scripts/slurm/train_generator.sbatch").read_text(encoding="utf-8")

        self.assertIn("SMD_PROJECT_NAME", runtime_env)
        self.assertIn("SMD_PROJECT_ROOT", runtime_env)
        self.assertIn("SMD_RUNS_ROOT", train_job)
        self.assertIn("scripts/train/train_generator.py", train_job)
        self.assertIn("rsync -a", bootstrap)


if __name__ == "__main__":
    unittest.main()
