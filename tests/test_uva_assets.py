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
        self.assertEqual(roots["trajectories_root"], "/scratch/tester/Diffusion_MRMP_DFM/data/data_trajectories")
        self.assertEqual(roots["trained_models_root"], "/scratch/tester/Diffusion_MRMP_DFM/data/data_trained_models")
        self.assertEqual(roots["instances_root"], "/scratch/tester/Diffusion_MRMP_DFM/data/instances_data")
        self.assertEqual(roots["init4proj_root"], "/scratch/tester/Diffusion_MRMP_DFM/data/init4proj_data")
        self.assertEqual(roots["experiments_root"], "/scratch/tester/Diffusion_MRMP_DFM/runs/experiments")

    def test_runtime_and_slurm_assets_reference_canonical_env_vars(self):
        runtime_env = (REPO_ROOT / "scripts/uva/runtime_env.sh").read_text(encoding="utf-8")
        bootstrap = (REPO_ROOT / "scripts/uva/bootstrap_home_checkout.sh").read_text(encoding="utf-8")
        train_job = (REPO_ROOT / "scripts/slurm/train_generator.sbatch").read_text(encoding="utf-8")
        inference_job = (REPO_ROOT / "scripts/slurm/inference_composite.sbatch").read_text(encoding="utf-8")
        collision_job = (REPO_ROOT / "scripts/slurm/check_collision.sbatch").read_text(encoding="utf-8")
        setup_env = (REPO_ROOT / "scripts/uva/setup_miniforge_env.sh").read_text(encoding="utf-8")
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("SMD_PROJECT_NAME", runtime_env)
        self.assertIn("SMD_PROJECT_ROOT", runtime_env)
        self.assertIn("SMD_DATA_ROOT", runtime_env)
        self.assertIn("data_trained_models", runtime_env)
        self.assertIn("SMD_RUNS_ROOT", train_job)
        self.assertIn("scripts/train/train_generator.py", train_job)
        self.assertIn("SMD_GIT_URL", bootstrap)
        self.assertIn("SMD_GIT_REF", bootstrap)
        self.assertIn("git clone", bootstrap)
        self.assertNotIn("rsync -a", bootstrap)
        self.assertNotIn("#SBATCH -A", train_job)
        self.assertNotIn("#SBATCH -A", inference_job)
        self.assertNotIn("#SBATCH -A", collision_job)
        self.assertIn("SMD_INFERENCE_ARGS", inference_job)
        self.assertIn("launch_smd_composite_experiment.py", inference_job)
        self.assertIn("SMD_COLLISION_ARGS", collision_job)
        self.assertIn("is_collision.py", collision_job)
        self.assertIn("setuptools==70.2.0", setup_env)
        self.assertIn("torch==2.1.2", setup_env)
        self.assertIn("ipopt", setup_env)
        self.assertNotIn("Hydra==2.5", requirements)


if __name__ == "__main__":
    unittest.main()
