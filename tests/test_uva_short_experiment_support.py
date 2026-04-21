import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class InferenceEntrypointTests(unittest.TestCase):
    def test_init4proj_path_uses_cli_map_name(self):
        source = (REPO_ROOT / "scripts/inference/launch_smd_composite_experiment.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('args.map_name', source)
        self.assertNotIn("f'../../init4proj_data/{map_name}_init4proj_agent_3.pkl'", source)

    def test_dataset_root_does_not_require_git_metadata(self):
        source = (REPO_ROOT / "smd/datasets/trajectories.py").read_text(encoding="utf-8")

        self.assertIn("Path(__file__).resolve().parents[2]", source)
        self.assertNotIn("git.Repo('.', search_parent_directories=True)", source)

    def test_instance_data_paths_are_file_anchored(self):
        env_source = (
            REPO_ROOT / "deps/torch_robotics/torch_robotics/environments/env_empty_nowait_2d_extra_objects.py"
        ).read_text(encoding="utf-8")
        config_source = (REPO_ROOT / "smd/config/smd_experiment_configs.py").read_text(encoding="utf-8")

        self.assertIn("Path(__file__).resolve()", env_source)
        self.assertNotIn("'../../instances_data/'", env_source)
        self.assertIn("Path(__file__).resolve()", config_source)
        self.assertNotIn("'../../instances_data/'", config_source)


class CollisionCliTests(unittest.TestCase):
    def test_three_agent_results_can_be_checked_via_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_root = tmp_path / "results"
            result_dir = (
                results_root
                / "2026-04-20"
                / "trial-0"
                / "instance_name___EnvEmptyNoWait2DRobotCompositeThreePlanarDiskRandom"
                / "num_agents___3"
                / "planner___SMDComposite"
                / "single_agent_planner___SMDEnsemble"
                / "0"
            )
            result_dir.mkdir(parents=True)

            map_info = {"map_name": "instances_simple", "instance_idx": 0}
            with (result_dir / "map_info.pkl").open("wb") as handle:
                pickle.dump(map_info, handle)

            paths_script = """
import numpy as np
from pathlib import Path

result_dir = Path(r\"\"\"%s\"\"\")
paths = np.zeros((1, 4, 12), dtype=float)
paths[0, :, 0:2] = np.array([0.1, 0.1])
paths[0, :, 2:4] = np.array([0.4, 0.4])
paths[0, :, 4:6] = np.array([0.8, 0.8])
np.save(result_dir / "paths.npy", paths)
""" % (result_dir,)
            subprocess.run(
                [sys.executable, "-c", paths_script],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            map_folder = tmp_path / "instances_data"
            map_folder.mkdir()
            map_payload = [[None, None, ([], [[[0.0, 0.0], [0.0, 0.0]]] * 3)]]
            with (map_folder / "instances_simple.pkl").open("wb") as handle:
                pickle.dump(map_payload, handle)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "is_collision.py"),
                    "--results-root",
                    str(results_root),
                    "--experiment-name",
                    "EnvEmptyNoWait2DRobotCompositeThreePlanarDiskRandom",
                    "--num-agents",
                    "3",
                    "--map-folder",
                    str(map_folder),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Success rate: 1.0", proc.stdout)


class ConfigSurfaceTests(unittest.TestCase):
    def test_smoke_and_short_configs_use_legacy_planner_model_id(self):
        smoke_path = REPO_ROOT / "configs/experiments/composite_three_dfm_smoke.yaml"
        short_path = REPO_ROOT / "configs/experiments/composite_three_dfm_short.yaml"

        with smoke_path.open("r", encoding="utf-8") as handle:
            smoke_config = yaml.safe_load(handle)
        with short_path.open("r", encoding="utf-8") as handle:
            short_config = yaml.safe_load(handle)

        self.assertEqual(smoke_config["model_id"], "EnvEmptyNoWait2D-RobotCompositeThreePlanarDisk")
        self.assertEqual(short_config["model_id"], "EnvEmptyNoWait2D-RobotCompositeThreePlanarDisk")
        self.assertEqual(smoke_config["map_name"], "instances_empty")
        self.assertEqual(short_config["map_name"], "instances_empty")
        self.assertEqual(smoke_config["num_train_steps"], 300)
        self.assertEqual(short_config["num_train_steps"], 2000)
        self.assertEqual(smoke_config["dfm_tasks_per_batch"], 4)
        self.assertEqual(short_config["dfm_trajectories_per_task"], 4)


if __name__ == "__main__":
    unittest.main()
