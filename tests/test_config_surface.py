import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigSurfaceTests(unittest.TestCase):
    def test_dfm_experiment_config_uses_new_generator_surface(self):
        config_path = REPO_ROOT / "configs/experiments/composite_three_dfm.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["runtime"]["project_name"], "Diffusion_MRMP_DFM")
        self.assertEqual(config["generator_family"], "dfm")
        self.assertEqual(config["generator_model_class"], "TrajectoryDriftFlowMatchingModel")
        self.assertIn("generator_rollout_steps", config)
        self.assertIn("dfm_groups_per_task", config)
        self.assertIn("dfm_drift_form", config)


if __name__ == "__main__":
    unittest.main()
