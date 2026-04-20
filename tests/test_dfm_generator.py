import unittest
from unittest import mock

import torch

from smd.models.flow_models.drift_flow_matching import TrajectoryDriftFlowMatchingModel


class DummyBackbone(torch.nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.last_context = None
        self.linear = torch.nn.Linear(state_dim, state_dim, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x, t, context):
        self.last_context = context
        return self.linear(x)


class CountingGuide:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.zeros_like(x)


def build_model(horizon: int = 5, state_dim: int = 2, groups_per_task: int = 1):
    backbone = DummyBackbone(state_dim=state_dim)
    model = TrajectoryDriftFlowMatchingModel(
        model=backbone,
        n_diffusion_steps=3,
        dfm_groups_per_task=groups_per_task,
        dfm_unconditional_per_group=2,
    )
    model.horizon = horizon
    return model, backbone


class DriftFlowMatchingGeneratorTests(unittest.TestCase):
    def test_run_inference_keeps_hard_conditions(self):
        model, _ = build_model()
        context = {"tasks": torch.tensor([[0.0, 1.0, 2.0, 3.0]])}
        hard_conds = {
            0: torch.tensor([1.0, 2.0]),
            4: torch.tensor([3.0, 4.0]),
        }

        chain = model.run_inference(context=context, hard_conds=hard_conds, n_samples=2, return_chain=True)

        self.assertEqual(chain.shape, (4, 2, 5, 2))
        self.assertTrue(torch.allclose(chain[:, :, 0, :], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(chain[:, :, -1, :], torch.tensor([3.0, 4.0])))

    def test_loss_uses_task_condition_and_task_balanced_groups(self):
        model, backbone = build_model(groups_per_task=1)
        x = torch.randn(4, 5, 2)
        hard_conds = {0: x[:, 0, :].clone(), 4: x[:, -1, :].clone()}
        context = {
            "tasks": torch.randn(4, 4),
            "task_ids": torch.tensor([0, 0, 1, 1]),
        }

        loss, info = model.loss(x, context, hard_conds)

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("drift_loss", info)
        self.assertIsNotNone(backbone.last_context)
        self.assertEqual(backbone.last_context.shape[-1], 6)

    def test_local_inference_matches_legacy_chain_shape(self):
        model, _ = build_model(horizon=6)
        seed = torch.randn(3, 6, 2)
        hard_conds = {
            0: torch.tensor([0.0, 0.0]),
            5: torch.tensor([1.0, 1.0]),
        }
        context = {"tasks": torch.randn(3, 4)}

        chain = model.run_local_inference(
            seed,
            n_noising_steps=1,
            n_denoising_steps=2,
            context=context,
            hard_conds=hard_conds,
            n_samples=3,
            return_chain=True,
        )

        self.assertEqual(chain.shape, (3, 3, 6, 2))
        self.assertTrue(torch.allclose(chain[:, :, 0, :], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(chain[:, :, -1, :], torch.tensor([1.0, 1.0])))

    def test_rollout_triggers_guide_and_projection_hooks(self):
        model, _ = build_model()
        guide = CountingGuide()
        projection_calls = []

        def fake_project(x, **kwargs):
            projection_calls.append(kwargs["proj_params"]["projection_step"])
            return x

        with mock.patch.object(model, "_project_if_needed", side_effect=fake_project):
            context = {"tasks": torch.randn(1, 4)}
            hard_conds = {0: torch.tensor([0.0, 0.0]), 4: torch.tensor([1.0, 1.0])}
            model.run_inference(
                context=context,
                hard_conds=hard_conds,
                n_samples=1,
                return_chain=False,
                guide=guide,
                n_guide_steps=2,
                t_start_guide=10,
                dataset=object(),
                init_traj4proj=None,
                proj_params={"projection_step": 1},
            )

        self.assertEqual(guide.calls, 6)
        self.assertEqual(len(projection_calls), 3)


if __name__ == "__main__":
    unittest.main()
