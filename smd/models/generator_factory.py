from __future__ import annotations

import os
from typing import Any, Dict

import torch

from smd.models.diffusion_models import GaussianDiffusionModel
from smd.models.diffusion_models.temporal_unet import TemporalUnet, UNET_DIM_MULTS
from smd.models.flow_models import FlowModelsEnsemble, TaskIdentityContextModel, TrajectoryDriftFlowMatchingModel


DEFAULT_GENERATOR_FAMILY = "diffusion"
DEFAULT_DFM_PROJECT_NAME = "Diffusion_MRMP_DFM"


def resolve_generator_family(args: Dict[str, Any]) -> str:
    if args.get("generator_family") is not None:
        return args["generator_family"]
    if args.get("generator_model_class") == "TrajectoryDriftFlowMatchingModel":
        return "dfm"
    return DEFAULT_GENERATOR_FAMILY


def resolve_generator_model_class(args: Dict[str, Any]) -> str:
    if args.get("generator_model_class") is not None:
        return args["generator_model_class"]
    if resolve_generator_family(args) == "dfm":
        return "TrajectoryDriftFlowMatchingModel"
    return args["diffusion_model_class"]


def resolve_generator_rollout_steps(args: Dict[str, Any]) -> int:
    if args.get("generator_rollout_steps") is not None:
        return int(args["generator_rollout_steps"])
    return int(args["n_diffusion_steps"])


def build_task_context(dataset, start_state_pos: torch.Tensor, goal_state_pos: torch.Tensor, normalize: bool = True):
    task = torch.cat((dataset.robot.get_position(start_state_pos), dataset.robot.get_position(goal_state_pos)), dim=-1)
    if normalize:
        task = dataset.normalize_tasks(task)
    return {"tasks": task}


def _common_unet_configs(args: Dict[str, Any], dataset) -> Dict[str, Any]:
    return dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args["unet_input_dim"],
        dim_mults=UNET_DIM_MULTS[args["unet_dim_mults_option"]],
    )


def build_generator_from_args(args: Dict[str, Any], dataset, tensor_args):
    generator_family = resolve_generator_family(args)
    unet_configs = _common_unet_configs(args, dataset)

    if generator_family == "dfm":
        task_dim = dataset.fields[f"{dataset.field_key_task}_normalized"].shape[-1]
        conditioning_embed_dim = int(args.get("dfm_conditioning_embed_dim", task_dim + 2))
        dfm_unet_configs = dict(
            conditioning_embed_dim=conditioning_embed_dim,
            conditioning_type=args.get("dfm_conditioning_type", "default"),
            **unet_configs,
        )
        generator = TrajectoryDriftFlowMatchingModel(
            model=TemporalUnet(**dfm_unet_configs),
            context_model=TaskIdentityContextModel(task_dim=task_dim),
            n_diffusion_steps=resolve_generator_rollout_steps(args),
            predict_epsilon=False,
            dfm_groups_per_task=args.get("dfm_groups_per_task", args.get("dfm_trajectories_per_task", 4)),
            dfm_drift_form=args.get("dfm_drift_form", "split_v2"),
            dfm_omega_min=args.get("dfm_omega_min", 1.0),
            dfm_omega_max=args.get("dfm_omega_max", 9.0),
            dfm_omega_exponent=args.get("dfm_omega_exponent", 1.0),
            dfm_unconditional_per_group=args.get("dfm_unconditional_per_group", 64),
            dfm_use_ema=args.get("dfm_use_ema", args.get("use_ema", True)),
            dfm_ema_decay=args.get("dfm_ema_decay", args.get("ema_decay", 0.999)),
            dfm_P_mean_t=args.get("dfm_P_mean_t", -1.0),
            dfm_P_std_t=args.get("dfm_P_std_t", 2.5),
            dfm_P_mean_r=args.get("dfm_P_mean_r", 1.0),
            dfm_P_std_r=args.get("dfm_P_std_r", 2.5),
            dfm_norm_eps=args.get("dfm_norm_eps", 1e-4),
            dfm_norm_p=args.get("dfm_norm_p", 0.0),
            dfm_kernel_temp_pos=args.get("dfm_kernel_temp_pos", 1.0),
            dfm_kernel_temp_neg=args.get("dfm_kernel_temp_neg", 1.0),
            dfm_sinkhorn_iters=args.get("dfm_sinkhorn_iters", 20),
            dfm_inference_omega=args.get("dfm_inference_omega", 1.0),
            **dfm_unet_configs,
        ).to(tensor_args["device"])
        generator.submodules = {}
        generator.horizon = dataset.n_support_points
        return generator

    diffusion_configs = dict(
        variance_schedule=args["variance_schedule"],
        n_diffusion_steps=args["n_diffusion_steps"],
        predict_epsilon=args["predict_epsilon"],
    )
    generator = GaussianDiffusionModel(
        model=TemporalUnet(**unet_configs),
        **diffusion_configs,
        **unet_configs,
    ).to(tensor_args["device"])
    generator.submodules = {}
    generator.horizon = dataset.n_support_points
    return generator


def load_generator_checkpoint(model, model_dir: str, args: Dict[str, Any], tensor_args):
    checkpoint_name = "ema_model_current_state_dict.pth" if args.get("use_ema", True) else "model_current_state_dict.pth"
    checkpoint_path = os.path.join(model_dir, "checkpoints", checkpoint_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=tensor_args["device"]))
    return model


def build_generator_ensemble(models, transforms, args):
    if resolve_generator_family(args) == "dfm":
        return FlowModelsEnsemble(models, transforms)
    from smd.models.diffusion_models import DiffusionsEnsemble
    return DiffusionsEnsemble(models, transforms)
