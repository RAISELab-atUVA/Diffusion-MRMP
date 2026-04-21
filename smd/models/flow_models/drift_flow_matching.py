"""
Trajectory Drift Flow Matching model compatible with the Diffusion-MRMP planner interface.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn

from smd.models.diffusion_models.sample_functions import apply_cross_conditioning, apply_hard_conditioning, guide_gradient_steps


def _as_batch_tensor(x: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if x is None:
        return None
    x = x.to(device=device, dtype=dtype)
    if x.ndim == 1:
        x = einops.repeat(x, "d -> b d", b=batch_size)
    elif x.ndim == 2 and x.shape[0] == 1 and batch_size > 1:
        x = x.repeat(batch_size, 1)
    return x


def _normalize_group_ids(group_ids: torch.Tensor) -> torch.Tensor:
    if group_ids.ndim > 1:
        group_ids = group_ids.reshape(group_ids.shape[0], -1)[:, 0]
    return group_ids.long()


def logit_normal_timestep_sample(
    mean: float,
    std: float,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.randn((num_samples,), device=device, dtype=dtype)
    sampled = torch.sigmoid(values * std + mean)
    return torch.clamp(sampled, min=0.0, max=1.0)


def sample_drift_timestep(
    mean_t: float,
    std_t: float,
    mean_r: float,
    std_r: float,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = logit_normal_timestep_sample(mean_t, std_t, num_samples, device, dtype)
    r = logit_normal_timestep_sample(mean_r, std_r, num_samples, device, dtype)
    return torch.minimum(t, r), torch.maximum(t, r)


def sample_power_law_omega(
    num_samples: int,
    *,
    omega_min: float,
    omega_max: float,
    exponent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if exponent == 1.0:
        uniform = torch.rand(num_samples, device=device, dtype=dtype)
        return omega_min * torch.exp(uniform * math.log(omega_max / omega_min))
    power = 1.0 - float(exponent)
    lower = omega_min**power
    upper = omega_max**power
    uniform = torch.rand(num_samples, device=device, dtype=dtype)
    return (lower + uniform * (upper - lower)).pow(1.0 / power)


def _expand_marginals(
    marginals: torch.Tensor,
    *,
    expected_size: int,
    leading_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    marginals = marginals.to(device=device, dtype=dtype)
    if marginals.ndim == 1:
        view_shape = (1,) * len(leading_shape) + (expected_size,)
        return marginals.reshape(view_shape).expand(*leading_shape, expected_size)
    return marginals


def _sinkhorn_from_logits(
    logits: torch.Tensor,
    *,
    row_marginals: torch.Tensor,
    col_marginals: torch.Tensor,
    num_iters: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    logits = logits.to(dtype=work_dtype)
    leading_shape = logits.shape[:-2]
    row_marginals = _expand_marginals(
        row_marginals,
        expected_size=logits.shape[-2],
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
    )
    col_marginals = _expand_marginals(
        col_marginals,
        expected_size=logits.shape[-1],
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
    )

    log_row = torch.log(row_marginals.clamp_min(eps))
    log_col = torch.log(col_marginals.clamp_min(eps))
    log_u = torch.zeros_like(logits[..., :, 0])
    log_v = torch.zeros_like(logits[..., 0, :])

    for _ in range(int(num_iters)):
        log_u = log_row - torch.logsumexp(logits + log_v.unsqueeze(-2), dim=-1)
        log_v = log_col - torch.logsumexp(logits + log_u.unsqueeze(-1), dim=-2)

    return torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).clamp_min(0.0)


def _row_normalize_plan(plan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(eps)


def _compute_sinkhorn_barycentric_projection(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    kernel_temp: float,
    num_iters: int,
    row_marginals: Optional[torch.Tensor] = None,
    col_marginals: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    orig_dtype = source_points.dtype
    work_dtype = torch.float32 if source_points.dtype in (torch.float16, torch.bfloat16) else source_points.dtype
    source_points = source_points.to(dtype=work_dtype)
    target_points = target_points.to(dtype=work_dtype)

    if row_marginals is None:
        row_marginals = torch.full(
            (source_points.shape[1],),
            1.0 / float(source_points.shape[1]),
            device=source_points.device,
            dtype=work_dtype,
        )
    if col_marginals is None:
        col_marginals = torch.full(
            (target_points.shape[1],),
            1.0 / float(target_points.shape[1]),
            device=target_points.device,
            dtype=work_dtype,
        )

    distances = torch.cdist(source_points, target_points)
    logits = -distances / float(kernel_temp)
    plan = _sinkhorn_from_logits(
        logits,
        row_marginals=row_marginals,
        col_marginals=col_marginals,
        num_iters=num_iters,
        eps=eps,
    )
    weights = _row_normalize_plan(plan, eps=eps)
    return torch.matmul(weights, target_points).to(dtype=orig_dtype)


def _build_conditioned_block_marginals(
    omega: torch.Tensor,
    *,
    cond_count: int,
    unc_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    col_weights = torch.ones((omega.shape[0], cond_count + unc_count), device=omega.device, dtype=dtype)
    col_weights[:, :cond_count] = omega.unsqueeze(-1).to(dtype=dtype)
    return col_weights / col_weights.sum(dim=-1, keepdim=True)


def compute_grouped_sinkhorn_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    unc_pos: Optional[torch.Tensor],
    unc_neg: Optional[torch.Tensor],
    *,
    drift_form: str,
    omega: Optional[torch.Tensor],
    kernel_temp_pos: float,
    kernel_temp_neg: float,
    sinkhorn_iters: int,
) -> torch.Tensor:
    if drift_form not in {"split_v0", "split_v1", "split_v2"}:
        raise ValueError(f"Unsupported drift_form={drift_form}")

    if drift_form == "split_v0":
        pos_proj = _compute_sinkhorn_barycentric_projection(
            gen,
            pos,
            kernel_temp=kernel_temp_pos,
            num_iters=sinkhorn_iters,
        )
        neg_proj = _compute_sinkhorn_barycentric_projection(
            gen,
            gen,
            kernel_temp=kernel_temp_neg,
            num_iters=sinkhorn_iters,
        )
        return pos_proj - neg_proj

    pos_all = pos
    neg_all = gen
    gen_source = gen
    row_marginals = None
    col_marginals = None
    if unc_pos is not None and unc_neg is not None and unc_pos.shape[1] > 0:
        pos_all = torch.cat([pos, unc_pos], dim=1)
        neg_all = torch.cat([gen, unc_neg], dim=1)
        gen_source = torch.cat([gen, unc_neg.detach()], dim=1)
        col_marginals = _build_conditioned_block_marginals(
            omega,
            cond_count=gen.shape[1],
            unc_count=unc_pos.shape[1],
            dtype=gen.dtype,
        )
        if drift_form == "split_v2":
            row_marginals = _build_conditioned_block_marginals(
                omega,
                cond_count=gen.shape[1],
                unc_count=unc_neg.shape[1],
                dtype=gen.dtype,
            )

    pos_proj = _compute_sinkhorn_barycentric_projection(
        gen_source,
        pos_all,
        kernel_temp=kernel_temp_pos,
        num_iters=sinkhorn_iters,
        row_marginals=row_marginals,
        col_marginals=col_marginals,
    )
    neg_proj = _compute_sinkhorn_barycentric_projection(
        gen_source,
        neg_all,
        kernel_temp=kernel_temp_neg,
        num_iters=sinkhorn_iters,
        row_marginals=row_marginals,
        col_marginals=col_marginals,
    )
    return (pos_proj - neg_proj)[:, : gen.shape[1], :]


def _adaptive_matching_loss(
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    norm_eps: float,
    norm_p: float,
) -> torch.Tensor:
    terms = (predicted_points - target_points) ** 2
    terms = terms.reshape(terms.shape[0], -1).sum(dim=1)
    adaptive_weight = (terms.detach() + norm_eps) ** norm_p
    return torch.mean(terms / adaptive_weight)


def _reshape_flat_groups_to_task_subgroups(
    flat_tensor: torch.Tensor,
    *,
    num_tasks: int,
    samples_per_task: int,
    groups_per_task: int,
) -> torch.Tensor:
    subgroup_size = samples_per_task // groups_per_task
    return flat_tensor.reshape(
        num_tasks,
        samples_per_task,
        *flat_tensor.shape[1:],
    ).reshape(
        num_tasks,
        groups_per_task,
        subgroup_size,
        *flat_tensor.shape[1:],
    )


def _flatten_task_subgroups_to_groups(task_tensor: torch.Tensor) -> torch.Tensor:
    return task_tensor.reshape(task_tensor.shape[0] * task_tensor.shape[1], *task_tensor.shape[2:])


def _group_batch_into_task_subgroups(
    x0: torch.Tensor,
    x1: torch.Tensor,
    task_context: torch.Tensor,
    task_ids: torch.Tensor,
    groups_per_task: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task_ids = _normalize_group_ids(task_ids)
    sort_index = torch.argsort(task_ids)
    x0 = x0[sort_index]
    x1 = x1[sort_index]
    task_context = task_context[sort_index]
    task_ids = task_ids[sort_index]

    unique, counts = torch.unique_consecutive(task_ids, return_counts=True)
    if unique.numel() == 0:
        raise ValueError("task_ids must be non-empty")
    if counts.min() != counts.max():
        raise ValueError("All tasks must contribute the same number of samples to a DFM batch")
    samples_per_task = int(counts[0].item())
    if samples_per_task % groups_per_task != 0:
        raise ValueError("Per-task sample count must be divisible by dfm_groups_per_task")

    num_tasks = unique.numel()
    x0_groups = _reshape_flat_groups_to_task_subgroups(
        x0,
        num_tasks=num_tasks,
        samples_per_task=samples_per_task,
        groups_per_task=groups_per_task,
    )
    x1_groups = _reshape_flat_groups_to_task_subgroups(
        x1,
        num_tasks=num_tasks,
        samples_per_task=samples_per_task,
        groups_per_task=groups_per_task,
    )
    task_context_groups = _reshape_flat_groups_to_task_subgroups(
        task_context,
        num_tasks=num_tasks,
        samples_per_task=samples_per_task,
        groups_per_task=groups_per_task,
    )
    return x0_groups, x1_groups, task_context_groups


def _sample_shared_group_timesteps(
    *,
    mean_t: float,
    std_t: float,
    mean_r: float,
    std_r: float,
    num_tasks: int,
    groups_per_task: int,
    subgroup_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_groups, r_groups = sample_drift_timestep(
        mean_t,
        std_t,
        mean_r,
        std_r,
        groups_per_task,
        device,
        dtype,
    )
    target_shape = (num_tasks, groups_per_task, subgroup_size, 1, 1)
    t_groups = t_groups.view(1, groups_per_task, 1, 1, 1).expand(target_shape)
    r_groups = r_groups.view(1, groups_per_task, 1, 1, 1).expand(target_shape)
    return t_groups, r_groups


def _sample_shared_group_omega(
    *,
    omega_min: float,
    omega_max: float,
    omega_exponent: float,
    num_tasks: int,
    groups_per_task: int,
    subgroup_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    omega_groups = sample_power_law_omega(
        groups_per_task,
        omega_min=omega_min,
        omega_max=omega_max,
        exponent=omega_exponent,
        device=device,
        dtype=dtype,
    )
    return omega_groups.view(1, groups_per_task, 1, 1).expand(num_tasks, groups_per_task, subgroup_size, 1)


def _build_same_time_unc_samples(group_tensor: torch.Tensor, unc_count: int) -> Optional[torch.Tensor]:
    if unc_count <= 0:
        return None
    num_tasks, groups_per_task, subgroup_size = group_tensor.shape[:3]
    feature_shape = group_tensor.shape[3:]
    pools = group_tensor.movedim(0, 1).reshape(groups_per_task, num_tasks * subgroup_size, *feature_shape)
    sample_indices = torch.randint(
        low=0,
        high=pools.shape[1],
        size=(groups_per_task, unc_count),
        device=group_tensor.device,
    )
    group_index = torch.arange(groups_per_task, device=group_tensor.device).unsqueeze(1)
    sampled = pools[group_index, sample_indices]
    return sampled.view(1, groups_per_task, unc_count, *feature_shape).expand(
        num_tasks,
        groups_per_task,
        unc_count,
        *feature_shape,
    )


def _flatten_group_features(group_tensor: torch.Tensor) -> torch.Tensor:
    return group_tensor.reshape(group_tensor.shape[0], group_tensor.shape[1], -1)


class TaskIdentityContextModel(nn.Module):
    def __init__(self, task_dim: int):
        super().__init__()
        self.out_dim = task_dim

    def forward(self, input_d):
        if input_d is None:
            return None
        if isinstance(input_d, dict):
            return input_d["tasks"]
        return input_d


class TrajectoryDriftFlowMatchingModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_diffusion_steps: int = 32,
        context_model: Optional[nn.Module] = None,
        predict_epsilon: bool = False,
        dfm_groups_per_task: int = 4,
        dfm_drift_form: str = "split_v2",
        dfm_omega_min: float = 1.0,
        dfm_omega_max: float = 9.0,
        dfm_omega_exponent: float = 1.0,
        dfm_unconditional_per_group: int = 64,
        dfm_use_ema: bool = True,
        dfm_ema_decay: float = 0.999,
        dfm_P_mean_t: float = -1.0,
        dfm_P_std_t: float = 2.5,
        dfm_P_mean_r: float = 1.0,
        dfm_P_std_r: float = 2.5,
        dfm_norm_eps: float = 1e-4,
        dfm_norm_p: float = 0.0,
        dfm_kernel_temp_pos: float = 1.0,
        dfm_kernel_temp_neg: float = 1.0,
        dfm_sinkhorn_iters: int = 20,
        dfm_inference_omega: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.state_dim = self.model.state_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.predict_epsilon = predict_epsilon
        self.context_model = context_model
        self.dfm_groups_per_task = dfm_groups_per_task
        self.dfm_drift_form = dfm_drift_form
        self.dfm_omega_min = dfm_omega_min
        self.dfm_omega_max = dfm_omega_max
        self.dfm_omega_exponent = dfm_omega_exponent
        self.dfm_unconditional_per_group = dfm_unconditional_per_group
        self.dfm_use_ema = dfm_use_ema
        self.dfm_ema_decay = dfm_ema_decay
        self.dfm_P_mean_t = dfm_P_mean_t
        self.dfm_P_std_t = dfm_P_std_t
        self.dfm_P_mean_r = dfm_P_mean_r
        self.dfm_P_std_r = dfm_P_std_r
        self.dfm_norm_eps = dfm_norm_eps
        self.dfm_norm_p = dfm_norm_p
        self.dfm_kernel_temp_pos = dfm_kernel_temp_pos
        self.dfm_kernel_temp_neg = dfm_kernel_temp_neg
        self.dfm_sinkhorn_iters = dfm_sinkhorn_iters
        self.dfm_inference_omega = dfm_inference_omega
        self.visualize_local_inference = False

    def _encode_base_context(self, context: Optional[torch.Tensor | Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if self.context_model is not None:
            return self.context_model(context)
        if isinstance(context, dict):
            return context.get("tasks")
        return context

    def _derive_task_context_from_states(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x[:, 0, :], x[:, -1, :]), dim=-1)

    def _prepare_velocity_context(
        self,
        base_context: Optional[torch.Tensor],
        h: torch.Tensor,
        omega: Optional[torch.Tensor],
        current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = h.shape[0]
        if base_context is None:
            if current is None:
                raise ValueError("Current trajectory is required when no explicit task context is provided.")
            base_context = self._derive_task_context_from_states(current)
        base_context = _as_batch_tensor(base_context, batch_size, h.device, h.dtype)
        omega = torch.full((batch_size, 1), self.dfm_inference_omega, device=h.device, dtype=h.dtype) if omega is None else omega.to(device=h.device, dtype=h.dtype)
        if omega.ndim == 1:
            omega = omega[:, None]
        return torch.cat((base_context, h[:, None], omega), dim=-1)

    def _project_if_needed(
        self,
        x: torch.Tensor,
        *,
        hard_conds: Optional[dict],
        step_index: int,
        total_steps: int,
        projection_info=None,
        init_traj4proj=None,
        proj_params=None,
    ) -> torch.Tensor:
        if projection_info is None or proj_params is None:
            return x
        projection_step = proj_params.get("projection_step", 0)
        if isinstance(projection_step, (list, tuple)):
            projection_steps = {int(step) for step in projection_step}
            should_project = (step_index in projection_steps) or (step_index == total_steps - 1)
        else:
            projection_step = int(projection_step)
            if projection_step <= 0:
                return x
            should_project = ((step_index + 1) % projection_step == 0) or (step_index == total_steps - 1)
        if not should_project:
            return x
        from smd.projection.projection import apply_projection_alm
        is_first_projection = 1 if step_index == 0 else 0
        x, _ = apply_projection_alm(
            x,
            projection_info,
            hard_conds,
            is_first_projection,
            init_traj4proj,
            proj_params,
        )
        return x

    def sampling_step(
        self,
        current: torch.Tensor,
        *,
        step_index: int,
        total_steps: int,
        context: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None,
        hard_conds: Optional[dict] = None,
        guide=None,
        n_guide_steps: int = 1,
        t_start_guide: float = torch.inf,
        omega: Optional[torch.Tensor] = None,
        projection_info=None,
        init_traj4proj=None,
        proj_params=None,
        **kwargs,
    ) -> torch.Tensor:
        current = apply_hard_conditioning(current, hard_conds or {})
        t0 = step_index / float(total_steps)
        t1 = (step_index + 1) / float(total_steps)
        h_scalar = torch.full((current.shape[0],), t1 - t0, device=current.device, dtype=current.dtype)
        t_scalar = torch.full((current.shape[0],), t0, device=current.device, dtype=current.dtype)
        velocity_context = self._prepare_velocity_context(
            self._encode_base_context(context),
            h_scalar,
            omega,
            current=current,
        )
        velocity = self.model(current, t_scalar, velocity_context)
        current = current + h_scalar[:, None, None] * velocity
        current = apply_hard_conditioning(current, hard_conds or {})

        if guide is not None and (total_steps - step_index - 1) < t_start_guide:
            current = guide_gradient_steps(
                current,
                hard_conds=hard_conds or {},
                guide=guide,
                n_guide_steps=n_guide_steps,
                unnormalize_data=False,
            )
            current = apply_hard_conditioning(current, hard_conds or {})

        current = self._project_if_needed(
            current,
            hard_conds=hard_conds,
            step_index=step_index,
            total_steps=total_steps,
            projection_info=projection_info,
            init_traj4proj=init_traj4proj,
            proj_params=proj_params,
        )
        current = apply_hard_conditioning(current, hard_conds or {})
        return current

    def _run_rollout(
        self,
        *,
        current: torch.Tensor,
        total_steps: int,
        context: Optional[torch.Tensor | Dict[str, torch.Tensor]],
        hard_conds: Optional[dict],
        return_chain: bool,
        **sample_kwargs,
    ):
        chain = [current.clone()] if return_chain else None
        for step_index in range(total_steps):
            current = self.sampling_step(
                current,
                step_index=step_index,
                total_steps=total_steps,
                context=context,
                hard_conds=hard_conds,
                **sample_kwargs,
            )
            if return_chain:
                chain.append(current.clone())
        if return_chain:
            return current, torch.stack(chain, dim=1)
        return current, None

    def _repeat_hard_conds(self, hard_conds: Optional[dict], n_samples: int) -> dict:
        repeated = deepcopy(hard_conds or {})
        for key, value in repeated.items():
            repeated[key] = einops.repeat(value, "d -> b d", b=n_samples)
        return repeated

    @torch.no_grad()
    def warmup(self, horizon: int = 64, device: str = "cuda", context=None):
        sample = torch.randn((1, horizon, self.state_dim), device=device)
        if context is None:
            context = {"tasks": torch.zeros((1, self.state_dim * 2), device=device)}
        hard_conds = {0: torch.zeros((self.state_dim,), device=device), horizon - 1: torch.zeros((self.state_dim,), device=device)}
        self.run_inference(context=context, hard_conds=hard_conds, n_samples=1, return_chain=False)

    @torch.no_grad()
    def run_inference(
        self,
        context=None,
        hard_conds: Optional[dict] = None,
        n_samples: int = 1,
        return_chain: bool = False,
        horizon: Optional[int] = None,
        n_diffusion_steps_without_noise: int = 0,
        **sample_kwargs,
    ):
        del n_diffusion_steps_without_noise
        hard_conds = self._repeat_hard_conds(hard_conds, n_samples)
        if horizon is None:
            horizon = getattr(self, "horizon", None)
        if horizon is None:
            non_negative_keys = [key for key in hard_conds.keys() if key >= 0]
            horizon = max(non_negative_keys) + 1 if non_negative_keys else 64
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        current = torch.randn((n_samples, horizon, self.state_dim), device=device)
        current = apply_hard_conditioning(current, hard_conds)
        final, chain = self._run_rollout(
            current=current,
            total_steps=self.n_diffusion_steps,
            context=context,
            hard_conds=hard_conds,
            return_chain=True,
            projection_info=sample_kwargs.get("dataset"),
            init_traj4proj=sample_kwargs.get("init_traj4proj"),
            proj_params=sample_kwargs.get("proj_params"),
            guide=sample_kwargs.get("guide"),
            n_guide_steps=sample_kwargs.get("n_guide_steps", 1),
            t_start_guide=sample_kwargs.get("t_start_guide", torch.inf),
            omega=sample_kwargs.get("dfm_inference_omega"),
        )
        chain = einops.rearrange(chain, "b steps h d -> steps b h d")
        if return_chain:
            return chain
        return final

    @torch.no_grad()
    def run_local_inference(
        self,
        seed_trajectory_b: torch.Tensor,
        n_noising_steps: int,
        n_denoising_steps: int,
        context=None,
        hard_conds: Optional[dict] = None,
        n_samples: int = 1,
        return_chain: bool = False,
        horizon: Optional[int] = None,
        n_diffusion_steps_without_noise: int = 0,
        **sample_kwargs,
    ):
        del n_diffusion_steps_without_noise
        if n_samples != seed_trajectory_b.shape[0]:
            current = seed_trajectory_b[:1].repeat(n_samples, 1, 1)
        else:
            current = seed_trajectory_b.clone()
        if n_noising_steps is not None and n_noising_steps > 0:
            noise_scale = float(n_noising_steps) / float(max(self.n_diffusion_steps, 1))
            current = current + noise_scale * torch.randn_like(current)
        hard_conds = self._repeat_hard_conds(hard_conds, current.shape[0])
        current = apply_hard_conditioning(current, hard_conds)
        final, chain = self._run_rollout(
            current=current,
            total_steps=n_denoising_steps,
            context=context,
            hard_conds=hard_conds,
            return_chain=True,
            projection_info=sample_kwargs.get("dataset"),
            init_traj4proj=sample_kwargs.get("init_traj4proj"),
            proj_params=sample_kwargs.get("proj_params"),
            guide=sample_kwargs.get("guide"),
            n_guide_steps=sample_kwargs.get("n_guide_steps", 1),
            t_start_guide=sample_kwargs.get("t_start_guide", torch.inf),
            omega=sample_kwargs.get("dfm_inference_omega"),
        )
        chain = einops.rearrange(chain, "b steps h d -> steps b h d")
        if return_chain:
            return chain
        return final

    def loss(self, x: torch.Tensor, context=None, hard_conds: Optional[dict] = None):
        if context is None or not isinstance(context, dict) or "tasks" not in context or "task_ids" not in context:
            raise ValueError("DFM training requires context with 'tasks' and 'task_ids'.")

        task_context = self._encode_base_context(context)
        task_ids = _normalize_group_ids(context["task_ids"]).to(device=x.device)
        x0 = torch.randn_like(x)
        x0_groups, x1_groups, task_context_groups = _group_batch_into_task_subgroups(
            x0,
            x,
            task_context,
            task_ids,
            self.dfm_groups_per_task,
        )

        num_tasks, groups_per_task, subgroup_size = x0_groups.shape[:3]
        t_groups, r_groups = _sample_shared_group_timesteps(
            mean_t=self.dfm_P_mean_t,
            std_t=self.dfm_P_std_t,
            mean_r=self.dfm_P_mean_r,
            std_r=self.dfm_P_std_r,
            num_tasks=num_tasks,
            groups_per_task=groups_per_task,
            subgroup_size=subgroup_size,
            device=x.device,
            dtype=x.dtype,
        )
        use_omega = self.dfm_drift_form in {"split_v1", "split_v2"}
        omega_groups = (
            _sample_shared_group_omega(
                omega_min=self.dfm_omega_min,
                omega_max=self.dfm_omega_max,
                omega_exponent=self.dfm_omega_exponent,
                num_tasks=num_tasks,
                groups_per_task=groups_per_task,
                subgroup_size=subgroup_size,
                device=x.device,
                dtype=x.dtype,
            )
            if use_omega
            else None
        )

        h_groups = r_groups - t_groups
        x_t_groups = t_groups * x1_groups + (1.0 - t_groups) * x0_groups
        x_r_groups = r_groups * x1_groups + (1.0 - r_groups) * x0_groups

        x_t = _flatten_task_subgroups_to_groups(x_t_groups).reshape(-1, *x.shape[1:])
        x_r = _flatten_task_subgroups_to_groups(x_r_groups).reshape(-1, *x.shape[1:])
        base_context = _flatten_task_subgroups_to_groups(task_context_groups).reshape(-1, task_context.shape[-1])
        h = _flatten_task_subgroups_to_groups(h_groups).reshape(-1)
        omega = None if omega_groups is None else _flatten_task_subgroups_to_groups(omega_groups).reshape(-1, 1)

        if hard_conds:
            x_t = apply_hard_conditioning(x_t, hard_conds)
            x_r = apply_hard_conditioning(x_r, hard_conds)

        velocity_context = self._prepare_velocity_context(base_context, h, omega, current=x_t)
        predicted_velocity = self.model(
            x_t,
            time=_flatten_task_subgroups_to_groups(t_groups).reshape(-1),
            context=velocity_context,
        )
        x_r_pred = x_t + h[:, None, None] * predicted_velocity
        if hard_conds:
            x_r_pred = apply_hard_conditioning(x_r_pred, hard_conds)

        x_r_pred_groups = x_r_pred.reshape_as(x_t_groups)
        unc_pos_groups = (
            _build_same_time_unc_samples(x_r_groups, self.dfm_unconditional_per_group)
            if use_omega
            else None
        )
        unc_neg_groups = (
            _build_same_time_unc_samples(x_r_pred_groups.detach(), self.dfm_unconditional_per_group)
            if use_omega
            else None
        )

        gen = _flatten_group_features(_flatten_task_subgroups_to_groups(x_r_pred_groups.detach()))
        pos = _flatten_group_features(_flatten_task_subgroups_to_groups(x_r_groups))
        unc_pos = None if unc_pos_groups is None else _flatten_group_features(_flatten_task_subgroups_to_groups(unc_pos_groups))
        unc_neg = None if unc_neg_groups is None else _flatten_group_features(_flatten_task_subgroups_to_groups(unc_neg_groups))
        subgroup_omega = None if omega_groups is None else omega_groups[:, :, 0, 0].reshape(-1)

        drift = compute_grouped_sinkhorn_drift(
            gen=gen,
            pos=pos,
            unc_pos=unc_pos,
            unc_neg=unc_neg,
            drift_form=self.dfm_drift_form,
            omega=subgroup_omega,
            kernel_temp_pos=self.dfm_kernel_temp_pos,
            kernel_temp_neg=self.dfm_kernel_temp_neg,
            sinkhorn_iters=self.dfm_sinkhorn_iters,
        )
        target_x_r = (gen + drift).detach()
        target_x_r = target_x_r.reshape(_flatten_task_subgroups_to_groups(x_r_pred_groups).shape).reshape_as(x_r_pred)
        drift_loss = _adaptive_matching_loss(
            x_r_pred,
            target_x_r,
            norm_eps=self.dfm_norm_eps,
            norm_p=self.dfm_norm_p,
        )
        info = {
            "loss": float(drift_loss.detach().item()),
            "drift_loss": float(drift_loss.detach().item()),
        }
        if subgroup_omega is not None:
            info["omega_mean"] = float(subgroup_omega.detach().mean().item())
        return drift_loss, info


class FlowModelsEnsemble(nn.Module):
    def __init__(self, models: Dict[int, TrajectoryDriftFlowMatchingModel], transforms: Dict[int, torch.Tensor], **kwargs):
        super().__init__()
        self.models = models
        self.transforms = transforms
        self.n_diffusion_steps = next(iter(models.values())).n_diffusion_steps
        self.predict_epsilon = False

    @torch.no_grad()
    def warmup(self, horizon: int = 64, device: str = "cuda", context=None):
        for model in self.models.values():
            model.warmup(horizon=horizon, device=device, context=context)

    def _repeat_hard_conds(self, hard_conds: Dict[int, dict], n_samples: int) -> Dict[int, dict]:
        repeated = deepcopy(hard_conds)
        for m, c_dict in repeated.items():
            for key, value in c_dict.items():
                repeated[m][key] = einops.repeat(value, "d -> b d", b=n_samples)
        return repeated

    @torch.no_grad()
    def _run_joint_rollout(
        self,
        x: Dict[int, torch.Tensor],
        *,
        contexts: Optional[Dict[int, dict]],
        hard_conds: Dict[int, dict],
        cross_conds: Dict[Tuple[int, int], Tuple[int, int]],
        total_steps: int,
        return_chain: bool,
        sample_kwargs=None,
    ):
        if isinstance(sample_kwargs, list):
            sample_kwargs = {index: value for index, value in enumerate(sample_kwargs)}
        sample_kwargs = sample_kwargs or {}
        x = {k: apply_hard_conditioning(v, hard_conds.get(k, {})) for k, v in x.items()}
        x = apply_cross_conditioning(x, cross_conds, self.transforms)
        chains = {k: [v.clone()] for k, v in x.items()} if return_chain else None

        for step_index in range(total_steps):
            for m, model in self.models.items():
                model_kwargs = sample_kwargs.get(m, {})
                x[m] = model.sampling_step(
                    x[m],
                    step_index=step_index,
                    total_steps=total_steps,
                    context=None if contexts is None else contexts.get(m),
                    hard_conds=hard_conds.get(m, {}),
                    guide=model_kwargs.get("guide"),
                    n_guide_steps=model_kwargs.get("n_guide_steps", 1),
                    t_start_guide=model_kwargs.get("t_start_guide", torch.inf),
                    omega=model_kwargs.get("dfm_inference_omega"),
                )
                x = apply_cross_conditioning(x, cross_conds, self.transforms)
            if return_chain:
                for m, value in x.items():
                    chains[m].append(value.clone())

        if return_chain:
            return {k: torch.stack(v, dim=1) for k, v in chains.items()}
        return x

    @torch.no_grad()
    def run_inference(
        self,
        contexts: Optional[Dict[int, dict]] = None,
        hard_conds: Optional[Dict[int, dict]] = None,
        cross_conds: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        n_samples: int = 1,
        return_chain: bool = False,
        sample_kwargs: Optional[Dict[str, dict]] = None,
        **kwargs,
    ):
        hard_conds = self._repeat_hard_conds(deepcopy(hard_conds or {}), n_samples)
        cross_conds = deepcopy(cross_conds or {})
        x = {}
        for m, model in self.models.items():
            horizon = getattr(model, "horizon", None)
            if horizon is None:
                non_negative_keys = [key for key in hard_conds.get(m, {}).keys() if key >= 0]
                horizon = max(non_negative_keys) + 1 if non_negative_keys else 64
            x[m] = torch.randn((n_samples, horizon, model.state_dim), device=next(model.parameters()).device)
        chains = self._run_joint_rollout(
            x,
            contexts=contexts,
            hard_conds=hard_conds,
            cross_conds=cross_conds,
            total_steps=self.n_diffusion_steps,
            return_chain=return_chain,
            sample_kwargs=sample_kwargs,
        )
        if return_chain:
            return {k: einops.rearrange(v, "b steps h d -> steps b h d") for k, v in chains.items()}
        return chains

    @torch.no_grad()
    def run_local_inference(
        self,
        seed_trajectory_b: torch.Tensor,
        n_noising_steps: int,
        n_denoising_steps: int,
        contexts: Optional[Dict[int, dict]] = None,
        hard_conds: Optional[Dict[int, dict]] = None,
        cross_conds: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        n_samples: int = 1,
        return_chain: bool = False,
        sample_kwargs: Optional[Dict[str, dict]] = None,
        **kwargs,
    ):
        hard_conds = self._repeat_hard_conds(deepcopy(hard_conds or {}), n_samples)
        cross_conds = deepcopy(cross_conds or {})
        x = {}
        start = 0
        for m, model in self.models.items():
            horizon = getattr(model, "horizon", None)
            if horizon is None:
                non_negative_keys = [key for key in hard_conds.get(m, {}).keys() if key >= 0]
                horizon = max(non_negative_keys) + 1 if non_negative_keys else 64
            x[m] = seed_trajectory_b[:, start:start + horizon, :].clone()
            if n_noising_steps is not None and n_noising_steps > 0:
                x[m] = x[m] + (float(n_noising_steps) / float(max(model.n_diffusion_steps, 1))) * torch.randn_like(x[m])
            start += horizon
        chains = self._run_joint_rollout(
            x,
            contexts=contexts,
            hard_conds=hard_conds,
            cross_conds=cross_conds,
            total_steps=n_denoising_steps,
            return_chain=return_chain,
            sample_kwargs=sample_kwargs,
        )
        if return_chain:
            return {k: einops.rearrange(v, "b steps h d -> steps b h d") for k, v in chains.items()}
        return chains
