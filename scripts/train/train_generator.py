#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from smd.models import build_generator_from_args
from smd.runtime import resolve_runtime_config
from smd.trainer import get_dataset, get_loss, get_summary, train
from smd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.torch_utils import get_torch_device


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Train a Diffusion-MRMP generator.")
    parser.add_argument("--config", required=True, help="Path to a training yaml config.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    runtime = resolve_runtime_config(config.get("runtime"))
    device = get_torch_device(config.get("device", "cuda"))
    tensor_args = {"device": device, "dtype": torch.float32}

    model_id = config["model_id"]
    trained_models_root = Path(config.get("trained_models_root", runtime["trained_models_root"])).resolve()
    model_dir = trained_models_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    config["runtime"] = runtime
    config["generator_family"] = config.get("generator_family", "dfm")
    config["loss_class"] = config.get(
        "loss_class",
        "DriftFlowMatchingLoss" if config["generator_family"] == "dfm" else "GaussianDiffusionLoss",
    )

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        tensor_args=tensor_args,
        results_dir=str(model_dir),
        **config,
    )
    dataset = train_subset.dataset
    model = build_generator_from_args(config, dataset, tensor_args)
    loss_class = config["loss_class"]
    loss_args = dict(config)
    loss_args.pop("loss_class", None)
    loss_fn = get_loss(loss_class, **loss_args)

    summary_class = config.get("summary_class", "SummaryTrajectoryGeneration")
    summary_args = dict(config)
    summary_args.pop("summary_class", None)
    summary_fn = get_summary(summary_class, **summary_args)

    batch_size = config.get("batch_size", 1)
    if config["generator_family"] == "dfm":
        batch_size = config.get("dfm_tasks_per_batch", 1) * config.get("dfm_trajectories_per_task", config.get("dfm_groups_per_task", 4))
    epochs = config.get("epochs")
    if epochs is None:
        epochs = get_num_epochs(config["num_train_steps"], batch_size, len(train_subset))

    with (model_dir / "args.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_subset=train_subset,
        val_subset=val_subset,
        epochs=epochs,
        lr=config["lr"],
        steps_til_summary=config.get("steps_til_summary", 100),
        steps_til_checkpoint=config.get("steps_til_checkpoint", 1000),
        model_dir=str(model_dir),
        loss_fn=loss_fn,
        val_loss_fn=loss_fn,
        summary_fn=summary_fn,
        use_ema=config.get("dfm_use_ema", config.get("use_ema", True)),
        ema_decay=config.get("dfm_ema_decay", config.get("ema_decay", 0.999)),
        debug=config.get("debug", False),
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    main()
