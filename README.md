# Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models

**[Paper](https://arxiv.org/pdf/2502.03607)** | **[Project Page](https://multi-robot-constrained-diffusion.github.io/)** | **[arXiv](https://arxiv.org/abs/2502.03607)**

Official implementation of "Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models"



## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)



## Overview

This repository contains the official implementation of our method for **simultaneous multi-robot motion planning with projected diffusion models**. Our approach enables efficient and collision-free path planning for multiple robots operating in shared environments.

### Key Features

- **Scalable**: Handles 9 robots simultaneously
- **Efficient**: Leverages diffusion models with projection-based constraints
- **Flexible**: Works across various environment configurations (empty, basic, dense, rooms, shelf.)
- **Generalizable**: Handles unseen scenarios



## Installation

### System Requirements

- **OS**: Rocky Linux 8.10 (tested), other Linux distributions should work
- **Python**: 3.8.20
- **CUDA**: 12.1 (for GPU acceleration)



### Setup Instructions

1. **Create and activate the conda environment:**

```bash
conda create -n smd python=3.8.20
conda activate smd
```

2. **Install core dependencies:**

```bash
conda install patchelf
pip install setuptools==70.2.0
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. **Install project dependencies:**

```bash
# Install torch_robotics
cd deps/torch_robotics
pip install -e .

# Install experiment_launcher
cd ../experiment_launcher
pip install -e .

# Install motion_planning_baselines
cd ../motion_planning_baselines
pip install -e .

# Install the main package
cd ../..
pip install -e .
```

4. **Install IPOPT optimizer:**

```bash
conda install -c conda-forge ipopt
```

5. **Download pre-trained models and data:**

Download the data and checkpoints from [Google Drive](https://drive.google.com/file/d/1M0hfM5TlY45mzMxoyaeslvohSC2e74Yk/view?usp=sharing) and extract:

```bash
tar -xzvf data_checkpoints.tar.gz
```



## Quick Start

Run inference with pre-trained models:

```bash
cd scripts/inference/
python launch_smd_composite_experiment.py
cd ../..
python is_collision.py
```

This will:

1. Load pre-trained models for multi-robot scenarios
2. Generate motion plans for test instances
3. Evaluate collision metrics

## Drift Flow Matching Integration

This checkout now supports a **DFM trajectory generator** alongside the original diffusion generator. The planner,
projection, collision checking, and result-directory behavior stay unchanged; only the learned trajectory prior is
swappable.

### New generator surface

- `generator_family`: `diffusion` or `dfm`
- `generator_model_class`: generator implementation class
- `generator_rollout_steps`: sampling rollout steps
- `dfm_*`: Drift Flow Matching hyperparameters and batching controls

For DFM training, use one of the example configs in `configs/experiments/`
and run:

```bash
python scripts/train/train_generator.py --config configs/experiments/composite_three_dfm.yaml
```

The generated checkpoint layout remains:

```text
data_trained_models/<model_id>/checkpoints/ema_model_current_state_dict.pth
```

so `SMDComposite`, `SMD`, and `SMDEnsemble` can keep loading trained models through the existing planner code paths.

## UVA Runtime

The integrated repo now carries a UVA-first runtime contract with:

- `runtime.project_name`
- `/home/$USER/{project_name}`
- `/scratch/$USER/{project_name}/data`
- `/scratch/$USER/{project_name}/runs`
- `/home/$USER/envs/{project_name}`

Helper scripts:

```bash
export SMD_PROJECT_NAME=Diffusion_MRMP_DFM
bash scripts/uva/bootstrap_home_checkout.sh
bash scripts/uva/setup_miniforge_env.sh
sbatch --export=SMD_PROJECT_NAME="$SMD_PROJECT_NAME",SMD_CONFIG=/home/$USER/$SMD_PROJECT_NAME/configs/experiments/composite_three_dfm.yaml /home/$USER/$SMD_PROJECT_NAME/scripts/slurm/train_generator.sbatch
sbatch --export=SMD_PROJECT_NAME="$SMD_PROJECT_NAME" /home/$USER/$SMD_PROJECT_NAME/scripts/slurm/inference_composite.sbatch
sbatch --export=SMD_PROJECT_NAME="$SMD_PROJECT_NAME" /home/$USER/$SMD_PROJECT_NAME/scripts/slurm/check_collision.sbatch
```



## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{liang2025simultaneous,
  author    = {Liang, Jinhao and Christopher, Jacob K. and Koenig, Sven and Fioretto, Ferdinando},
  title     = {Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models},
  journal   = {Forty-second International Conference on Machine Learning},
  year      = {2025},
}
```



## Acknowledgments

This codebase is built upon [MMD (Multi-Robot Motion Planning with Diffusion Models)](https://github.com/yoraish/mmd) by Shaoul et al. We thank the authors for providing a robust foundation for multi-robot motion planning research.

We also acknowledge the contributions from [MPD (Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models)](https://github.com/jacarvalho/mpd-public) for foundational work on motion planning with diffusion models.



## Contact

For questions or issues regarding this implementation, please contact:

**Jinhao Liang**  

Email: [jliang@email.virginia.edu](mailto:jliang@email.virginia.edu)

Alternatively, feel free to open an issue on GitHub.
