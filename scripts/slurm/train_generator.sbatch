#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=a40|a6000|a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/smd_train_generator_%j.bootstrap.out
#SBATCH --export=NONE

set -euo pipefail

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before submitting this job.}"
CURRENT_USER="${USER:-$(id -un)}"
source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/runtime_env.sh"
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SMD_ENV_ROOT}"

mkdir -p \
  "${SMD_DATA_ROOT}" \
  "${SMD_TRAINED_MODELS_ROOT}" \
  "${SMD_RUNS_ROOT}" \
  "${SMD_RUNS_ROOT}/slurm"

LOG_PATH="${SMD_RUNS_ROOT}/slurm/train_generator_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "${LOG_PATH}") 2>&1

CONFIG_PATH="${SMD_CONFIG:?Set SMD_CONFIG to a training yaml before submission.}"
python "${SMD_PROJECT_ROOT}/scripts/train/train_generator.py" --config "${CONFIG_PATH}"
