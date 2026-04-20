#!/bin/bash
#SBATCH -A <allocation>
#SBATCH --partition=gpu
#SBATCH --constraint=a40|a6000|a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/smd_inference_composite_%j.bootstrap.out
#SBATCH --export=NONE

set -euo pipefail

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before submitting this job.}"
CURRENT_USER="${USER:-$(id -un)}"
source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/runtime_env.sh"
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SMD_ENV_ROOT}"

mkdir -p "${SMD_DATA_ROOT}" "${SMD_RUNS_ROOT}" "${SMD_RUNS_ROOT}/slurm"

LOG_PATH="${SMD_RUNS_ROOT}/slurm/inference_composite_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "${LOG_PATH}") 2>&1

python "${SMD_PROJECT_ROOT}/scripts/inference/launch_smd_composite_experiment.py"
