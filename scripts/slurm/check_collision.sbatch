#!/bin/bash
#SBATCH -A <allocation>
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/smd_check_collision_%j.bootstrap.out
#SBATCH --export=NONE

set -euo pipefail

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before submitting this job.}"
CURRENT_USER="${USER:-$(id -un)}"
source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/runtime_env.sh"
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SMD_ENV_ROOT}"

mkdir -p "${SMD_RUNS_ROOT}/slurm"

LOG_PATH="${SMD_RUNS_ROOT}/slurm/check_collision_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ -n "${SMD_COLLISION_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python "${SMD_PROJECT_ROOT}/is_collision.py" ${SMD_COLLISION_ARGS}
else
  python "${SMD_PROJECT_ROOT}/is_collision.py"
fi
