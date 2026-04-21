#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before running this script.}"
source "${SCRIPT_DIR}/runtime_env.sh"

module purge || true
module load miniforge || true

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "${SMD_ENV_ROOT}" ]]; then
  conda create -y -p "${SMD_ENV_ROOT}" python=3.10
fi

conda activate "${SMD_ENV_ROOT}"
pip install setuptools==70.2.0
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
conda install -y -c conda-forge ipopt
pip install -r "${SMD_PROJECT_ROOT}/requirements.txt"
pip install -e "${SMD_PROJECT_ROOT}/deps/torch_robotics"
pip install -e "${SMD_PROJECT_ROOT}/deps/experiment_launcher"
pip install -e "${SMD_PROJECT_ROOT}/deps/motion_planning_baselines"
pip install -e "${SMD_PROJECT_ROOT}"
