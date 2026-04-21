#!/bin/bash

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before sourcing scripts/uva/runtime_env.sh.}"

CURRENT_USER="${USER:-$(id -un)}"

export SMD_PROJECT_ROOT="/home/${CURRENT_USER}/${SMD_PROJECT_NAME}"
export SMD_DATA_ROOT="/scratch/${CURRENT_USER}/${SMD_PROJECT_NAME}/data"
export SMD_RUNS_ROOT="/scratch/${CURRENT_USER}/${SMD_PROJECT_NAME}/runs"
export SMD_ENV_ROOT="/home/${CURRENT_USER}/envs/${SMD_PROJECT_NAME}"
export SMD_TRAJECTORIES_ROOT="${SMD_DATA_ROOT}/data_trajectories"
export SMD_TRAINED_MODELS_ROOT="${SMD_DATA_ROOT}/data_trained_models"
export SMD_INSTANCES_ROOT="${SMD_DATA_ROOT}/instances_data"
export SMD_INIT4PROJ_ROOT="${SMD_DATA_ROOT}/init4proj_data"
export SMD_EXPERIMENTS_ROOT="${SMD_RUNS_ROOT}/experiments"
