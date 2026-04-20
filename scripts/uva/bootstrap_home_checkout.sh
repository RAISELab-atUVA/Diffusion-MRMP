#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before running this script.}"
source "${SCRIPT_DIR}/runtime_env.sh"

SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_ROOT="${SMD_PROJECT_ROOT}"

mkdir -p "${TARGET_ROOT}"

rsync -a \
  --exclude ".git/" \
  --exclude ".pytest_cache/" \
  --exclude "__pycache__/" \
  --exclude ".venv/" \
  --exclude "data/" \
  --exclude "runs/" \
  "${SOURCE_ROOT}/" "${TARGET_ROOT}/"

echo "Synchronized ${SOURCE_ROOT} -> ${TARGET_ROOT}"
