#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before running this script.}"
: "${SMD_GIT_URL:?Set SMD_GIT_URL to the GitHub clone URL before running this script.}"
source "${SCRIPT_DIR}/runtime_env.sh"

TARGET_ROOT="${SMD_PROJECT_ROOT}"
TARGET_REF="${SMD_GIT_REF:-}"

mkdir -p "$(dirname "${TARGET_ROOT}")"

if [[ -d "${TARGET_ROOT}/.git" ]]; then
  git -C "${TARGET_ROOT}" remote set-url origin "${SMD_GIT_URL}"
  git -C "${TARGET_ROOT}" fetch origin --prune
else
  if [[ -e "${TARGET_ROOT}" && ! -d "${TARGET_ROOT}/.git" ]]; then
    echo "Refusing to reuse non-git directory at ${TARGET_ROOT}" >&2
    exit 1
  fi
  git clone "${SMD_GIT_URL}" "${TARGET_ROOT}"
fi

if [[ -n "${TARGET_REF}" ]]; then
  git -C "${TARGET_ROOT}" fetch origin "${TARGET_REF}" || true
  if git -C "${TARGET_ROOT}" show-ref --verify --quiet "refs/remotes/origin/${TARGET_REF}"; then
    git -C "${TARGET_ROOT}" checkout -B "${TARGET_REF}" "origin/${TARGET_REF}"
  else
    git -C "${TARGET_ROOT}" checkout "${TARGET_REF}"
  fi
fi

echo "Prepared git checkout at ${TARGET_ROOT}"
