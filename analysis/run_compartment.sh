#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/opt/miniconda3/envs/hicinterpolate/bin/python}"

"${PYTHON}" "${SCRIPT_DIR}/run_compartment.py" "$@"
