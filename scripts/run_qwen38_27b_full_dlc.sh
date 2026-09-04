#!/usr/bin/env bash
set -euo pipefail

export NGA_MODEL_SCALE=full
export NGA_MICRO_BATCH_SIZE="${NGA_MICRO_BATCH_SIZE:-1}"
export NGA_GLOBAL_BATCH_SIZE="${NGA_GLOBAL_BATCH_SIZE:-512}"
export NGA_PREFLIGHT_STEPS="${NGA_PREFLIGHT_STEPS:-5}"
export NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/qwen38-27b-full-bf16-fp32muon-fineweb100b-seed42}"

exec bash "$(dirname "$0")/run_qwen38_27b_quarter_dlc.sh"
