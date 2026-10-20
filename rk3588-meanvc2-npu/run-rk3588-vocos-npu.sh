#!/usr/bin/env bash
set -euo pipefail

root=${MEANVC2_ROOT:-/home/huangnv/MeanVC2}
threads=${MEANVC2_THREADS:-4}
cpus=${MEANVC2_CPUS:-1-4}
runtime_packages="$root/npu/.venv-runtime/lib/python3.12/site-packages"

export PYTHONPATH="$runtime_packages${PYTHONPATH:+:$PYTHONPATH}"

exec taskset -c "$cpus" "$root/.venv/bin/python" \
  "$root/npu/run_meanvc2_vocos_npu.py" \
  --meanvc2-root "$root" \
  --vocos-rknn "$root/npu/artifacts/vocos_backbone_14_fp16.rknn" \
  --threads "$threads" \
  "$@"
