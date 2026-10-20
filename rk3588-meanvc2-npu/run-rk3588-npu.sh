#!/usr/bin/env bash
set -euo pipefail

root=${MEANVC2_ROOT:-/home/huangnv/MeanVC2}
threads=${MEANVC2_THREADS:-4}
cpus=${MEANVC2_CPUS:-1-4}
runtime_packages="$root/npu/.venv-runtime-2.5.0/lib/python3.12/site-packages"
runtime_library="$root/npu/runtime-2.5.0/librknnrt.so"
runtime_pythonpath="$runtime_packages${PYTHONPATH:+:$PYTHONPATH}"

exec sudo -n unshare --mount --propagation private sh -c '
  mount --bind "$1" /usr/lib/librknnrt.so
  export PYTHONPATH="$2"
  shift 2
  exec "$@"
' sh "$runtime_library" "$runtime_pythonpath" taskset -c "$cpus" "$root/.venv/bin/python" \
  "$root/npu/run_meanvc2_npu.py" \
  --meanvc2-root "$root" \
  --asr-rknn "$root/npu/artifacts/fastu2pp_160ms_flat_cache_fp16_rknn250.rknn" \
  --asr-position-table "$root/npu/artifacts/fastu2pp_160ms_position.npy" \
  --dit-rknn "$root/npu/artifacts/meanvc2_dit_streaming_flat_cache_fp16_rknn250.rknn" \
  --vocos-rknn "$root/npu/artifacts/vocos_backbone_14_fp16.rknn" \
  --threads "$threads" \
  "$@"
