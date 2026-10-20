# RK3588 MeanVC2 NPU work

This directory contains reproducible tools used to validate the RK3588 NPU
before moving individual MeanVC2 stages from PyTorch CPU inference to RKNN.

The validated board setup uses RKNN Toolkit2/Lite2 2.3.2 and installs the
matching `librknnrt.so` in `/usr/lib`. The current kernel exposes RKNPU driver
0.8.2; runtime 2.3.2 works for the probes below while recommending driver
0.8.8 or newer.

## NPU smoke test

`npu_smoke.py` has three explicit stages:

1. `make-onnx` creates a deterministic elementwise ONNX model.
2. `convert` converts it for RK3588 with RKNN Toolkit2 in FP16 mode.
3. `run` executes it through RKNN Toolkit Lite2 on all NPU cores, benchmarks
   repeated inference, and compares the NPU output with NumPy.

The conversion environment and board runtime environment are kept separate so
the existing MeanVC2 PyTorch CPU environment remains reproducible.

## MeanVC2 DiT compatibility probe

`export_meanvc2_dit_onnx.py` exports one fixed-shape, stateless 120 ms DiT
step. `convert_meanvc2_dit_rknn.py` converts it to RKNN FP16, and
`test_meanvc2_dit_rknn.py` benchmarks the result and compares it with the
PyTorch fixture. This probe deliberately excludes streaming KV-cache state;
it establishes operator coverage and numerical quality before integrating an
NPU runner into the complete streaming pipeline.

## Vocos hybrid probe

`export_vocos_frontend_onnx.py` exports the convolutional Vocos backbone.
`test_vocos_rknn.py` runs that part on the RK3588 NPU, validates it against
TorchScript, then performs the spectral projection and complex ISTFT on the
CPU and compares the final waveform. Keeping the small projection and ISTFT
on the CPU avoids complex-number operators that RKNN does not support.

`run_meanvc2_vocos_npu.py` integrates the 14-frame RKNN backbone into the
120 ms MeanVC2 file pipeline. The first 12-frame Vocos call stays on the CPU;
subsequent steady-state calls use the NPU, followed by the original projection,
ISTFT, and overlap-add code on the CPU.

Run the validated file pipeline on the board with:

```bash
cd /home/huangnv/MeanVC2
./npu/run-rk3588-vocos-npu.sh \
  --target-emb test_audio/target_front_center.npy \
  --input test_audio/input_test.wav \
  --output test_audio/output_front_center_vocos_npu.wav \
  --seed 42
```

For the 5.79 second fixture, the 4-big-core CPU path measured RTF 1.550. The
hybrid Vocos-NPU path measured RTF 1.411, with 46 steady-state Vocos calls on
the NPU and one initial call on the CPU.

The stateless DiT graph exports to ONNX and matches PyTorch, but Toolkit2 2.3.2
on ARM64 currently exits during RKNN code generation for that complete graph.
No DiT `.rknn` is treated as deployable until conversion and numerical checks
both pass.
