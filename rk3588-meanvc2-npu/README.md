# RK3588 MeanVC2 NPU work

This directory contains reproducible tools used to validate the RK3588 NPU
before moving individual MeanVC2 stages from PyTorch CPU inference to RKNN.

The board keeps the installed RKNN 2.3.2 runtime intact. Complete Transformer
graphs are converted with RKNN Toolkit2 2.5.0 and run through an isolated
Lite2 2.5.0 mount namespace. The current kernel exposes RKNPU driver 0.8.2;
the runtime works for the tests below while recommending driver 0.8.8 or
newer.

## NPU smoke test

`npu_smoke.py` has three explicit stages:

1. `make-onnx` creates a deterministic elementwise ONNX model.
2. `convert` converts it for RK3588 with RKNN Toolkit2 in FP16 mode.
3. `run` executes it through RKNN Toolkit Lite2 on all NPU cores, benchmarks
   repeated inference, and compares the NPU output with NumPy.

The conversion environment and board runtime environment are kept separate so
the existing MeanVC2 PyTorch CPU environment remains reproducible.

## Fast-U2++ ASR

`export_fastu2pp_onnx.py` exports the fixed 120 ms streaming encoder.
`rewrite_fastu2pp_static_onnx.py` exposes the 12-frame positional encoding as
an input, freezes fixed Slice parameters, and flattens the attention and
convolution caches to three dimensions. Flattening is required because RKNN's
automatic NCHW/NHWC conversion corrupts four-dimensional Transformer cache
state on this graph.

The resulting `fastu2pp_160ms_flat_cache_fp16_rknn250.rknn` model preserves
streaming state over consecutive calls. Eight-step validation measured a
worst bottleneck cosine similarity of 0.99586 and about 12.0 ms per NPU call,
compared with about 92 ms per four-big-core CPU call.

## MeanVC2 streaming DiT

`export_meanvc2_dit_streaming_onnx.py` exports the steady-state 120 ms DiT
step with its four KV caches. The first three chunks run on the CPU to grow
the caches to fixed sizes; later chunks use the RKNN graph. Cache tensors are
flattened to three dimensions at the RKNN boundary and restored inside the
graph, avoiding the same four-dimensional layout conversion issue found in
the ASR model.

The steady-state graph runs in about 20.5 ms per ODE call. Twelve consecutive
cache updates kept velocity cosine similarity between 0.99196 and 0.99574.
The runner exposes zero, one, or two DiT NPU ODE steps per chunk so latency
and numerical fidelity can be compared on real audio.

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

## Combined file pipeline

Run the balanced pipeline on the board with:

```bash
cd /home/huangnv/MeanVC2
./npu/run-rk3588-npu.sh \
  --target-emb test_audio/target_front_center.npy \
  --input test_audio/input_test.wav \
  --output test_audio/output_front_center_balanced_npu.wav \
  --seed 42
```

The default uses NPU ASR, one NPU DiT ODE step followed by one CPU correction
step, and the NPU Vocos backbone. Select another mode with:

```bash
# Fidelity-oriented: ASR and Vocos NPU, DiT CPU
./npu/run-rk3588-npu.sh --dit-npu-steps 0 ...

# Fastest: both DiT ODE steps on NPU
./npu/run-rk3588-npu.sh --dit-npu-steps 2 ...

# ASR CPU comparison
./npu/run-rk3588-npu.sh --asr-device cpu ...
```

For the 5.79 second fixture on the four big CPU cores:

| Pipeline | Elapsed | RTF |
| --- | ---: | ---: |
| CPU baseline | 8.97 s | 1.550 |
| Vocos NPU only | 8.17 s | 1.411 |
| ASR + Vocos NPU, DiT CPU | 6.4 s | 1.100 |
| Balanced, one DiT NPU step | 5.0 s | 0.871 |
| Fastest, two DiT NPU steps | 4.1 s | 0.704 |

All generated waveforms were finite and unclipped. The balanced and fastest
modes intentionally trade some numerical fidelity for latency; listen to the
fixture outputs before choosing a production default. Driver 0.8.2 remains an
open platform limitation because Toolkit2/Lite2 2.5.0 recommends 0.8.8 or
newer.
