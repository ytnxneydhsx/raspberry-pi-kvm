# RK3588 MeanVC2 NPU work

This directory contains reproducible tools used to validate the RK3588 NPU
before moving individual MeanVC2 stages from PyTorch CPU inference to RKNN.

The board keeps its system RKNN runtime intact. All production ASR, DiT, and
Vocos RKNN graphs are converted with RKNN Toolkit2 2.5.0 and run through an
isolated Lite2 2.5.0 mount namespace. The current kernel exposes RKNPU driver
0.8.2; the runtime works for the tests below while recommending driver 0.8.8
or newer.

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

`export_meanvc2_dit_startup_onnx.py` exports three fixed startup graphs: empty
KV cache, 12-frame cache, and 24-frame cache. They preserve the real cache
history from the first chunk onward. `export_meanvc2_dit_streaming_onnx.py`
exports the steady-state graph with cache lengths 36/36/24/24. Cache tensors
are flattened to three dimensions at every RKNN boundary and restored inside
the graph, avoiding the same four-dimensional layout conversion issue found
in the ASR model.

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
120 ms MeanVC2 file pipeline. The combined runner additionally loads a
12-frame startup RKNN backbone, so the first and steady-state Vocos neural
calls both use the NPU. The original projection, ISTFT, and overlap-add remain
on the CPU.

Both Vocos backbones now use Toolkit2 2.5.0. The steady 14-frame model is
`vocos_backbone_14_fp16_rknn250.rknn` with SHA256
`852b6c8d640ba7414728754b2739b9f6b9a76aadc166fca734e957da36e36a0d`.
On the board, 50 repeated calls averaged 4.058 ms, spectral cosine similarity
was 0.99997580, and all hidden, spectral, and waveform outputs were finite. A
full 5.79-second regression run measured 222.593 ms cold-start first output,
74.724 ms median / 129.160 ms P95 steady packet processing, and 0.639702 RTF.
It used 48 Vocos NPU calls with zero fallback, and its waveform was sample-for-
sample identical to the previously accepted quality-path output.

## Combined file pipeline

Run the quality-protected low-latency pipeline on the board with:

```bash
cd /home/huangnv/MeanVC2
./npu/run-rk3588-npu.sh \
  --target-emb test_audio/target_front_center.npy \
  --input test_audio/input_test.wav \
  --output test_audio/output_front_center_balanced_npu.wav \
  --seed 42
```

The default uses NPU ASR, an immediate empty-cache NPU DiT first chunk, two
CPU DiT chunks to establish numerically accurate history, steady-state NPU
DiT, and NPU Vocos. Select another mode with:

```bash
# Fidelity-oriented: ASR and Vocos NPU, DiT CPU
./npu/run-rk3588-npu.sh --dit-npu-steps 0 ...

# Balanced numerical path: one DiT NPU step and one CPU correction
./npu/run-rk3588-npu.sh --dit-npu-steps 1 ...

# Experimental strict NPU DiT startup (lower first-syllable fidelity)
./npu/run-rk3588-npu.sh --dit-startup-policy all-npu ...

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

## Cold-start warm-up

The 120 ms model receives 160 ms (2560 samples) per input packet. A normal
Kaldi fbank call produces 14 real frames from the first packet, while the
Fast-U2++ graph needs a 19-frame window. The runner initializes `frame_cache`
with five genuine Kaldi silence-feature frames exactly once. The first ASR
window is therefore 5 silence + 14 real frames. With stride 16 it retains the
last three real frames; the second packet adds 16 new frames, so the next
window is continuous and does not duplicate or drop a real frame. ASR offset
advances once per actual encoder call (8 -> 12 -> 16).

The empty/12/24-frame RKNN graphs make a strict NPU startup available. FP16
errors in the growing cache graphs audibly affect the first syllable, so the
default protects quality: the empty-cache graph emits the first chunk on NPU,
its exact KV state is rebuilt between packets, the next two chunks grow that
state on CPU, and the fourth chunk switches to the steady RKNN graph. Runtime
contexts are primed before audio starts. The first waveform boundary retains
the existing 160-sample (10 ms) overlap crossfade.

Three repeated default runs of the 5.79 second fixture measured a median
estimated first-output latency of 215.2 ms (160 ms input plus 55.2 ms
processing), steady 160 ms packet processing of 75.2 ms median / 130.4 ms P95,
and median RTF 0.642. The first 200 ms of voiced output, aligned against the
same warm-start path with CPU DiT, measured correlation 0.974 and SNR 12.08 dB.
The experimental strict-NPU startup reached about 216 ms first output, while
first-200-ms correlation fell to roughly 0.56 even with cache repair; it is
kept as a diagnostic mode rather than the production default.

All generated waveforms were finite and unclipped. The balanced and fastest
modes intentionally trade some numerical fidelity for latency; listen to the
fixture outputs before choosing a production default. Driver 0.8.2 remains an
open platform limitation because Toolkit2/Lite2 2.5.0 recommends 0.8.8 or
newer.
