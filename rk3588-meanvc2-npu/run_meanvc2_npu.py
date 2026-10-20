#!/usr/bin/env python3
"""Run MeanVC2 with steady-state DiT and Vocos stages on the RK3588 NPU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rknnlite.api import RKNNLite
from torchaudio.compliance import kaldi

from run_meanvc2_vocos_npu import VocosNpuMixin, load_meanvc_runtime


class AsrNpuMixin:
    """Run the fixed 120 ms Fast-U2++ streaming encoder on RKNN."""

    @staticmethod
    def _make_silence_fbanks(frame_count: int) -> torch.Tensor:
        """Create real Kaldi silence features on the same 10 ms frame grid."""
        sample_count = 400 + max(0, frame_count - 1) * 160
        silence = torch.zeros(1, sample_count, dtype=torch.float32)
        return kaldi.fbank(
            silence,
            frame_length=25,
            frame_shift=10,
            snip_edges=True,
            num_mel_bins=80,
            energy_floor=0.0,
            dither=0.0,
            sample_frequency=16000,
        )

    def _init_cache(self) -> None:
        super()._init_cache()
        warmup_frames = getattr(self, "asr_warmup_frames", 5)
        if self._bn_window == 19 and warmup_frames:
            self.frame_cache = self._make_silence_fbanks(warmup_frames)
        self.asr_warmup_pending = bool(warmup_frames)
        self.asr_first_real_frames = 0
        self.asr_first_history_frames = 0
        self.asr_packet_trace: list[dict[str, int]] = []
        reset_dit = getattr(self, "_reset_dit_cold_start_state", None)
        if reset_dit is not None:
            reset_dit()

    def init_asr_npu(
        self,
        model_path: Path,
        position_table_path: Path,
        enabled: bool,
        warmup_frames: int,
        runtime_warmup: bool,
    ) -> None:
        self.asr_warmup_frames = warmup_frames
        self.asr_npu_enabled = enabled
        self.asr_npu_calls = 0
        self.asr_npu_seconds = 0.0
        self.asr_runtime_warmup_seconds = 0.0
        if not enabled:
            self.asr_npu = None
            return
        self.asr_npu = RKNNLite(verbose=False)
        if self.asr_npu.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"failed to load ASR RKNN model: {model_path}")
        if self.asr_npu.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
            raise RuntimeError("failed to initialize ASR RKNN runtime")
        self.asr_position_table = np.load(position_table_path).astype(np.float32)
        if runtime_warmup:
            started = time.perf_counter()
            self.asr_npu.inference(
                inputs=[
                    self._make_silence_fbanks(19).unsqueeze(0).numpy(),
                    np.zeros((1, 192, 128), dtype=np.float32),
                    np.zeros((1, 1536, 8), dtype=np.float32),
                    self.asr_position_table[:, 8:20, :],
                ]
            )
            self.asr_runtime_warmup_seconds = time.perf_counter() - started

    def close_asr_npu(self) -> None:
        if getattr(self, "asr_npu", None) is not None:
            self.asr_npu.release()
            self.asr_npu = None

    def _encode_chunk(self, samples: np.ndarray) -> torch.Tensor | None:
        if not self.asr_npu_enabled:
            return super()._encode_chunk(samples)
        with torch.no_grad():
            padded = (
                np.concatenate((self.samples_cache, samples))
                if self.samples_cache.size
                else samples
            )
            if len(padded) < 400:
                self.samples_cache = padded
                return None

            wav_t = torch.from_numpy(padded * (1 << 15)).unsqueeze(0)
            fbanks = kaldi.fbank(
                wav_t,
                frame_length=25,
                frame_shift=10,
                snip_edges=True,
                num_mel_bins=80,
                energy_floor=0.0,
                dither=0.0,
                sample_frequency=16000,
            )
            real_frame_count = int(fbanks.shape[0])
            self.samples_cache = padded[160 * fbanks.shape[0] :]
            cached_frame_count = 0
            if self.frame_cache is not None:
                cached_frame_count = int(self.frame_cache.shape[0])
                fbanks = torch.cat([self.frame_cache, fbanks], dim=0)

            if self.asr_offset >= 4000:
                cache_frames = self.asr_att_cache.size(2)
                self.asr_offset = max(cache_frames, self.asr_offset - 4000)

            bns = []
            index = 0
            calls_before = self.asr_npu_calls
            while index + self._bn_window <= fbanks.shape[0]:
                fbank = fbanks[index : index + self._bn_window].unsqueeze(0)
                started = time.perf_counter()
                output, attention_cache, convolution_cache = self.asr_npu.inference(
                    inputs=[
                        fbank.numpy().astype(np.float32),
                        self.asr_att_cache.numpy().astype(np.float32).reshape(1, 192, 128),
                        self.asr_cnn_cache.numpy().astype(np.float32).reshape(1, 1536, 8),
                        self.asr_position_table[
                            :, self.asr_offset : self.asr_offset + 12, :
                        ],
                    ]
                )
                self.asr_npu_seconds += time.perf_counter() - started
                self.asr_npu_calls += 1
                self.asr_att_cache = torch.from_numpy(
                    attention_cache.astype(np.float32, copy=False).reshape(6, 4, 8, 128).copy()
                )
                self.asr_cnn_cache = torch.from_numpy(
                    convolution_cache.astype(np.float32, copy=False).reshape(6, 1, 256, 8).copy()
                )
                bns.append(torch.from_numpy(output.astype(np.float32, copy=False)).squeeze(0))
                self.asr_offset += self._asr_offset_step
                index += self._bn_stride
            self.frame_cache = fbanks[index:]
            if len(self.asr_packet_trace) < 2:
                self.asr_packet_trace.append(
                    {
                        "packet": len(self.asr_packet_trace) + 1,
                        "cached_frames_before": cached_frame_count,
                        "real_frames": real_frame_count,
                        "window_frames": int(fbanks.shape[0]),
                        "asr_calls": self.asr_npu_calls - calls_before,
                        "cached_frames_after": int(self.frame_cache.shape[0]),
                        "offset_after": int(self.asr_offset),
                    }
                )
            if self.asr_warmup_pending and bns:
                self.asr_first_real_frames = real_frame_count
                self.asr_first_history_frames = int(self.frame_cache.shape[0])
                self.asr_warmup_pending = False
            if not bns:
                return None

            bn = torch.cat(bns, dim=0).unsqueeze(0)
            if self.encoder_output_cache is not None:
                bn = torch.cat([self.encoder_output_cache, bn], dim=1)
            self.encoder_output_cache = bn[:, -1:, :]
            if bn.shape[1] < 2:
                return None
            bn_up = F.interpolate(
                bn.transpose(1, 2), size=17, mode="linear", align_corners=True
            ).transpose(1, 2)
            return bn_up[:, 1:, :]


class DiTSteadyNpuMixin:
    """Run empty, growing, and steady DiT KV-cache stages with fixed RKNN graphs."""

    def init_dit_npu(
        self,
        model_path: Path,
        startup_model_paths: list[Path],
        npu_ode_steps: int,
        startup_policy: str,
        runtime_warmup: bool,
    ) -> None:
        self.dit_npu = RKNNLite(verbose=False)
        if self.dit_npu.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"failed to load DiT RKNN model: {model_path}")
        if self.dit_npu.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
            raise RuntimeError("failed to initialize DiT RKNN runtime")
        self.dit_startup_npus = []
        for stage, startup_model_path in enumerate(startup_model_paths):
            runtime = RKNNLite(verbose=False)
            if runtime.load_rknn(str(startup_model_path)) != 0:
                raise RuntimeError(
                    f"failed to load DiT startup stage {stage}: {startup_model_path}"
                )
            if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
                raise RuntimeError(
                    f"failed to initialize DiT startup stage {stage}"
                )
            self.dit_startup_npus.append(runtime)
        self.dit_npu_calls = 0
        self.dit_npu_chunks = 0
        self.dit_cpu_chunks = 0
        self.dit_npu_seconds = 0.0
        self.dit_cpu_seconds = 0.0
        self.dit_cpu_ode_calls = 0
        self.dit_npu_ode_steps = npu_ode_steps
        self.dit_startup_policy = startup_policy
        self.dit_startup_chunks = [0, 0, 0]
        self.dit_cache_repair_seconds = 0.0
        self.dit_cache_repair_wait_seconds = 0.0
        self.dit_cache_repair_chunks = 0
        self._dit_cache_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="meanvc2-kv-repair"
        )
        self._reset_dit_cold_start_state()
        self.dit_runtime_warmup_seconds = 0.0
        if runtime_warmup:
            self._warmup_dit_runtimes()

    def _warmup_dit_runtimes(self) -> None:
        started = time.perf_counter()
        x = np.zeros((1, 16, 80), dtype=np.float32)
        t = np.asarray([1.0], dtype=np.float32)
        r = np.asarray([0.5], dtype=np.float32)
        cond = np.zeros((1, 16, 256), dtype=np.float32)
        speaker = self.vc_spk_emb.detach().cpu().numpy().astype(np.float32)
        gtm_key = self.vc_gtm_kv[0].detach().cpu().numpy().astype(np.float32)
        gtm_value = self.vc_gtm_kv[1].detach().cpu().numpy().astype(np.float32)
        for stage, runtime in enumerate(self.dit_startup_npus):
            rope = (
                self.vc.rotary_embed.forward_from_seq_len(stage * 12 + 16)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            caches = []
            if stage:
                flat_length = stage * 24
                for _ in range(4):
                    caches.extend(
                        [
                            np.zeros((1, flat_length, 64), dtype=np.float32),
                            np.zeros((1, flat_length, 64), dtype=np.float32),
                        ]
                    )
            runtime.inference(
                inputs=[x, t, r, cond, speaker, gtm_key, gtm_value, rope, *caches]
            )
        rope = (
            self.vc.rotary_embed.forward_from_seq_len(52)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        steady_caches = []
        for layer in range(4):
            flat_length = 72 if layer < 2 else 48
            steady_caches.extend(
                [
                    np.zeros((1, flat_length, 64), dtype=np.float32),
                    np.zeros((1, flat_length, 64), dtype=np.float32),
                ]
            )
        self.dit_npu.inference(
            inputs=[
                x,
                t,
                r,
                cond,
                speaker,
                gtm_key,
                gtm_value,
                rope,
                *steady_caches,
            ]
        )
        self.dit_runtime_warmup_seconds = time.perf_counter() - started

    def close_dit_npu(self) -> None:
        self._adopt_repaired_cache()
        executor = getattr(self, "_dit_cache_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._dit_cache_executor = None
        if getattr(self, "dit_npu", None) is not None:
            self.dit_npu.release()
            self.dit_npu = None
        for runtime in getattr(self, "dit_startup_npus", []):
            runtime.release()
        self.dit_startup_npus = []

    def _reset_dit_cold_start_state(self) -> None:
        future = getattr(self, "_dit_cache_repair_future", None)
        if future is not None and not future.done():
            future.result()
        self._dit_cache_repair_future: Future | None = None
        self._dit_cache_repair_payload = None
        self.stream_input_samples = 0
        self.chunk_process_ms: list[float] = []
        self.chunk_produced_output: list[bool] = []
        self.first_output_after_samples = 0
        self.first_output_compute_ms = 0.0
        self.first_output_samples = 0

    def _repair_startup_cache(self, payload):
        cond, ode_inputs, pre_kv_cache, offset = payload
        started = time.perf_counter()
        repaired_cache = None
        with torch.no_grad():
            for x, t_value, r_value in ode_inputs:
                _, repaired_cache = self.vc(
                    x,
                    torch.tensor([t_value], dtype=x.dtype),
                    torch.tensor([r_value], dtype=x.dtype),
                    cache=None,
                    cond=cond,
                    spks=self.vc_spk_emb,
                    offset=offset,
                    is_inference=True,
                    kv_cache=pre_kv_cache,
                    gtm_kv=self.vc_gtm_kv,
                )
        self.dit_cache_repair_seconds += time.perf_counter() - started
        return repaired_cache

    def _schedule_cache_repair(self) -> None:
        if self._dit_cache_repair_payload is None:
            return
        if self._dit_cache_repair_future is not None:
            raise RuntimeError("DiT cache repair was scheduled twice")
        payload = self._dit_cache_repair_payload
        self._dit_cache_repair_payload = None
        self._dit_cache_repair_future = self._dit_cache_executor.submit(
            self._repair_startup_cache, payload
        )

    def _adopt_repaired_cache(self) -> None:
        future = getattr(self, "_dit_cache_repair_future", None)
        if future is None:
            return
        started = time.perf_counter()
        self.vc_kv_cache = future.result()
        self.dit_cache_repair_wait_seconds += time.perf_counter() - started
        self.dit_cache_repair_chunks += 1
        self._dit_cache_repair_future = None

    def process_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        self.stream_input_samples += len(samples)
        started = time.perf_counter()
        output = super().process_chunk(samples)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.chunk_process_ms.append(elapsed_ms)
        self.chunk_produced_output.append(output is not None)
        if output is not None and not self.first_output_after_samples:
            self.first_output_after_samples = self.stream_input_samples
            self.first_output_compute_ms = elapsed_ms
            self.first_output_samples = len(output)
        if output is not None:
            self._schedule_cache_repair()
        return output

    @staticmethod
    def _steady_cache(caches) -> bool:
        if caches is None or len(caches) != 4:
            return False
        expected = (36, 36, 24, 24)
        return all(
            tuple(key.shape) == (1, 2, length, 64)
            and tuple(value.shape) == (1, 2, length, 64)
            for (key, value), length in zip(caches, expected, strict=True)
        )

    @staticmethod
    def _flatten_cache(caches) -> list[np.ndarray]:
        flattened = []
        for key, value in caches:
            flattened.extend(
                (
                    key.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 64),
                    value.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 64),
                )
            )
        return flattened

    @staticmethod
    def _restore_cache(outputs: list[np.ndarray]):
        caches = []
        for layer in range(4):
            length = outputs[layer * 2].shape[1] // 2
            key = torch.from_numpy(
                outputs[layer * 2].astype(np.float32, copy=False).reshape(1, 2, length, 64).copy()
            )
            value = torch.from_numpy(
                outputs[layer * 2 + 1].astype(np.float32, copy=False).reshape(1, 2, length, 64).copy()
            )
            caches.append((key, value))
        return caches

    @staticmethod
    def _startup_stage(caches) -> int | None:
        if caches is None:
            return 0
        lengths = tuple(int(key.shape[2]) for key, _ in caches)
        if lengths == (12, 12, 12, 12):
            return 1
        if lengths == (24, 24, 24, 24):
            return 2
        return None

    def _vc_step(self, cond: torch.Tensor) -> torch.Tensor:
        self._adopt_repaired_cache()
        if self.vc_offset >= 4000:
            self.vc_offset = 0
            self.vc_kv_cache = None
        startup_stage = self._startup_stage(self.vc_kv_cache)
        if startup_stage in (1, 2) and self.dit_startup_policy == "quality":
            started = time.perf_counter()
            mel = super()._vc_step(cond)
            self.dit_cpu_seconds += time.perf_counter() - started
            self.dit_cpu_chunks += 1
            self.dit_cpu_ode_calls += 2
            return mel
        if (
            startup_stage is None
            and not self._steady_cache(self.vc_kv_cache)
        ):
            started = time.perf_counter()
            mel = super()._vc_step(cond)
            self.dit_cpu_seconds += time.perf_counter() - started
            self.dit_cpu_chunks += 1
            self.dit_cpu_ode_calls += 2
            return mel

        step_size = 0.5
        with torch.no_grad():
            x = torch.randn(1, cond.shape[1], 80, device=cond.device, dtype=cond.dtype)
            if self.noise_cache is not None:
                x[:, : self.block_size, :] = self.noise_cache
            self.noise_cache = x[:, -self.block_size :, :].clone()

            pre_kv_cache = self.vc_kv_cache
            pre_cache = (
                self._flatten_cache(pre_kv_cache)
                if pre_kv_cache is not None
                else []
            )
            rope = self.vc.rotary_embed.forward_from_seq_len(
                self.vc_offset + cond.shape[1]
            )[0]
            rope = rope[:, -52:, :].detach().cpu().numpy().astype(np.float32)
            common = [
                cond.detach().cpu().numpy().astype(np.float32),
                self.vc_spk_emb.detach().cpu().numpy().astype(np.float32),
                self.vc_gtm_kv[0].detach().cpu().numpy().astype(np.float32),
                self.vc_gtm_kv[1].detach().cpu().numpy().astype(np.float32),
                rope,
            ]
            ode_inputs = []
            for index in range(2):
                t_value = 1.0 - index * step_size
                r_value = max(0.0, t_value - step_size)
                ode_inputs.append((x.detach().clone(), t_value, r_value))
                if index < self.dit_npu_ode_steps:
                    npu_started = time.perf_counter()
                    runtime = (
                        self.dit_startup_npus[startup_stage]
                        if startup_stage is not None
                        else self.dit_npu
                    )
                    outputs = runtime.inference(
                        inputs=[
                            x.detach().cpu().numpy().astype(np.float32),
                            np.asarray([t_value], dtype=np.float32),
                            np.asarray([r_value], dtype=np.float32),
                            *common,
                            *pre_cache,
                        ]
                    )
                    self.dit_npu_seconds += time.perf_counter() - npu_started
                    velocity = torch.from_numpy(outputs[0].astype(np.float32, copy=False))
                    new_cache = self._restore_cache(outputs[1:])
                    self.dit_npu_calls += 1
                else:
                    cpu_started = time.perf_counter()
                    velocity, new_cache = self.vc(
                        x,
                        torch.tensor([t_value]),
                        torch.tensor([r_value]),
                        cond=cond,
                        spks=self.vc_spk_emb,
                        offset=self.vc_offset,
                        is_inference=True,
                        kv_cache=pre_kv_cache,
                        gtm_kv=self.vc_gtm_kv,
                    )
                    self.dit_cpu_seconds += time.perf_counter() - cpu_started
                    self.dit_cpu_ode_calls += 1
                x = x - step_size * velocity
                if index == 1:
                    self.vc_kv_cache = new_cache
            self.dit_npu_chunks += 1
            if startup_stage is not None:
                self.dit_startup_chunks[startup_stage] += 1
                if self.dit_npu_ode_steps == 2:
                    self._dit_cache_repair_payload = (
                        cond.detach().clone(),
                        ode_inputs,
                        pre_kv_cache,
                        self.vc_offset,
                    )
            self.vc_offset += self.chunk_size
            return x[:, : self.chunk_size, :].transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--asr-rknn", type=Path, required=True)
    parser.add_argument("--asr-position-table", type=Path, required=True)
    parser.add_argument("--dit-rknn", type=Path, required=True)
    parser.add_argument(
        "--dit-startup-rknn",
        type=Path,
        nargs=3,
        metavar=("EMPTY", "CACHE12", "CACHE24"),
        required=True,
    )
    parser.add_argument("--vocos-rknn", type=Path, required=True)
    parser.add_argument("--vocos-startup-rknn", type=Path, required=True)
    parser.add_argument(
        "--vocos-startup-device", choices=("cpu", "npu"), default="npu"
    )
    parser.add_argument("--asr-device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--asr-warmup-frames", type=int, choices=range(0, 19), default=5)
    parser.add_argument("--dit-npu-steps", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--dit-startup-policy", choices=("quality", "all-npu"), default="quality"
    )
    parser.add_argument("--target-emb", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--disable-runtime-warmup", action="store_true")
    args = parser.parse_args()

    root = args.meanvc2_root.resolve()
    sys.path.insert(0, str(root / "npu"))
    runtime = load_meanvc_runtime(root)

    class MeanVC2NpuRunner(
        AsrNpuMixin, DiTSteadyNpuMixin, VocosNpuMixin, runtime.VCRunner
    ):
        pass

    runner = MeanVC2NpuRunner(
        target_wav=None,
        target_emb=args.target_emb,
        device="cpu",
        model="120ms",
        threads=args.threads,
    )
    runner.init_asr_npu(
        args.asr_rknn.resolve(),
        args.asr_position_table.resolve(),
        enabled=args.asr_device == "npu",
        warmup_frames=args.asr_warmup_frames,
        runtime_warmup=not args.disable_runtime_warmup,
    )
    runner.init_dit_npu(
        args.dit_rknn.resolve(),
        [path.resolve() for path in args.dit_startup_rknn],
        args.dit_npu_steps,
        args.dit_startup_policy,
        runtime_warmup=not args.disable_runtime_warmup,
    )
    runner.init_vocos_npu(
        args.vocos_rknn.resolve(),
        startup_model_path=(
            args.vocos_startup_rknn.resolve()
            if args.vocos_startup_device == "npu"
            else None
        ),
        warmup=not args.disable_runtime_warmup,
    )
    try:
        import soundfile as sf

        input_info = sf.info(args.input)
        input_seconds = input_info.frames / input_info.samplerate
        run_started = time.perf_counter()
        runner.process_file(args.input, args.output, seed=args.seed)
        run_seconds = time.perf_counter() - run_started
        steady_samples = [
            latency
            for index, (latency, produced) in enumerate(
                zip(
                    runner.chunk_process_ms,
                    runner.chunk_produced_output,
                    strict=True,
                )
            )
            if index >= 4 and produced
        ]
        if not steady_samples:
            steady_samples = runner.chunk_process_ms[1:]
        first_input_ms = runner.first_output_after_samples / 16.0
        first_output_ms = first_input_ms + runner.first_output_compute_ms
        benchmark = {
            "asr_warmup_frames": args.asr_warmup_frames,
            "asr_first_real_frames": runner.asr_first_real_frames,
            "asr_first_history_frames": runner.asr_first_history_frames,
            "asr_packet_trace": runner.asr_packet_trace,
            "first_output_input_ms": round(first_input_ms, 3),
            "first_output_compute_ms": round(runner.first_output_compute_ms, 3),
            "first_output_samples": runner.first_output_samples,
            "cold_start_first_output_ms": round(first_output_ms, 3),
            "steady_chunk_median_ms": round(statistics.median(steady_samples), 3),
            "steady_chunk_p95_ms": round(float(np.percentile(steady_samples, 95)), 3),
            "rtf": round(run_seconds / input_seconds, 6),
            "dit_startup_npu_chunks": runner.dit_startup_chunks,
            "dit_startup_policy": args.dit_startup_policy,
            "dit_cache_repair_chunks": runner.dit_cache_repair_chunks,
            "dit_cache_repair_ms": round(
                runner.dit_cache_repair_seconds * 1000, 3
            ),
            "dit_cache_repair_wait_ms": round(
                runner.dit_cache_repair_wait_seconds * 1000, 3
            ),
            "first_boundary_crossfade_samples": runner.vocoder_wav_overlap,
            "first_boundary_crossfade_ms": round(
                runner.vocoder_wav_overlap / 16.0, 3
            ),
            "runtime_warmup_ms": round(
                (
                    runner.asr_runtime_warmup_seconds
                    + runner.dit_runtime_warmup_seconds
                    + runner.vocos_runtime_warmup_seconds
                )
                * 1000,
                3,
            ),
        }
        print(f"[Warm-up] {json.dumps(benchmark, ensure_ascii=False)}")
        if args.benchmark_json:
            args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
            args.benchmark_json.write_text(
                json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            f"[ASR NPU] calls: {runner.asr_npu_calls}, time: {runner.asr_npu_seconds:.3f}s"
        )
        print(
            f"[DiT NPU] NPU chunks: {runner.dit_npu_chunks}, calls: {runner.dit_npu_calls}, "
            f"CPU ODE calls: {runner.dit_cpu_ode_calls}, CPU warm-up chunks: "
            f"{runner.dit_cpu_chunks}, time: "
            f"{runner.dit_npu_seconds + runner.dit_cpu_seconds:.3f}s"
        )
        print(
            f"[Vocos NPU] NPU calls: {runner.vocos_npu_calls}, CPU fallback calls: "
            f"{runner.vocos_cpu_calls}, time: "
            f"{runner.vocos_npu_seconds + runner.vocos_cpu_seconds:.3f}s"
        )
    finally:
        runner.close_asr_npu()
        runner.close_dit_npu()
        runner.close_vocos_npu()


if __name__ == "__main__":
    main()
