#!/usr/bin/env python3
"""Build, convert, and run a tiny RKNN model to verify the RK3588 NPU path."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def make_onnx(output: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    shape = [1, 8, 32, 32]
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    scale = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "scale")
    bias = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "bias")
    nodes = [
        helper.make_node("Mul", ["input", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "bias"], ["biased"]),
        helper.make_node("Relu", ["biased"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "rk3588_npu_smoke",
        [input_info],
        [output_info],
        [scale, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="rk3588-meanvc2-npu",
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(f"saved_onnx={output}")


def convert(onnx_path: Path, rknn_path: Path) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        _check(rknn.config(target_platform="rk3588"), "config")
        _check(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        _check(rknn.build(do_quantization=False), "build")
        _check(rknn.export_rknn(str(rknn_path)), "export_rknn")
    finally:
        rknn.release()
    print(f"saved_rknn={rknn_path}")


def run(rknn_path: Path, loops: int) -> None:
    from rknnlite.api import RKNNLite

    rng = np.random.default_rng(42)
    input_data = rng.normal(size=(1, 8, 32, 32)).astype(np.float32)
    expected = np.maximum(input_data * 2.0 + 0.5, 0.0)

    # Verbose runtime mode prints a per-layer profile for every inference and
    # distorts the benchmark substantially, so keep it disabled here.
    runtime = RKNNLite(verbose=False)
    try:
        _check(runtime.load_rknn(str(rknn_path)), "load_rknn")
        _check(
            runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL),
            "init_runtime",
        )
        output = runtime.inference(inputs=[input_data], data_format="nchw")[0]
        start = time.perf_counter()
        for _ in range(loops):
            output = runtime.inference(inputs=[input_data], data_format="nchw")[0]
        elapsed = time.perf_counter() - start
    finally:
        runtime.release()

    max_abs_error = float(np.max(np.abs(output.astype(np.float32) - expected)))
    print(f"loops={loops}")
    print(f"total_ms={elapsed * 1000:.3f}")
    print(f"mean_ms={elapsed * 1000 / loops:.3f}")
    print(f"max_abs_error={max_abs_error:.6f}")
    if not np.isfinite(output).all() or max_abs_error > 0.02:
        raise SystemExit("NPU output validation failed")
    print("npu_smoke=PASS")


def _check(code: int, step: str) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with RKNN error {code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-onnx")
    make_parser.add_argument("--output", type=Path, required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--onnx", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--loops", type=int, default=100)

    args = parser.parse_args()
    if args.command == "make-onnx":
        make_onnx(args.output)
    elif args.command == "convert":
        convert(args.onnx, args.output)
    else:
        run(args.model, args.loops)


if __name__ == "__main__":
    main()
