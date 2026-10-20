#!/usr/bin/env python3
"""Export the steady-state 120 ms MeanVC2 DiT step with streaming KV caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn


class StreamingDiTStep(nn.Module):
    """DiT forward with fixed steady-state caches and externally supplied RoPE."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: torch.Tensor,
        speaker: torch.Tensor,
        gtm_key: torch.Tensor,
        gtm_value: torch.Tensor,
        rope_freqs: torch.Tensor,
        key_0: torch.Tensor,
        value_0: torch.Tensor,
        key_1: torch.Tensor,
        value_1: torch.Tensor,
        key_2: torch.Tensor,
        value_2: torch.Tensor,
        key_3: torch.Tensor,
        value_3: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        time_embedding = self.model.t_time_embed(t) + self.model.r_time_embed(r)
        timbre = self.model.temporal_timbre(cond, gtm_key, gtm_value)
        expanded_speaker = speaker.unsqueeze(1).expand(-1, cond.shape[1], -1)
        hidden = self.model.input_embed(x, timbre, expanded_speaker)
        caches = (
            (key_0.reshape(1, 2, 36, 64), value_0.reshape(1, 2, 36, 64)),
            (key_1.reshape(1, 2, 36, 64), value_1.reshape(1, 2, 36, 64)),
            (key_2.reshape(1, 2, 24, 64), value_2.reshape(1, 2, 24, 64)),
            (key_3.reshape(1, 2, 24, 64), value_3.reshape(1, 2, 24, 64)),
        )
        new_caches = []
        rope = (rope_freqs, 1.0)
        for block, cache in zip(self.model.transformer_blocks, caches, strict=True):
            hidden, new_cache = block(
                hidden,
                time_embedding,
                rope=rope,
                is_inference=True,
                kv_cache=cache,
            )
            new_caches.append(new_cache)
        hidden = self.model.norm_out(hidden, time_embedding)
        velocity = self.model.proj_out(hidden)
        return (
            velocity,
            new_caches[0][0].reshape(1, 72, 64),
            new_caches[0][1].reshape(1, 72, 64),
            new_caches[1][0].reshape(1, 72, 64),
            new_caches[1][1].reshape(1, 72, 64),
            new_caches[2][0].reshape(1, 48, 64),
            new_caches[2][1].reshape(1, 48, 64),
            new_caches[3][0].reshape(1, 48, 64),
            new_caches[3][1].reshape(1, 48, 64),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()

    root = args.meanvc2_root.resolve()
    sys.path.insert(0, str(root / "runtime"))
    from run_rt import _load_vc_model

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = _load_vc_model(
        str(root / "src/config/config_120ms_40ms.json"),
        str(root / "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
        "cpu",
    )
    for module in model.modules():
        if hasattr(module, "native_rms_norm"):
            module.native_rms_norm = False
    wrapper = StreamingDiTStep(model).eval()

    speaker = torch.randn(1, 256)
    with torch.no_grad():
        gtm_key, gtm_value = model.gtm(speaker)
        cache = None
        for index in range(3):
            _, cache = model(
                torch.randn(1, 16, 80),
                torch.tensor([1.0]),
                torch.tensor([0.5]),
                cond=torch.randn(1, 16, 256),
                spks=speaker,
                offset=index * 12,
                is_inference=True,
                kv_cache=cache,
                gtm_kv=(gtm_key, gtm_value),
            )

    x = torch.randn(1, 16, 80)
    t = torch.tensor([1.0], dtype=torch.float32)
    r = torch.tensor([0.5], dtype=torch.float32)
    cond = torch.randn(1, 16, 256)
    rope_freqs = model.rotary_embed.forward_from_seq_len(52)[0]
    inputs = (
        x,
        t,
        r,
        cond,
        speaker,
        gtm_key,
        gtm_value,
        rope_freqs,
        cache[0][0].reshape(1, 72, 64),
        cache[0][1].reshape(1, 72, 64),
        cache[1][0].reshape(1, 72, 64),
        cache[1][1].reshape(1, 72, 64),
        cache[2][0].reshape(1, 48, 64),
        cache[2][1].reshape(1, 48, 64),
        cache[3][0].reshape(1, 48, 64),
        cache[3][1].reshape(1, 48, 64),
    )
    with torch.no_grad():
        expected = wrapper(*inputs)

    input_names = [
        "x", "t", "r", "cond", "speaker", "gtm_key", "gtm_value", "rope_freqs",
        "key_0", "value_0", "key_1", "value_1", "key_2", "value_2", "key_3", "value_3",
    ]
    output_names = [
        "velocity", "key_0_out", "value_0_out", "key_1_out", "value_1_out",
        "key_2_out", "value_2_out", "key_3_out", "value_3_out",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(args.output),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(args.output))
    dropout_nodes = [node for node in onnx_model.graph.node if node.op_type == "Dropout"]
    for dropout in dropout_nodes:
        data_input = dropout.input[0]
        data_output = dropout.output[0]
        for node in onnx_model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == data_output:
                    node.input[index] = data_input
        onnx_model.graph.node.remove(dropout)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(args.output))
    fixture = {name: value.detach().numpy() for name, value in zip(input_names, inputs, strict=True)}
    fixture.update(
        {f"expected_{name}": value.detach().numpy() for name, value in zip(output_names, expected, strict=True)}
    )
    np.savez(args.fixture, **fixture)
    print(f"saved_onnx={args.output}")
    print(f"saved_fixture={args.fixture}")
    print(f"cache_shapes={[tuple(item[0].shape) for item in cache]}")
    print(f"removed_zero_dropout_nodes={len(dropout_nodes)}")


if __name__ == "__main__":
    main()
