#!/usr/bin/env python3
"""Export one of the three fixed-cache MeanVC2 DiT startup stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn


class StartupDiTStep(nn.Module):
    """DiT forward with an empty, 12-frame, or 24-frame KV cache."""

    def __init__(self, model: nn.Module, cache_lengths: tuple[int, ...]):
        super().__init__()
        self.model = model
        self.cache_lengths = cache_lengths

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
        *flat_caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        time_embedding = self.model.t_time_embed(t) + self.model.r_time_embed(r)
        timbre = self.model.temporal_timbre(cond, gtm_key, gtm_value)
        expanded_speaker = speaker.unsqueeze(1).expand(-1, cond.shape[1], -1)
        hidden = self.model.input_embed(x, timbre, expanded_speaker)
        if self.cache_lengths:
            caches = tuple(
                (
                    flat_caches[layer * 2].reshape(
                        1, 2, self.cache_lengths[layer], 64
                    ),
                    flat_caches[layer * 2 + 1].reshape(
                        1, 2, self.cache_lengths[layer], 64
                    ),
                )
                for layer in range(4)
            )
        else:
            caches = (None, None, None, None)

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
            *(
                tensor.reshape(1, -1, 64)
                for cache in new_caches
                for tensor in cache
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanvc2-root", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
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

    speaker = torch.randn(1, 256)
    with torch.no_grad():
        gtm_key, gtm_value = model.gtm(speaker)
        cache = None
        for index in range(args.stage):
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

    cache_lengths = (
        tuple(int(layer[0].shape[2]) for layer in cache) if cache else ()
    )
    wrapper = StartupDiTStep(model, cache_lengths).eval()
    x = torch.randn(1, 16, 80)
    t = torch.tensor([1.0], dtype=torch.float32)
    r = torch.tensor([0.5], dtype=torch.float32)
    cond = torch.randn(1, 16, 256)
    offset = args.stage * 12
    rope_freqs = model.rotary_embed.forward_from_seq_len(offset + 16)[0]
    flat_caches = (
        tuple(tensor.reshape(1, -1, 64) for layer in cache for tensor in layer)
        if cache
        else ()
    )
    inputs = (
        x,
        t,
        r,
        cond,
        speaker,
        gtm_key,
        gtm_value,
        rope_freqs,
        *flat_caches,
    )
    with torch.no_grad():
        expected = wrapper(*inputs)

    input_names = [
        "x",
        "t",
        "r",
        "cond",
        "speaker",
        "gtm_key",
        "gtm_value",
        "rope_freqs",
    ]
    if cache:
        input_names.extend(
            f"{kind}_{layer}"
            for layer in range(4)
            for kind in ("key", "value")
        )
    output_names = ["velocity"] + [
        f"{kind}_{layer}_out"
        for layer in range(4)
        for kind in ("key", "value")
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
    fixture = {
        name: value.detach().numpy()
        for name, value in zip(input_names, inputs, strict=True)
    }
    fixture.update(
        {
            f"expected_{name}": value.detach().numpy()
            for name, value in zip(output_names, expected, strict=True)
        }
    )
    np.savez(args.fixture, **fixture)
    print(f"stage={args.stage}")
    print(f"input_cache_lengths={cache_lengths}")
    print(f"output_cache_lengths={[int(expected[i].shape[1] // 2) for i in range(1, 9, 2)]}")
    print(f"saved_onnx={args.output}")
    print(f"saved_fixture={args.fixture}")
    print(f"removed_zero_dropout_nodes={len(dropout_nodes)}")


if __name__ == "__main__":
    main()
