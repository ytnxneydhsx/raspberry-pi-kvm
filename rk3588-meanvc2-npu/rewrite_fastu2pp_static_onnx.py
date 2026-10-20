#!/usr/bin/env python3
"""Make the exported Fast-U2++ graph static enough for RKNN.

The original TorchScript graph slices a 5000-frame positional-encoding table
with the streaming offset. RKNN requires Slice parameters to be constants, so
this tool exposes the already-sliced four-frame position encoding as an input
and freezes the remaining shape/cache Slice parameters for the 120 ms model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


def prune_graph(model: onnx.ModelProto) -> None:
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    required_values = {output.name for output in model.graph.output}
    required_nodes: set[int] = set()
    pending = list(required_values)
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is None or id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        for input_name in node.input:
            if input_name and input_name not in required_values:
                required_values.add(input_name)
                pending.append(input_name)

    kept_nodes = [node for node in model.graph.node if id(node) in required_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    used_inputs = {name for node in kept_nodes for name in node.input if name}
    used_inputs.update(output.name for output in model.graph.output)
    kept_inputs = [value for value in model.graph.input if value.name in used_inputs]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)

    kept_initializers = [value for value in model.graph.initializer if value.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def add_probe_outputs(model: onnx.ModelProto, names: list[str]) -> None:
    existing = {output.name for output in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.INT64, None))


def flatten_cache_io(model: onnx.ModelProto) -> None:
    input_specs = {
        "attention_cache": ("attention_cache_flat", [1, 192, 128], [6, 4, 8, 128]),
        "convolution_cache": ("convolution_cache_flat", [1, 1536, 8], [6, 1, 256, 8]),
    }
    new_inputs = []
    reshape_inputs = []
    for value in model.graph.input:
        if value.name not in input_specs:
            new_inputs.append(value)
            continue
        flat_name, flat_shape, original_shape = input_specs[value.name]
        new_inputs.append(helper.make_tensor_value_info(flat_name, TensorProto.FLOAT, flat_shape))
        shape_name = f"{value.name}_shape"
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(original_shape, dtype=np.int64), shape_name)
        )
        reshape_inputs.append(
            helper.make_node(
                "Reshape", [flat_name, shape_name], [value.name], name=f"reshape_{value.name}"
            )
        )
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)
    for node in reversed(reshape_inputs):
        model.graph.node.insert(0, node)

    output_specs = {
        "attention_cache_out": ([1, 192, 128], [6, 4, 8, 128]),
        "convolution_cache_out": ([1, 1536, 8], [6, 1, 256, 8]),
    }
    for output in model.graph.output:
        if output.name not in output_specs:
            continue
        flat_shape, original_shape = output_specs[output.name]
        output_name = output.name
        internal_name = f"{output_name}_4d"
        for node in model.graph.node:
            for index, node_output in enumerate(node.output):
                if node_output == output_name:
                    node.output[index] = internal_name
        shape_name = f"{output_name}_flat_shape"
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(flat_shape, dtype=np.int64), shape_name)
        )
        model.graph.node.append(
            helper.make_node(
                "Reshape", [internal_name, shape_name], [output_name], name=f"flatten_{output_name}"
            )
        )
        output.CopyFrom(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, flat_shape))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-table", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    constant_tensors = {value.name: value for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        value_attribute = next(
            (attribute for attribute in node.attribute if attribute.name == "value"), None
        )
        if value_attribute is not None:
            constant_tensors[node.output[0]] = value_attribute.t
    position_slice = next(
        node
        for node in model.graph.node
        if node.op_type == "Slice"
        and node.input[0] in constant_tensors
        and tuple(constant_tensors[node.input[0]].dims) == (1, 5000, 256)
    )
    position_table = numpy_helper.to_array(constant_tensors[position_slice.input[0]]).astype(
        np.float32
    )
    args.position_table.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.position_table, position_table)

    position_input = "position_encoding"
    position_output = position_slice.output[0]
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == position_output:
                node.input[index] = position_input
    model.graph.node.remove(position_slice)
    model.graph.input.append(
        helper.make_tensor_value_info(position_input, TensorProto.FLOAT, [1, 12, 256])
    )
    prune_graph(model)

    slice_nodes = [node for node in model.graph.node if node.op_type == "Slice"]
    parameter_names = []
    for node in slice_nodes:
        parameter_names.extend(name for name in node.input[1:] if name)
    parameter_names = list(dict.fromkeys(parameter_names))
    original_outputs = [output.name for output in model.graph.output]
    add_probe_outputs(model, parameter_names)

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feed = {
        "fbank": np.zeros((1, 19, 80), dtype=np.float32),
        "required_cache_size": np.asarray(8, dtype=np.int64),
        "attention_cache": np.zeros((6, 4, 8, 128), dtype=np.float32),
        "convolution_cache": np.zeros((6, 1, 256, 8), dtype=np.float32),
        "position_encoding": position_table[:, 8:20, :],
    }
    active_feed = {item.name: feed[item.name] for item in session.get_inputs()}
    values = session.run(parameter_names, active_feed)
    constants = dict(zip(parameter_names, values, strict=True))

    del model.graph.output[:]
    for name in original_outputs:
        source = next(value for value in session.get_outputs() if value.name == name)
        model.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, source.shape)
        )

    for slice_index, node in enumerate(slice_nodes):
        for parameter_index in range(1, len(node.input)):
            old_name = node.input[parameter_index]
            if not old_name:
                continue
            new_name = f"slice_{slice_index}_parameter_{parameter_index}"
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.asarray(constants[old_name], dtype=np.int64), name=new_name
                )
            )
            node.input[parameter_index] = new_name

    prune_graph(model)
    flatten_cache_io(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"saved_onnx={args.output}")
    print(f"saved_position_table={args.position_table}")
    print(f"inputs={[value.name for value in model.graph.input]}")


if __name__ == "__main__":
    main()
