"""Locate batch-context divergence inside the pinned quantized MiniLM graph.

This offline diagnostic writes only a temporary instrumented ONNX graph. It
does not modify the original artifact or any Urdyn index.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from tokenizers import Tokenizer

from semantic_batch_probe import MODEL_SHA256, TOKENIZER_SHA256, file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if file_hash(args.model) != MODEL_SHA256 or file_hash(args.tokenizer) != TOKENIZER_SHA256:
        parser.error("artifact checksum mismatch")

    graph = onnx.load(args.model)
    quant_nodes = [node for node in graph.graph.node if node.op_type == "DynamicQuantizeLinear"]
    assert len(quant_nodes) == 96
    first_changed = quant_nodes[7]
    assert "layer.0/intermediate" in first_changed.input[0]
    observed = {
        "pre_quant": first_changed.input[0],
        "quantized": first_changed.output[0],
        "scale": first_changed.output[1],
        "hidden": "last_hidden_state",
    }
    for name, value_type in ((observed["pre_quant"], TensorProto.FLOAT),
                             (observed["quantized"], TensorProto.UINT8),
                             (observed["scale"], TensorProto.FLOAT)):
        graph.graph.output.append(helper.make_tensor_value_info(name, value_type, None))

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=128)
    pad_id = tokenizer.token_to_id("<pad>")
    assert pad_id is not None
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>", length=128)
    target = "Roma è la capitale italiana."
    mixed = [target] + [" ".join(["Il contesto cambia radicalmente."] * (index % 7 + 1))
                        for index in range(31)]
    duplicate = [target] * 32

    def feed(texts: list[str]) -> dict[str, np.ndarray]:
        encodings = tokenizer.encode_batch(texts)
        ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}

    left_feed, right_feed = feed(mixed), feed(duplicate)
    assert np.array_equal(left_feed["input_ids"][0], right_feed["input_ids"][0])
    assert np.array_equal(left_feed["attention_mask"][0], right_feed["attention_mask"][0])

    with tempfile.TemporaryDirectory(prefix="urdyn-m08c-onnx-") as directory:
        instrumented = Path(directory) / "instrumented.onnx"
        onnx.save(graph, instrumented)
        for optimization in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                             ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            options = ort.SessionOptions()
            options.graph_optimization_level = optimization
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(instrumented), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            names = list(observed.values())
            left, right = [dict(zip(names, session.run(names, item)))
                           for item in (left_feed, right_feed)]
            result = {"optimization": str(optimization), "onnxruntime": ort.__version__}
            for name in ("pre_quant", "quantized", "hidden"):
                result[f"{name}_first_row_max_abs"] = float(np.max(np.abs(
                    left[observed[name]][0].astype(np.float64)
                    - right[observed[name]][0].astype(np.float64))))
            result["scale_mixed"] = float(left[observed["scale"]])
            result["scale_duplicate"] = float(right[observed["scale"]])
            result["scale_relative_difference"] = abs(result["scale_mixed"] - result["scale_duplicate"]) / result["scale_duplicate"]
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
