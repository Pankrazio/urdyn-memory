"""Evaluate frozen MEMORY labels with the pinned MiniLM artifact, offline.

The labels in tests/data/m08c_holdout.json are independent of the M0.8B
exploratory corpus and are never used to tune Urdyn's admission thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from urdyn import _semantic as semantic


MODEL_SHA256 = "783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672"
TOKENIZER_SHA256 = "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"
HOLDOUT = Path(__file__).parent / "data" / "m08c_holdout.json"
HOLDOUT_SHA256 = "0b9e7882982da5cb80fe9adc6ec2b92755c57d3592ee5e4e9424afaa2d8946ff"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if file_hash(args.model) != MODEL_SHA256 or file_hash(args.tokenizer) != TOKENIZER_SHA256:
        parser.error("artifact checksum mismatch")
    if file_hash(HOLDOUT) != HOLDOUT_SHA256:
        parser.error("frozen holdout checksum mismatch")
    data = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    ids = [item["id"] for item in data["candidates"]]
    assert len(ids) == len(set(ids)) == 12
    assert len(data["queries"]) == 32
    assert all(item["expected"] is None or item["expected"] in ids for item in data["queries"])
    texts = [item["text"] for item in data["candidates"] + data["queries"]]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=128)
    pad_id = tokenizer.token_to_id("<pad>")
    assert pad_id is not None

    def embed(batch: list[str]) -> np.ndarray:
        encodings = tokenizer.encode_batch(batch)
        token_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        hidden = session.run(["last_hidden_state"], {
            "input_ids": token_ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(token_ids),
        })[0]
        float_mask = mask[..., None].astype(np.float32)
        pooled = (hidden * float_mask).sum(axis=1) / np.clip(float_mask.sum(axis=1), 1e-9, None)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    print(json.dumps({"holdout_sha256": HOLDOUT_SHA256, "artifact_sha256": MODEL_SHA256,
                      "tokenizer_sha256": TOKENIZER_SHA256, "runtime": ort.__version__,
                      "positive_queries": 24, "negative_queries": 8}), flush=True)
    results: dict[str, tuple[np.ndarray, list[str | None]]] = {}
    for fixed_padding, batch_size in ((False, 1), (True, 1), (False, 4), (True, 4),
                                      (False, 8), (True, 8), (False, 32), (True, 32)):
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>",
                                 length=128 if fixed_padding else None)
        matrix = np.vstack([embed(texts[start:start + batch_size])
                            for start in range(0, len(texts), batch_size)])
        decisions: list[str | None] = []
        correct = wrong = missed = false_positive = true_abstention = 0
        false_positive_queries = []
        for offset, query in enumerate(data["queries"]):
            ranked = semantic.rank_candidates(matrix[12 + offset], ids, matrix[:12])
            admitted = semantic.semantic_admitted_id(ranked, semantic.ENTITY_MEMORY)
            decisions.append(admitted)
            expected = query["expected"]
            if expected is None:
                false_positive += admitted is not None
                true_abstention += admitted is None
                if admitted is not None:
                    false_positive_queries.append(offset)
            elif admitted == expected:
                correct += 1
            elif admitted is None:
                missed += 1
            else:
                wrong += 1
        label = f"{'fixed' if fixed_padding else 'dynamic'}_{batch_size}"
        results[label] = (matrix, decisions)
        print(json.dumps({"policy": label, "correct_of_24": correct, "wrong_of_24": wrong,
                          "missed_of_24": missed, "false_positive_of_8": false_positive,
                          "true_abstention_of_8": true_abstention,
                          "false_positive_query_indices": false_positive_queries}), flush=True)
    baseline_matrix, baseline_decisions = results["dynamic_1"]
    for label, (matrix, decisions) in results.items():
        cosines = np.sum(matrix * baseline_matrix, axis=1)
        print(json.dumps({"compare_to_dynamic_1": label,
                          "min_vector_cosine": float(np.min(cosines)),
                          "decision_changes": sum(left != right for left, right in zip(decisions, baseline_decisions))}),
              flush=True)


if __name__ == "__main__":
    main()
