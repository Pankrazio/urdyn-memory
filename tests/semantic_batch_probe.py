"""Offline diagnostic for the pinned MiniLM ARM64 artifact's batch invariance.

Run from any directory with ONNX Runtime, NumPy and Tokenizers installed:
python tests/semantic_batch_probe.py --model /path/model_qint8_arm64.onnx --tokenizer /path/tokenizer.json
The script reads a frozen synthetic corpus and prints aggregate metrics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from urdyn import _semantic as semantic

MODEL_SHA256 = "783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672"
TOKENIZER_SHA256 = "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"
CORPUS_SHA256 = "8c9f32de14b82c4fa876dfea30f57dec9baa7ab9453519c389252238f8a19e5e"
CORPUS_PATH = Path(__file__).parent / "data" / "m08b_semantic_corpus.json"


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
        parser.error("model or tokenizer checksum does not match the pinned artifact")
    texts = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_bytes = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
    if len(texts) != 32 or hashlib.sha256(corpus_bytes).hexdigest() != CORPUS_SHA256:
        parser.error("frozen corpus checksum mismatch")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=128)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        parser.error("pinned tokenizer has no <pad> token")
    input_names = {item.name for item in session.get_inputs()}

    def encode(batch: list[str]):
        encodings = tokenizer.encode_batch(batch)
        ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        feed = {"input_ids": ids}
        if "attention_mask" in input_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, feed)[0]
        float_mask = mask[..., None].astype(np.float32)
        pooled = (hidden * float_mask).sum(axis=1) / np.clip(float_mask.sum(axis=1), 1e-9, None)
        vectors = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return encodings, ids, mask, hidden, pooled, vectors

    print(json.dumps({"corpus_sha256": CORPUS_SHA256, "onnxruntime": ort.__version__,
                      "artifact_sha256": MODEL_SHA256, "tokenizer_sha256": TOKENIZER_SHA256}), flush=True)
    for fixed_padding in (False, True):
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>", length=128 if fixed_padding else None)
        reference = [encode([text]) for text in texts]
        for batch_size in (1, 4, 8, 32):
            start_time = time.monotonic()
            rows = []
            for start in range(0, len(texts), batch_size):
                enc, ids, mask, hidden, pooled, vectors = encode(texts[start:start + batch_size])
                for index, item in enumerate(enc):
                    rows.append((item, ids[index], mask[index], hidden[index], pooled[index], vectors[index]))
            token_mismatch = mask_mismatch = padding_changes = order_changes = admission_changes = 0
            max_hidden = max_pooled = max_vector = max_score = 0.0
            min_cosine = 1.0
            for index, row in enumerate(rows):
                ref_enc, ref_ids, ref_mask, ref_hidden, ref_pooled, ref_vector = reference[index]
                valid = int(ref_mask[0].sum())
                token_mismatch += row[0].ids[:valid] != ref_enc[0].ids[:valid]
                mask_mismatch += row[0].attention_mask[:valid] != ref_enc[0].attention_mask[:valid]
                padding_changes += len(row[0].ids) != len(ref_enc[0].ids)
                max_hidden = max(max_hidden, float(np.max(np.abs(row[3][:valid] - ref_hidden[0, :valid]))))
                max_pooled = max(max_pooled, float(np.max(np.abs(row[4] - ref_pooled[0]))))
                max_vector = max(max_vector, float(np.max(np.abs(row[5] - ref_vector[0]))))
                min_cosine = min(min_cosine, float(row[5] @ ref_vector[0]))
            candidate_ids = [str(index) for index in range(20)]
            ref_matrix = np.stack([reference[index][5][0] for index in range(20)])
            matrix = np.stack([rows[index][5] for index in range(20)])
            for query in range(20, 30):
                expected = semantic.rank_candidates(reference[query][5][0], candidate_ids, ref_matrix)
                actual = semantic.rank_candidates(rows[query][5], candidate_ids, matrix)
                order_changes += [item[0] for item in expected] != [item[0] for item in actual]
                admission_changes += semantic.semantic_admitted_id(expected, semantic.ENTITY_MEMORY) != semantic.semantic_admitted_id(actual, semantic.ENTITY_MEMORY)
                max_score = max(max_score, max(abs(left[1] - right[1]) for left, right in zip(sorted(expected), sorted(actual))))
            negative_ids = [str(index) for index in range(19)]
            negative_matrix = np.stack([rows[index][5] for index in range(19)])
            negative_admissions = sum(
                semantic.semantic_admitted_id(
                    semantic.rank_candidates(rows[query][5], negative_ids, negative_matrix),
                    semantic.ENTITY_MEMORY,
                ) is not None
                for query in (19, 20)
            )
            print(json.dumps({"fixed_padding": fixed_padding, "batch": batch_size,
                              "seconds_host": round(time.monotonic() - start_time, 3),
                              "token_mismatch": token_mismatch, "mask_mismatch": mask_mismatch,
                              "padding_changes": padding_changes, "max_hidden_abs": max_hidden,
                              "max_pool_abs": max_pooled, "max_vector_abs": max_vector,
                              "min_vector_cosine": min_cosine, "order_changes_of_10": order_changes,
                              "admission_changes_of_10": admission_changes,
                              "negative_admissions_of_2": negative_admissions,
                              "max_score_abs": max_score}), flush=True)

    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>", length=128)
    target = texts[0]
    mixed = [target] + [" ".join(["Il contesto cambia radicalmente."] * (index % 7 + 1)) for index in range(31)]
    duplicate = [target] * 32
    left = encode(mixed)
    right = encode(duplicate)
    print(json.dumps({"same_shape": left[3].shape == right[3].shape,
                      "same_first_ids": bool(np.array_equal(left[1][0], right[1][0])),
                      "same_first_mask": bool(np.array_equal(left[2][0], right[2][0])),
                      "first_hidden_max_abs": float(np.max(np.abs(left[3][0] - right[3][0]))),
                      "first_pool_max_abs": float(np.max(np.abs(left[4][0] - right[4][0]))),
                      "first_vector_cosine": float(left[5][0] @ right[5][0])}), flush=True)


if __name__ == "__main__":
    main()
