"""The verifiable identity of a semantic representation.

Vectors built under incompatible embedding configurations must never be
compared with each other, nor added to the same index. Before representation
references were introduced, the built-in encoder's index recorded only
`onnxruntime` + `repo@revision#artifact` + the normalization name: the bytes
actually executed, the tokenizer, truncation, padding, batching, pooling, the
runtime and the numeric environment were not part of it.

A REPRESENTATION is described by a canonical JSON document (the descriptor);
its FINGERPRINT is the SHA-256 of that document, and its STATIC FINGERPRINT the
SHA-256 of the part that can be known without loading the model (everything but
the digests of the loaded bytes and what the graph itself declares). The index
records a REFERENCE to it -- schema, assurance level and both fingerprints --
appended to the existing `semantic_meta.provider` column, in the same row as
the `building`/`ready` status: no new table, no schema change, and every build
that compares that column exactly (0.4.0 and earlier development builds)
refuses the new index. An index without the reference predates representation
references: never relabelled,
only rebuilt. Compatibility is EXACT equality of the full fingerprint, decided
by the core. There is no approximate equivalence, no canary and no
cross-runtime certificate here.

Every descriptor says how much of it is known, and by whom (`assurance`):

- `core-verified` -- the built-in ONNX encoder. The core hashed the
  tokenizer bytes it then handed to `tokenizers`, and hashed the model file
  and re-checked after ONNX Runtime loaded it that it is still that file (see
  `_semantic._construct_encoder` for why, and for the limit of a re-check);
  it configured truncation/padding/pooling itself and read them back from the
  loaded objects. Runtime versions and the environment are OBSERVED,
  not verified: a version string reported by a library is not an attestation
  of its machine code.
- `backend-declared` -- an external `SemanticBackend`. Every encoder field is
  the backend's own declaration (digests included): the core checks its shape
  and the vectors it returns, never the files it says it ran. A declared
  identity is never promoted to verified, and is never equal to a verified one.
- `unattested` -- any other object used as an encoder (one the core did not
  construct, e.g. a substituted loader). The core knows nothing about it, not
  even what it claims to be: all such encoders share ONE class with no
  identity, compatible only with itself and never with a verified or declared
  representation. In production no such encoder exists; it is the honest label
  for in-process substitution (the test suite's fakes), which the identity
  contract does not contain. It is never evidence of compatibility.

The `core` section is common to all three: the text composition the core
feeds the encoder, its normalization and vector format, and the observed
numeric environment of the process that normalizes (NumPy, architecture,
operating system, byte order, the CPU features NumPy reports). A different
environment is a different representation: no equivalence across platforms,
runtimes or machines is presumed. The price is an explicit rebuild of the
derived index after such a change; canonical memory is never involved.

This module has no third-party imports at module level.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import platform
import re
import sys
from typing import Any

SCHEMA = "urdyn-representation-v1"
CORE_VERIFIED = "core-verified"
BACKEND_DECLARED = "backend-declared"
UNATTESTED = "unattested"
ASSURANCES = frozenset({CORE_VERIFIED, BACKEND_DECLARED, UNATTESTED})

ONNX_ENCODER_KIND = "urdyn-onnx-v1"
EXTERNAL_ENCODER_KIND = "external-backend-v1"
UNATTESTED_ENCODER_KIND = "unattested-v1"

# Never a valid character sequence of a provider (built-in "onnxruntime", or
# an external `index_provider`, whose parts are [a-z0-9._-] joined by ";k=v").
PROVIDER_TAG = ";representation="
_HEX64 = re.compile(r"[0-9a-f]{64}")
UNOBSERVED = "unobserved"


class InvalidRepresentation(ValueError):
    """A descriptor, or an index's reference to one, is incomplete, unknown or
    malformed: it identifies nothing, so nothing may be compared under it."""


def canonical_json(descriptor: dict) -> str:
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def fingerprint_of(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Representation:
    """One validated descriptor, in canonical form, and its fingerprints."""

    descriptor_json: str
    fingerprint: str
    static_fingerprint: str
    assurance: str

    @classmethod
    def build(cls, descriptor: dict) -> "Representation":
        validate(descriptor)
        text = canonical_json(descriptor)
        return cls(text, fingerprint_of(text), static_fingerprint_of(descriptor), descriptor["assurance"])

    @property
    def descriptor(self) -> dict:
        return json.loads(self.descriptor_json)


@dataclasses.dataclass(frozen=True, slots=True)
class IndexReference:
    """What an index records about the representation it was built with."""

    assurance: str
    static_fingerprint: str
    fingerprint: str


def tag_provider(base: str, representation: Representation) -> str:
    """The `semantic_meta.provider` value of an index built under
    `representation`: never equal to `base` alone, which is what every
    earlier build compares."""
    return (f"{base}{PROVIDER_TAG}{SCHEMA}/{representation.assurance}/"
            f"{representation.static_fingerprint}/{representation.fingerprint}")


def read_provider(stored: str) -> tuple[str, IndexReference | None]:
    """`(base provider, reference or None)`. None: an index written before
    representation references existed (or by a build that does not know this
    contract). A reference
    that is present but malformed, or of an unknown schema or assurance, is
    `InvalidRepresentation` -- never read as "no reference"."""
    base, sep, tag = stored.rpartition(PROVIDER_TAG)
    if not sep:
        return stored, None
    parts = tag.split("/")
    if len(parts) != 4 or parts[0] != SCHEMA:
        raise InvalidRepresentation("unknown or malformed representation reference")
    _, assurance, static_fingerprint, fingerprint = parts
    if assurance not in ASSURANCES or not (_HEX64.fullmatch(static_fingerprint) and _HEX64.fullmatch(fingerprint)):
        raise InvalidRepresentation("unknown or malformed representation reference")
    return base, IndexReference(assurance, static_fingerprint, fingerprint)


def base_provider(stored: str) -> str:
    return stored.rpartition(PROVIDER_TAG)[0] if PROVIDER_TAG in stored else stored


# ---------------------------------------------------------------------------
# Completeness: a descriptor names every field of its schema, or nothing
# ---------------------------------------------------------------------------

_STR = (str,)


def _require(mapping: Any, keys: dict[str, Any], where: str) -> None:
    if not isinstance(mapping, dict):
        raise InvalidRepresentation(f"{where} is missing")
    if set(mapping) != set(keys):
        raise InvalidRepresentation(f"{where} must name exactly {sorted(keys)}, got {sorted(mapping)}")
    for key, kind in keys.items():
        value = mapping[key]
        if isinstance(kind, dict):
            _require(value, kind, f"{where}.{key}")
        elif kind == "hex64":
            if not (isinstance(value, str) and _HEX64.fullmatch(value)):
                raise InvalidRepresentation(f"{where}.{key} must be a SHA-256 digest")
        elif kind == "strs":
            if not (isinstance(value, list) and all(isinstance(v, str) and v for v in value)):
                raise InvalidRepresentation(f"{where}.{key} must be a list of names")
        elif kind is int:
            if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                raise InvalidRepresentation(f"{where}.{key} must be a positive integer")
        elif kind is bool:
            if not isinstance(value, bool):
                raise InvalidRepresentation(f"{where}.{key} must be a boolean")
        elif kind == "int?":
            if not (value is None or (isinstance(value, int) and not isinstance(value, bool) and value > 0)):
                raise InvalidRepresentation(f"{where}.{key} must be a positive integer or null")
        elif kind == "str?":
            if not (value is None or (isinstance(value, str) and value)):
                raise InvalidRepresentation(f"{where}.{key} must be a name or null")
        elif not (isinstance(value, kind) and value):
            raise InvalidRepresentation(f"{where}.{key} must be a non-empty name")


_CORE = {
    "text": {"policy": str, "chunk_max_chars": int},
    "normalization": str,
    "vector": {"dtype": str, "byte_order": str},
    "environment": {"numpy": str, "machine": str, "system": str, "cpu_features": "strs"},
}

_ENCODERS = {
    CORE_VERIFIED: {
        "kind": str,
        "assets": {
            "repository": str, "revision": str, "artifact": str,
            "model_sha256": "hex64", "tokenizer_sha256": "hex64",
        },
        "recipe": {
            "truncation": {"max_length": int, "strategy": str, "direction": str},
            "padding": {"strategy": str, "token": str, "direction": str},
            "batch": {"policy": str, "size": int, "composition_independent": bool},
            "inputs": "strs",
            "token_type_ids": str,
            "pooling": str,
            "dimensions": int,
        },
        "runtime": {
            "name": str, "version": str, "build": str, "providers": "strs",
            "session_options": str, "tokenizers": str,
        },
    },
    BACKEND_DECLARED: {
        "kind": str,
        "declared": {
            "provider": str, "model_id": str, "model_revision": str, "dimensions": int,
            "artifact_sha256": "hex64", "tokenizer_sha256": "hex64", "encoding_policy": str,
            "normalization": str,
        },
    },
    UNATTESTED: {"kind": str},
}


def validate(descriptor: Any) -> None:
    if not isinstance(descriptor, dict) or descriptor.get("schema") != SCHEMA:
        raise InvalidRepresentation("unknown representation schema")
    assurance = descriptor.get("assurance")
    if assurance not in ASSURANCES:
        raise InvalidRepresentation("unknown assurance level")
    _require(descriptor, {"schema": str, "assurance": str, "encoder": _ENCODERS[assurance], "core": _CORE}, "descriptor")
    kinds = {CORE_VERIFIED: ONNX_ENCODER_KIND, BACKEND_DECLARED: EXTERNAL_ENCODER_KIND,
             UNATTESTED: UNATTESTED_ENCODER_KIND}
    if descriptor["encoder"]["kind"] != kinds[assurance]:
        raise InvalidRepresentation("encoder kind does not match the assurance level")


# ---------------------------------------------------------------------------
# Observed environment (never verified; computed once per process)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _observed_environment() -> tuple:
    import numpy

    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features  # NumPy 2
    except ImportError:  # pragma: no cover - NumPy 1.x layout
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as features
        except ImportError:
            features = None
    names = sorted(name for name, present in features.items() if present) if isinstance(features, dict) else []
    return (numpy.__version__, platform.machine() or UNOBSERVED, platform.system() or UNOBSERVED,
            tuple(names) or (UNOBSERVED,))


def core_environment() -> dict:
    numpy_version, machine, system, features = _observed_environment()
    return {"numpy": numpy_version, "machine": machine, "system": system, "cpu_features": list(features)}


def core_section(*, text_policy: str, chunk_max_chars: int, normalization: str) -> dict:
    return {
        "text": {"policy": text_policy, "chunk_max_chars": chunk_max_chars},
        "normalization": normalization,
        "vector": {"dtype": "float32", "byte_order": sys.byteorder},
        "environment": core_environment(),
    }


def describe(assurance: str, encoder: dict, core: dict) -> Representation:
    return Representation.build({"schema": SCHEMA, "assurance": assurance, "encoder": encoder, "core": core})


def static_view(descriptor: dict) -> dict:
    """A descriptor minus what only loading the model establishes (for a
    core-verified encoder: the digests of the bytes executed, the graph's
    inputs and output width). For the other levels everything is known
    without loading, so the view is the descriptor itself."""
    view = json.loads(canonical_json(descriptor))
    encoder = view.get("encoder")
    if view.get("assurance") == CORE_VERIFIED and isinstance(encoder, dict):
        assets = encoder.get("assets")
        if isinstance(assets, dict):
            assets.pop("model_sha256", None)
            assets.pop("tokenizer_sha256", None)
        recipe = encoder.get("recipe")
        if isinstance(recipe, dict):
            recipe.pop("inputs", None)
            recipe.pop("dimensions", None)
    return view


def static_fingerprint_of(descriptor: dict) -> str:
    return fingerprint_of(canonical_json(static_view(descriptor)))
