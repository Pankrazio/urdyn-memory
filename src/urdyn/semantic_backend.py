"""Public contract for a local semantic encoder supplied by a client.

The encoder executes inference only. Urdyn owns normalization, canonical
records, the derived index, ranking, admission and fallback. The current
semantic policy is calibrated for the pinned MiniLM artifacts; other model
families require a separately validated policy before they can be used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ._semantic_identity import SEMANTIC_NORMALIZATION


@dataclass(frozen=True, slots=True)
class SemanticBackendIdentity:
    """Stable identity of an encoder and its complete vector space.

    ``provider`` must distinguish implementations whose output has not
    been proven equivalent. ``model_id`` identifies the exact model
    artifact; both asset digests and the encoding policy participate in
    the persisted identity. A policy changes when batching or padding
    changes in a way that can change output vectors.
    The client must verify actual file hashes before reporting available.
    """

    provider: str
    model_id: str
    model_revision: str
    dimensions: int
    artifact_sha256: str
    tokenizer_sha256: str
    encoding_policy: str
    # Names the engine normalization contract the index is built
    # under (Urdyn normalizes, not the backend). The default follows the
    # engine; the legacy "l2" value is refused by `configure_semantic_backend`.
    normalization: str = SEMANTIC_NORMALIZATION

    @property
    def index_provider(self) -> str:
        """Persist both asset fingerprints without changing the index schema."""
        return (
            f"{self.provider};artifact={self.artifact_sha256};"
            f"tokenizer={self.tokenizer_sha256};encoding={self.encoding_policy}"
        )


class SemanticBackend(Protocol):
    """Client-owned local inference behind Urdyn's semantic lifecycle.

    ``available`` must be cheap, read-only and offline. ``encode`` returns
    one finite, nonzero, unnormalized vector per text; Urdyn converts to
    float32, validates dtype, cardinality, dimensions and finiteness and
    applies the versioned L2 normalization contract (`normalization`). The
    backend must bound its own inference batch and must never download
    artifacts implicitly. Calls are synchronous; Android clients must
    invoke Urdyn from a worker thread.

    ``encode`` runs outside the Brain's lifecycle lock, so other
    operations on the Brain proceed while it works. Within one process Urdyn
    never calls the same backend concurrently for the same Brain; calls for
    different Brains may overlap, as before. A call that does not return is
    not interrupted. The same holds for ``semantic_setup``.
    """

    @property
    def identity(self) -> SemanticBackendIdentity: ...

    def available(self) -> bool: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
