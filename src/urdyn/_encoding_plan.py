"""Semantic inference outside the lifecycle lock.

Previously every retrieval call (`preflight`, `context`, `guard`) ran its
encoder -- the query embedding and the incremental refresh of pending records
-- inside the Brain's lifecycle hold (`_lifecycle.guarded`). A slow or stuck
encoder therefore held the lock, and every other operation on that Brain,
from any thread or cooperating process, waited for it.

Such a call now runs in three steps, and only the first and the last hold the
lock:

1. PLAN (lifecycle hold, read-only, best effort): the texts this call will
   ask the encoder for -- each stale pool's pending records, in the order and
   batches `_semantic_refresh` uses, then the query -- plus an
   `EncodingTicket` naming the encoder and the storage incarnation they are
   meant for. Planning authorizes nothing: any error just means "no plan".
2. INFER (no lifecycle hold, no SQLite connection): run the encoder on those
   batches through `_semantic.embed`, the single numeric contract, and keep
   private read-only copies of the results (`PreparedEncodings`). A failure
   is recorded once and replayed, never retried.
3. PUBLISH (lifecycle hold): the unchanged guarded method runs against the
   storage as it is NOW -- identity, integrity and every canonical read are
   redone -- with a `PreparedEncoder`, which never infers. It serves a stored
   result only for the same encoder and the same texts; anything else fails
   observably (per-call semantic outcome). So nothing computed for another
   state can be published.

Validity is decided by CONTENT, not by a counter. A prepared row is a pure
function of (encoder, batch of texts), and step 3 looks it up by both,
recomputed under the lock. A refresh row is published only if its record is
still pending with the same text. That is exactly the batch `_semantic_refresh`
would have computed at planning time, so the outcome is that of a refresh
committed then and followed by the concurrent writes, which stay pending. The
storage incarnation (`StorageIncarnation`) only decides whether the plan is
worth using at all: a replaced store (restore) means one fresh plan.

What this does NOT do: it does not bound how long one inference runs (a stuck
native call keeps its own thread; Python cannot cancel it). It does not make a
plan valid after an arbitrary external replacement of the files: there the
content check still holds, but the replacement itself may go unnoticed. And it
does not change `semantic_setup`, which still infers under the lock.
"""
from __future__ import annotations

import contextvars
import dataclasses
from contextlib import contextmanager
from typing import Any


class EncodingNotPrepared(Exception):
    """The current call has no prepared result for this encoder and these
    texts. Inference never runs inside the lifecycle hold of a planned call,
    so this is reported as a failed semantic read, never answered by
    encoding under the lock."""


class EncoderBusy(EncodingNotPrepared):
    """Another call of this process was still using this Brain's encoder when
    the bounded wait for it expired. The other call is not interrupted."""


@dataclasses.dataclass(frozen=True)
class EncodingTicket:
    """What one call planned to encode, and for what.

    `brain_id`: the Brain identity checked when planning (the guarded call
    re-checks it). `incarnation`: `(memory.db, semantic_index.db)` file
    identities when planning -- a replacement signal, never an identity.
    `encoder_key`: the encoder the results are bound to (value, not object).
    `meta`, `backend`, `backend_identity`: what step 2 builds that encoder
    from, captured under the lock so a concurrent reconfiguration of the
    handle cannot change it. `query`: the query text, or None. `refresh`:
    `((entity_type, ((id, text), ...)), ...)` in refresh order."""

    brain_id: str
    incarnation: tuple
    encoder_key: tuple
    meta: Any
    backend: Any
    backend_identity: Any
    query: str | None
    refresh: tuple

    @property
    def has_work(self) -> bool:
        return self.query is not None or bool(self.refresh)


def _frozen(rows: Any) -> Any:
    """A private read-only copy of an encoder result, so nothing the encoder
    (or anyone else) still references can change it after the lock is
    released. A non-array result is kept as is: the numeric contract rejects
    it in step 3, exactly where it would have been rejected before."""
    import numpy as np

    if not isinstance(rows, np.ndarray):
        return rows
    copy = np.array(rows, copy=True)
    copy.setflags(write=False)
    return copy


class PreparedEncodings:
    """The results of step 2 for one ticket. Immutable once built."""

    __slots__ = ("ticket", "_failure", "_query_rows", "_query_failure", "_rows", "_failed")

    def __init__(self, ticket: EncodingTicket | None, *, failure: BaseException | None = None,
                 query_rows: Any = None, query_failure: BaseException | None = None,
                 refresh: tuple = ()) -> None:
        self.ticket = ticket
        self._failure = failure
        self._query_rows = query_rows
        self._query_failure = query_failure
        rows: dict[tuple[str, str, str], Any] = {}
        failed: dict[str, tuple[frozenset, BaseException]] = {}
        for entity_type, items, vectors, error in refresh:
            if error is not None:
                failed[entity_type] = (frozenset(items), error)
                continue
            for index, (entity_id, text) in enumerate(items):
                rows[(entity_type, entity_id, text)] = vectors[index]
        self._rows = rows
        self._failed = failed

    def encoder_for(self, key: tuple) -> "PreparedEncoder":
        """The encoder step 3 uses for `key`. Raises the recorded failure if
        the encoder could not be built in step 2 (as building it under the
        lock would have raised)."""
        if self.ticket is None or key != self.ticket.encoder_key:
            return PreparedEncoder(self, valid=False)
        if self._failure is not None:
            raise self._failure
        return PreparedEncoder(self, valid=True)


class PreparedEncoder:
    """Stands in for the encoder during step 3; never runs inference."""

    __slots__ = ("_prepared", "_valid")

    def __init__(self, prepared: PreparedEncodings, *, valid: bool) -> None:
        self._prepared = prepared
        self._valid = valid

    def _require_valid(self) -> PreparedEncodings:
        if not self._valid:
            raise EncodingNotPrepared("the encoder changed after this call was planned")
        return self._prepared

    def encode(self, texts: Any) -> Any:
        raise EncodingNotPrepared("inference never runs inside the lifecycle hold of a planned call")

    def query_rows(self, text: str) -> Any:
        """The rows `embed(model, [text])` returned in step 2 (a copy)."""
        prepared = self._require_valid()
        if prepared.ticket.query != text:
            raise EncodingNotPrepared("this query was not planned")
        if prepared._query_failure is not None:
            raise prepared._query_failure
        rows = prepared._query_rows
        return rows.copy() if hasattr(rows, "copy") else rows

    def refresh_rows(self, entity_type: str, pending: list[tuple[str, str]]) -> tuple[list[str], Any]:
        """`(ids, vectors)` for the records of `pending` (id and text, as
        recomputed now) that step 2 encoded with the same text; records
        written after planning are left out. Raises the recorded failure if
        the batch holding any of them failed, as that pool's refresh did."""
        import numpy as np

        prepared = self._require_valid()
        failed = prepared._failed.get(entity_type)
        if failed is not None and any(item in failed[0] for item in pending):
            raise failed[1]
        hits = [(entity_id, prepared._rows[(entity_type, entity_id, text)])
                for entity_id, text in pending if (entity_type, entity_id, text) in prepared._rows]
        if not hits:
            return [], None
        return [entity_id for entity_id, _ in hits], np.stack([row for _, row in hits])


def infer(ticket: EncodingTicket, *, build_model: Any, embed: Any, validate_refresh: Any) -> PreparedEncodings:
    """Step 2. Builds the encoder, then encodes in the same order as
    an in-lock call would: the refresh batches pool by pool, stopping at the first
    failure as `_semantic_refresh` does, then the query. `validate_refresh`
    applies the refresh's own acceptance check (row count, index width)."""
    try:
        model = build_model()
    except Exception as exc:
        return PreparedEncodings(ticket, failure=exc)
    if model is None:
        return PreparedEncodings(ticket, failure=EncodingNotPrepared("no encoder can answer this index"))
    refresh = []
    for entity_type, items in ticket.refresh:
        try:
            vectors = _frozen(validate_refresh(embed(model, [text for _, text in items]), len(items)))
        except Exception as exc:
            refresh.append((entity_type, items, None, exc))
            break
        refresh.append((entity_type, items, vectors, None))
    query_rows = query_failure = None
    if ticket.query is not None:
        try:
            query_rows = _frozen(embed(model, [ticket.query]))
        except Exception as exc:
            query_failure = exc
    return PreparedEncodings(ticket, query_rows=query_rows, query_failure=query_failure, refresh=tuple(refresh))


_ACTIVE: contextvars.ContextVar[PreparedEncodings | None] = contextvars.ContextVar(
    "urdyn_prepared_encodings", default=None
)


def active() -> PreparedEncodings | None:
    return _ACTIVE.get()


@contextmanager
def activated(prepared: PreparedEncodings):
    token = _ACTIVE.set(prepared)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
