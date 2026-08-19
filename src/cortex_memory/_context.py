"""The Context Compiler: a budgeted, task-relevant working context,
compiled from the same canonical experience `preflight()` reads.

`preflight()` answers "what does Cortex know that bears on this task" by
projecting every relevant category, unbounded, one item per line, letting
the consumer read and prioritize it themselves. That is the right answer
for a human or an agent doing its own audit, but it does not scale: the
projection grows with how much relevant experience Cortex holds, not with
how much of it actually constrains the task at hand, and it never
arbitrates between categories or removes redundancy.

`compile_context` answers a narrower, harder question: "what is the
smallest set of constraints and open work an agent must respect RIGHT
NOW to start this task safely" -- under an explicit, finite budget. Three
behaviors make this observably different from `preflight()`, not just a
different rendering of the same data:

1. BUDGET. Candidates are admitted in one fixed cross-category priority
   order (see `_SECTION_ORDER`) until the next one no longer fits, then
   selection stops -- a deterministic prefix, never a "fill what fits"
   scan. `preflight()` has no capacity concept at all.

2. INVARIANTS ARE FILTERED, NOT UNCONDITIONAL. `Preflight.invariants`
   deliberately bypasses relevance matching (A9.1): a project-wide
   invariant is shown regardless of what `task` says. That is correct
   for an audit view, but it means an unrelated task always receives
   every current invariant, and A28's real-world measurement found this
   is the only thing preflight ever returns for a genuinely unrelated
   task. `compile_context` requires the SAME relevance admission as
   every other category (`memory_is_relevant`, see `_preflight.py`), so
   an unrelated task can compile to nothing.

3. DECISIONS ARE ADMITTED. No code path in `build_preflight` ever reads
   Memory kind `decision` (see `_preflight.py`'s module docstring for
   what it does read); `compile_context` does. This is not a preflight
   bug fix: preflight's contract is deliberately "prior EXPERIENCE"
   (failures, causes, lessons), while a Decision is a commitment, not
   experience. Extending preflight to show it would blur that contract;
   admitting it here, where the question is "what must I respect", does
   not.

A fourth, smaller difference: an Attempt that shares Evidence with an
already-selected RootCause is a citation on that RootCause's line, not a
second line of its own (see `_absorb_known_failures`). This is
provenance-based, not text-similarity deduplication, and it is the same
trust `_preflight.py`'s own evidence-rescue channel already places in
shared Evidence.

WHAT THIS MODULE DOES NOT DO. It never opens a store, never loads a
model, and never decides relevance -- `Cortex.context()` is the only
caller, and it does all storage/semantic-retrieval work (through the
exact same `_gather_experience`/`_semantic_prepare`/`_semantic_widen`
machinery `preflight()` uses) before calling `compile_context` with
already-admitted candidates. This module's only job is composition: what
survives a limited budget, in what order, with how much redundancy
removed, and how it renders as text. No LLM, no summarization, no
paraphrasing: an admitted item's `content` is exactly the stored text it
came from, and an item that does not fit is entirely absent, never
truncated.
"""

from __future__ import annotations

import dataclasses

from ._attempt import Attempt
from ._conflict import Conflict
from ._evidence import Evidence
from ._memory import Memory
from ._semantic_store import SemanticState
from ._terminal import terminal_safe_text as _safe

# [A29.1] Character budget, not a token estimate: Cortex has no
# provider-specific tokenizer dependency anywhere in the Core (see
# `pyproject.toml`), and a character count is exactly measurable against
# what `render()` actually produces, with no model-specific conversion
# factor pretending to be a neutral unit. A future target-aware renderer
# can substitute its own cost function without changing anything in this
# module's selection policy.
DEFAULT_CONTEXT_BUDGET = 4000

SECTION_CONSTRAINTS = "CONSTRAINTS"
SECTION_OPEN_RISKS = "OPEN RISKS"
SECTION_LESSONS = "LESSONS"
SECTION_DECISIONS = "DECISIONS"
SECTION_HISTORY = "HISTORY"
SECTION_VALIDATION = "VALIDATION"

# [A29.1] The one fixed cross-category priority this tracer uses, derived
# from which categories a real Dev Memory Loop session (A28) actually
# acted on: a current constraint first, then open operational risk (the
# category A28 measured as having the highest marginal value -- absent
# from code, tests and docs alike), then verified experience, then a
# standing architectural commitment, then historical narrative, and
# finally the validation to re-run -- last because it is the least
# urgent to read before starting, and the easiest to recover with a
# second, unbudgeted command. No cross-category score: this ordering IS
# the priority, applied once, top to bottom.
_SECTION_ORDER = (
    SECTION_CONSTRAINTS,
    SECTION_OPEN_RISKS,
    SECTION_LESSONS,
    SECTION_DECISIONS,
    SECTION_HISTORY,
    SECTION_VALIDATION,
)


@dataclasses.dataclass(frozen=True, slots=True)
class ContextItem:
    """One candidate admitted into a compiled context: enough to render
    a line and to audit it back to canonical storage, nothing else.

    `content` is the RAW stored text (a Memory's `content`, an
    Evidence's `content`, or an Attempt's task/approach joined by
    `" -- "`) -- exactly like `Memory`/`Evidence`/`Attempt` themselves,
    it is never sanitized here. `CompiledContext.render()` is the one
    rendering boundary that applies terminal-safety (see `_render_item`),
    the same "sanitize on output, not on storage" split `_cli.py`
    already uses for `Preflight`.

    `authority` is `epistemic_state` for a Memory-kind item, an
    Evidence's `kind` for `kind == "evidence"`, or `None` for
    `kind == "attempt"` (an Attempt carries neither concept).

    `provenance` names Attempt ids ABSORBED into this item (see
    `_absorb_known_failures`) -- ids only, never their content, so the
    citation costs almost nothing and stays auditable.

    `conflicts_with` names every OTHER Memory id this item currently has
    an open canonical `Conflict` with (see `_conflict.py`). This is pure
    disclosure: nothing here ever resolves a conflict or prefers one
    side. It is computed BEFORE this item's budget cost is measured, so
    the item and its conflict marker are always admitted or rejected
    together -- never a marker with no item to explain it, and never an
    item that hides a contradiction Cortex already knows about.
    """

    entity_id: str
    kind: str
    content: str
    authority: str | None
    provenance: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ContextSection:
    """A non-empty, ordered group of `ContextItem`s under one heading.
    `CompiledContext.sections` never holds an empty section -- an empty
    category is omitted entirely, exactly like `preflight`'s CLI
    rendering omits empty categories."""

    heading: str
    items: tuple[ContextItem, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledContext:
    """A budgeted, task-relevant working context, compiled once and
    never mutated afterward. Derived and transient, like `Preflight`:
    never persisted as canonical Memory, never logged as an Event, and
    always fully reconstructible from canonical state plus `task` and
    `budget` alone (see `Cortex.context`).

    `used` is how many characters `sections`' headings and items
    actually cost, measured by the same rendering `render()` uses (see
    `_render_item`) -- always `<= budget`. The `Retrieval:` line and the
    trailing selection-summary line are fixed, O(1) structural overhead,
    exactly like `Preflight`'s always-present retrieval line, and are
    deliberately NOT counted against `budget`: budget governs the part
    of the output that scales with recorded experience, not the constant
    header/footer around it.

    `omitted` is how many otherwise-eligible candidates did not fit
    under `budget`.

    `invariants_excluded` is how many CURRENT project-wide invariants
    were excluded for lacking task relevance -- computed independently
    of `budget`: a larger budget never reintroduces them, because they
    were never candidates in the first place. This is the field that
    makes `context()` observably different from `preflight()`, which
    includes every current invariant unconditionally (A9.1).
    """

    task: str
    sections: tuple[ContextSection, ...]
    retrieval: SemanticState | None
    budget: int
    used: int
    omitted: int
    invariants_excluded: int

    def is_empty(self) -> bool:
        return not self.sections

    def render(self) -> str:
        """Deterministic plain-text rendering: the same `task`, canonical
        state, `budget` and `retrieval` state always produce
        byte-identical output.

        This is the rendering boundary for compiled context -- content
        stored raw on `ContextItem` is sanitized here, exactly once,
        through the same terminal-safety primitive the CLI already uses
        for `Preflight` (`_terminal.terminal_safe_text`), so this text is
        safe to print directly to a terminal or inject into a prompt
        without carrying an embedded escape sequence or a forged line."""
        lines: list[str] = []
        if self.retrieval is not None:
            lines.append(f"Retrieval: {self.retrieval.retrieval_mode()}")
            lines.append("")
        if self.sections:
            for section in self.sections:
                lines.append(section.heading)
                for item in section.items:
                    lines.append(_render_item(item))
        elif self.omitted:
            # Distinct from the no-candidates case below: `omitted` here
            # equals the full relevant-candidate count (see
            # `compile_context`), so a positive value means candidates
            # existed and were admissible, but none fit under `budget` --
            # never say "no context" when the footer is about to report
            # otherwise-eligible items as omitted.
            lines.append("No compiled items fit within the budget.")
        else:
            lines.append("No compiled context for this task.")
        lines.append("")
        selected = sum(len(section.items) for section in self.sections)
        lines.append(
            _footer_text(
                selected=selected,
                total=selected + self.omitted,
                omitted=self.omitted,
                invariants_excluded=self.invariants_excluded,
            )
        )
        return "\n".join(lines)


def _render_item(item: ContextItem) -> str:
    if item.authority is None:
        lines = [f"- [{item.entity_id}] {_safe(item.content)}"]
    else:
        lines = [f"- [{item.entity_id}] ({item.authority}) {_safe(item.content)}"]
    if item.provenance:
        cited = ", ".join(f"[{entity_id}]" for entity_id in item.provenance)
        lines.append(f"  from attempt {cited}")
    if item.conflicts_with:
        conflicting = ", ".join(f"[{memory_id}]" for memory_id in item.conflicts_with)
        lines.append(f"  CONFLICTS WITH {conflicting}")
    return "\n".join(lines)


def _footer_text(*, selected: int, total: int, omitted: int, invariants_excluded: int) -> str:
    text = f"-- {selected} of {total} selected; {omitted} omitted for budget"
    if invariants_excluded:
        text += f"; {invariants_excluded} project invariant(s) not relevant to this task (see: cortex preflight)"
    return text


def _conflict_partner_map(conflicts: list[Conflict]) -> dict[str, tuple[str, ...]]:
    """Every current, OPEN conflict a Memory id participates in, both
    directions -- independent of whether that conflict would itself
    clear task relevance under `preflight()`'s own gating (see
    `PreflightConflict`'s docstring): here, disclosure follows only from
    an item having ALREADY been selected on its own merit, so this
    module does not need or reuse preflight's relevance-gated conflict
    view at all."""
    partners: dict[str, list[str]] = {}
    for conflict in conflicts:
        memory_a, memory_b = conflict.memory_ids
        partners.setdefault(memory_a, []).append(memory_b)
        partners.setdefault(memory_b, []).append(memory_a)
    return {memory_id: tuple(ids) for memory_id, ids in partners.items()}


def _absorb_known_failures(
    root_causes: tuple[Memory, ...], known_failures: tuple[Attempt, ...]
) -> tuple[dict[str, tuple[str, ...]], tuple[Attempt, ...]]:
    """Provenance-based redundancy control (A29.1), and the only kind
    this tracer does: an Attempt that shares at least one Evidence id
    with an already-relevant RootCause describes the SAME underlying
    experience from a different angle (the RootCause explains *why*, the
    Attempt records *what happened*), so it is cited on the RootCause's
    line instead of costing budget on a line of its own. This is the
    identical "shared Evidence proves shared experience" trust
    `_preflight.py`'s own evidence-rescue channel and A7.8's
    memory-cluster rescue already place in provenance -- extended here
    to Attempt/RootCause instead of Memory/Memory.

    NOT text-similarity deduplication: two Attempts with near-identical
    wording but no shared Evidence are never merged. An Attempt that
    shares no Evidence with any candidate RootCause is left in the
    returned `standalone` tuple, exactly as eligible for its own HISTORY
    line as before -- this function only ever REMOVES an Attempt from
    that pool when a concrete provenance link justifies it, never on a
    generic "looks similar" basis.

    Deterministic and order-preserving: `root_causes` is walked in its
    given (chronological) order, and an Attempt that could be cited by
    more than one RootCause is absorbed into the first one encountered,
    never duplicated across two citation lines.
    """
    absorbed: dict[str, list[str]] = {root_cause.memory_id: [] for root_cause in root_causes}
    absorbed_attempt_ids: set[str] = set()
    for root_cause in root_causes:
        root_cause_evidence = frozenset(root_cause.evidence_ids)
        if not root_cause_evidence:
            continue
        for attempt in known_failures:
            if attempt.attempt_id in absorbed_attempt_ids:
                continue
            if root_cause_evidence.isdisjoint(attempt.evidence_ids):
                continue
            absorbed[root_cause.memory_id].append(attempt.attempt_id)
            absorbed_attempt_ids.add(attempt.attempt_id)
    standalone = tuple(attempt for attempt in known_failures if attempt.attempt_id not in absorbed_attempt_ids)
    return {memory_id: tuple(ids) for memory_id, ids in absorbed.items()}, standalone


def compile_context(
    *,
    task: str,
    budget: int,
    invariants: tuple[Memory, ...],
    invariants_excluded: int,
    pending: tuple[Memory, ...],
    lessons: tuple[Memory, ...],
    decisions: tuple[Memory, ...],
    root_causes: tuple[Memory, ...],
    known_failures: tuple[Attempt, ...],
    recommended_validation_candidates: tuple[Evidence, ...],
    open_conflicts: list[Conflict],
    retrieval: SemanticState | None,
) -> CompiledContext:
    """Pure composition/budgeting/rendering logic over ALREADY
    relevance-admitted candidates. `Cortex.context()` is the only caller
    and does every storage read and semantic-retrieval call before
    reaching here (see its docstring for the full pipeline); nothing in
    this function opens a store, loads a model, or decides whether a
    candidate is relevant to `task` -- it only decides what SURVIVES a
    limited `budget`, in what order, with how much redundancy removed.

    `invariants`/`pending`/`lessons`/`decisions`/`root_causes` are each
    expected in the SAME order their source category is already in
    (chronological, oldest first -- the order every current-state list
    in this codebase is materialized in), which is also this function's
    within-category ordering: no additional sort is applied.

    `known_failures` is exactly `Preflight.known_failures` for the same
    `task` -- relevant failed Attempts, some of which
    `_absorb_known_failures` will fold into a `root_causes` citation
    instead of a HISTORY line of their own.

    `recommended_validation_candidates` is exactly
    `Preflight.recommended_validation` for the same `task`: EVERY
    Evidence that would qualify, not yet filtered by what this call ends
    up selecting. Only the subset actually cited by an item this call
    selects (via `evidence_ids`) is included in the VALIDATION section
    -- an Evidence that would validate a Lesson or RootCause budget cut
    away, or one that only a non-displayed relevant success attempt
    cited, does not itself earn a place.
    """
    conflict_partners = _conflict_partner_map(open_conflicts)
    absorbed_by_root_cause, standalone_attempts = _absorb_known_failures(root_causes, known_failures)

    def _memory_item(
        memory: Memory, kind: str, *, provenance: tuple[str, ...] = ()
    ) -> tuple[ContextItem, tuple[str, ...]]:
        item = ContextItem(
            entity_id=memory.memory_id,
            kind=kind,
            content=memory.content,
            authority=memory.epistemic_state,
            provenance=provenance,
            conflicts_with=conflict_partners.get(memory.memory_id, ()),
        )
        return item, memory.evidence_ids

    candidates: list[tuple[str, ContextItem, tuple[str, ...]]] = []
    for memory in invariants:
        candidates.append((SECTION_CONSTRAINTS, *_memory_item(memory, "invariant")))
    for memory in pending:
        candidates.append((SECTION_OPEN_RISKS, *_memory_item(memory, "pending")))
    for memory in lessons:
        candidates.append((SECTION_LESSONS, *_memory_item(memory, "lesson")))
    for memory in decisions:
        candidates.append((SECTION_DECISIONS, *_memory_item(memory, "decision")))
    for memory in root_causes:
        candidates.append(
            (
                SECTION_HISTORY,
                *_memory_item(memory, "root_cause", provenance=absorbed_by_root_cause.get(memory.memory_id, ())),
            )
        )
    for attempt in standalone_attempts:
        item = ContextItem(
            entity_id=attempt.attempt_id,
            kind="attempt",
            content=f"{attempt.task} -- {attempt.approach}",
            authority=None,
        )
        candidates.append((SECTION_HISTORY, item, attempt.evidence_ids))

    sections_out: dict[str, list[ContextItem]] = {}
    opened: set[str] = set()
    used = 0
    selected_evidence_ids: set[str] = set()
    stopped = False

    # [A29.1] PREFIX MONOTONICITY. Candidates are walked in exactly the
    # fixed cross-category order they were appended above; the first one
    # that does not fit stops selection entirely -- never skipped in
    # favor of a smaller one further down. This is what guarantees
    # `selection(SMALL) subseteq selection(MEDIUM) subseteq
    # selection(LARGE)` for three budgets over the same candidates: a
    # smaller budget's selection is always a PREFIX of a larger one's,
    # never a different subset a "fill what fits" scan could produce.
    for section, item, evidence_ids in candidates:
        if stopped:
            break
        text = _render_item(item)
        cost = len(text) + 1
        if section not in opened:
            cost += len(section) + 1
        if used + cost > budget:
            stopped = True
            break
        used += cost
        opened.add(section)
        sections_out.setdefault(section, []).append(item)
        selected_evidence_ids.update(evidence_ids)

    total_candidates = len(candidates)

    if stopped:
        # Selection never reached VALIDATION, so which validation
        # candidates would have been cited was never computed. Every
        # member of the full, unfiltered pool is counted as an omission
        # here -- an upper bound, not a precise citation count, which is
        # the honest answer to "how much did not fit" when the citation
        # filter itself never ran.
        total_candidates += len(recommended_validation_candidates)
    else:
        validation_items = [
            ContextItem(
                entity_id=evidence.evidence_id,
                kind="evidence",
                content=evidence.content,
                authority=evidence.kind,
            )
            for evidence in recommended_validation_candidates
            if evidence.evidence_id in selected_evidence_ids
        ]
        total_candidates += len(validation_items)
        for item in validation_items:
            if stopped:
                break
            text = _render_item(item)
            cost = len(text) + 1
            if SECTION_VALIDATION not in opened:
                cost += len(SECTION_VALIDATION) + 1
            if used + cost > budget:
                stopped = True
                break
            used += cost
            opened.add(SECTION_VALIDATION)
            sections_out.setdefault(SECTION_VALIDATION, []).append(item)

    sections = tuple(
        ContextSection(heading=name, items=tuple(sections_out[name]))
        for name in _SECTION_ORDER
        if name in sections_out
    )
    selected_count = sum(len(items) for items in sections_out.values())

    return CompiledContext(
        task=task,
        sections=sections,
        retrieval=retrieval,
        budget=budget,
        used=used,
        omitted=total_candidates - selected_count,
        invariants_excluded=invariants_excluded,
    )
