"""Shared deterministic lexical relevance matching.

Both `preflight()` (what should I know before starting a task?) and
`guard()` (is there a known risk or applicable skill for this action?) need
to answer "is this candidate relevant to that text?" using the same notion
of relevance, so it lives in one place instead of two copies that could
drift apart. This is not a search engine: no ranking scores, no semantic
similarity, no configurable strategy is exposed to callers of either
function.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A small, fixed stopword list to keep single common words (e.g. "the",
# "to") from making unrelated experience look relevant. Not a language
# model, not configurable: just noise reduction for keyword overlap.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "to", "of", "for", "on", "in", "and", "or", "is",
        "are", "was", "were", "be", "been", "with", "this", "that", "it",
        "its", "as", "by", "at", "from", "into", "during", "not", "do",
        "does", "did",
    }
)


def tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token not in _STOPWORDS}


def is_relevant(query_tokens: frozenset[str], text: str) -> bool:
    """A candidate is relevant if it shares a strict majority of the
    query's significant vocabulary — more than half, not just a fixed
    count of two. A flat "two shared words" rule lets two completely
    unrelated candidates both match a long, generic query (e.g. "update",
    "error", and "change" all appearing somewhere) just because each
    happens to share exactly those two common engineering words; scaling
    the requirement with query length keeps that from happening, while a
    one-word query still only needs that one word."""
    if not query_tokens:
        return False
    shared = query_tokens & tokens(text)
    threshold = len(query_tokens) // 2 + 1
    return len(shared) >= threshold
