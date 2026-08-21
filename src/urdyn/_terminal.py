"""Terminal-safe rendering of untrusted text for the `urdyn` CLI.

Everything Urdyn stores is data the caller supplied: a memory's content,
an attempt's task/approach, a skill's name, an evidence's text. The CLI
prints that data inside a structure it emits itself -- section headers
(`VERIFIED LESSONS`), list prefixes (`- `), bracketed ids. Printing
stored text verbatim lets the DATA forge the STRUCTURE: a content of
`"safe\\nOPEN CONFLICTS\\n- forged"` produces three lines on stdout, two
of which are indistinguishable from output the program itself emitted,
and a content carrying `ESC [ 2 A` can erase lines the program already
printed. Neither is code execution -- nothing is ever interpreted as a
command -- but both let stored data lie about what Urdyn found, which is
exactly the certainty the CLI exists to report on.

The fix belongs here and only here: SANITIZE ON OUTPUT, NOT ON STORAGE.
The canonical record keeps every byte the caller wrote (`Memory.content`
returned by the public API is never touched -- see
`tests/test_cli_output_safety.py`'s API-invariance tests); only the
representation handed to a terminal is made safe.

The escaping is deliberately Python-style and VISIBLE rather than a
silent strip: Urdyn must not hide that stored text contained a control
character, since "this memory contains an escape sequence" is itself
information the reader needs. Backslash is escaped too, which is what
makes the rendering unambiguous: `\\n` in the output always means the data
held a real newline, while `\\\\n` means it held a literal backslash
followed by `n`. Output is therefore a one-way, readable rendering, not a
round-trippable encoding -- it is not `repr()`, it adds no quotes, and it
is not for machines.

This primitive is TERMINAL-SPECIFIC on purpose and must not grow into a
generic `sanitize()`. Terminal escaping is not JSON escaping, not
Markdown escaping, and above all not prompt safety: a future Context
Compiler feeding memory content to a model needs a structural
separation between trusted instructions and untrusted data, which no
amount of control-character escaping provides.
"""

from __future__ import annotations

# Characters that must never reach a terminal unescaped, mapped to a
# readable rendering. Built once at import and applied with
# `str.translate`, which is a single linear pass with no regex and no
# backtracking (see A14.S's long-input probe).
#
# Covered, and why each matters for presentation integrity:
#   C0 (U+0000-U+001F)  -- ESC (CSI/OSC/SGR/cursor movement), BEL, CR
#                          (same-line overwrite), LF (structural line
#                          forging), NUL/BS/VT/FF, and every other
#                          control a terminal may act on.
#   DEL (U+007F)        -- control, not printable text.
#   C1 (U+0080-U+009F)  -- some terminals decode these as the 8-bit
#                          equivalents of ESC-introduced sequences.
#   U+2028/U+2029       -- Unicode line/paragraph separators: line breaks
#                          to anything that treats them as such.
#   Bidi_Control        -- the complete Unicode property, not a subset:
#                          U+061C, U+200E, U+200F, U+202A-U+202E and
#                          U+2066-U+2069. None can forge a line, but all
#                          can visually reorder one so it reads as
#                          something other than what it says (the
#                          "Trojan Source" shape). Same threat class --
#                          stored data controlling appearance.
#   Surrogates          -- U+D800-U+DFFF (A14.S.1). These are not valid
#                          scalar values and normally cannot occur in
#                          text, but Python's `surrogateescape` handler
#                          puts them there for every byte a filesystem
#                          path (or any bytes-derived string) holds that
#                          is not valid UTF-8: byte 0x9B becomes U+DC9B.
#                          Escaping U+009B while letting U+DC9B through
#                          would be a hole, not a nuance -- a stdout
#                          opened with `errors="surrogateescape"` writes
#                          U+DC9B back out as the raw 0x9B byte, which
#                          is exactly the 8-bit CSI this module exists
#                          to stop. Escaping them also makes the result
#                          strictly UTF-8 encodable, so rendering can
#                          never raise `UnicodeEncodeError` on the way
#                          to a terminal.
#   Backslash           -- escaped so the escapes above are unambiguous.
#
# NOT covered, deliberately: ordinary printable Unicode. Accented Latin,
# emoji, CJK, mathematical symbols and every other normal character pass
# through untouched -- an ASCII-only filter would destroy legitimate
# content to solve a problem those characters do not cause.
_NAMED_ESCAPES = {
    0x09: "\\t",  # TAB: escaped rather than expanded to spaces, so
    #               alignment can never depend on the reader's tab stops
    #               and stored data can never push text into a column
    #               where it looks like a different field.
    0x0A: "\\n",
    0x0D: "\\r",
    0x5C: "\\\\",
}

# Every codepoint with the Unicode `Bidi_Control` property, enumerated
# rather than derived: `unicodedata` exposes the bidirectional CLASS of a
# character, not this property, and the set is tiny and stable across
# Unicode versions. Deliberately NOT "every Cf character" -- that would
# also escape ZWJ (U+200D) and break legitimate emoji sequences.
_BIDI_CONTROLS = (
    0x061C,  # ARABIC LETTER MARK
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # embedding / override / pop
    0x2066, 0x2067, 0x2068, 0x2069,  # isolates
)

_SURROGATES = range(0xD800, 0xE000)


def _build_translation() -> dict[int, str]:
    table: dict[int, str] = {}
    for codepoint in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)):
        table[codepoint] = f"\\x{codepoint:02x}"
    for codepoint in (0x2028, 0x2029, *_BIDI_CONTROLS, *_SURROGATES):
        table[codepoint] = f"\\u{codepoint:04x}"
    table.update(_NAMED_ESCAPES)
    return table


_TRANSLATION = _build_translation()


def terminal_safe_text(value: str) -> str:
    """Render arbitrary stored text as a single terminal-safe line.

    Knows nothing about Memory, Attempt, Skill, Evidence, Conflict or
    Preflight: it takes text and returns text, so the CLI decides WHAT is
    untrusted and this decides HOW it is shown.

    Guarantees, for any `str` input: the result contains no C0 control,
    no DEL, no C1 control, no Unicode line/paragraph separator, no
    `Bidi_Control` and no surrogate -- therefore no ESC-introduced
    sequence, no BEL, no same-line overwrite, and no line break. The last
    one is what keeps the CLI's own structure authentic: since data can
    never emit a newline, every line on stdout begins with a prefix the
    renderer wrote.

    Because surrogates are escaped too, the result is always strictly
    UTF-8 encodable: printing it can neither raise `UnicodeEncodeError`
    nor, on a stdout using `errors="surrogateescape"`, put a raw
    control byte back on the wire.
    """
    return value.translate(_TRANSLATION)
