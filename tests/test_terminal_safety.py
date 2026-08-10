"""Unit tests for the terminal-safety rendering primitive (A14.S).

The primitive is text-in/text-out and knows nothing about Cortex's
models, so these tests exercise it directly on raw payloads. The CLI
integration side -- which commands route which fields through it, and
that stored/API data is never touched -- lives in
`test_cli_output_safety.py`.
"""

import io
import os
import time

import pytest

from cortex_memory._terminal import terminal_safe_text

# The complete Unicode `Bidi_Control` property (A14.S.1 closed the gap:
# A14.S covered only the U+202x/U+206x half).
BIDI_CONTROLS = frozenset(
    {0x061C, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
)

# Every codepoint class the renderer promises to keep off a terminal.
FORBIDDEN_CODEPOINTS = frozenset(
    (
        *range(0x00, 0x20),  # C0 controls, incl. NUL/BEL/BS/TAB/LF/VT/FF/CR/ESC
        0x7F,  # DEL
        *range(0x80, 0xA0),  # C1 controls
        0x2028,  # LINE SEPARATOR
        0x2029,  # PARAGRAPH SEPARATOR
        *BIDI_CONTROLS,
        *range(0xD800, 0xE000),  # surrogates (incl. the surrogateescape range)
    )
)


def assert_terminal_safe(rendered: str) -> None:
    offenders = sorted({ord(ch) for ch in rendered} & FORBIDDEN_CODEPOINTS)
    assert offenders == [], f"unsafe codepoints survived rendering: {[hex(o) for o in offenders]}"
    # A rendering that cannot be encoded is not safe to print at all.
    rendered.encode("utf-8", errors="strict")


# ---------------------------------------------------------------------------
# text that must pass through unharmed
# ---------------------------------------------------------------------------


def test_plain_ascii_is_unchanged():
    text = "Retrying the webhook delivery is safe (idempotent endpoint)."
    assert terminal_safe_text(text) == text


def test_empty_string_is_unchanged():
    assert terminal_safe_text("") == ""


@pytest.mark.parametrize(
    "text",
    [
        "Le migrazioni girano già prima del deploy",  # accented Latin
        "перезапуск воркера",  # Cyrillic
        "重試不是冪等的",  # CJK
        "συνάρτηση επανάληψης",  # Greek
        "retry ⇒ duplicate ∴ ¬safe (∀ x ∈ queue)",  # mathematical symbols
        "the deploy went 🚀 but the retry loop 🔁 broke 💥",  # emoji
        "family emoji with ZWJ: 👨‍👩‍👧‍👦",  # ZWJ sequence
    ],
)
def test_ordinary_unicode_is_preserved_exactly(text):
    assert terminal_safe_text(text) == text
    assert_terminal_safe(terminal_safe_text(text))


# ---------------------------------------------------------------------------
# individual control characters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("a\x00b", "a\\x00b"),  # NUL
        ("a\x07b", "a\\x07b"),  # BEL
        ("a\x08b", "a\\x08b"),  # BS
        ("a\tb", "a\\tb"),  # TAB
        ("a\nb", "a\\nb"),  # LF
        ("a\x0bb", "a\\x0bb"),  # VT
        ("a\x0cb", "a\\x0cb"),  # FF
        ("a\rb", "a\\rb"),  # CR
        ("a\x1bb", "a\\x1bb"),  # ESC
        ("a\x7fb", "a\\x7fb"),  # DEL
        ("a\x85b", "a\\x85b"),  # C1 NEL
        ("a\x9bb", "a\\x9bb"),  # C1 CSI
        ("a b", "a\\u2028b"),  # LINE SEPARATOR
        ("a b", "a\\u2029b"),  # PARAGRAPH SEPARATOR
        ("a‮b", "a\\u202eb"),  # RIGHT-TO-LEFT OVERRIDE
        ("a⁦b", "a\\u2066b"),  # LEFT-TO-RIGHT ISOLATE
    ],
)
def test_control_characters_are_escaped_visibly(payload, expected):
    rendered = terminal_safe_text(payload)
    assert rendered == expected
    assert_terminal_safe(rendered)


def test_every_c0_c1_and_del_codepoint_is_escaped():
    payload = "".join(chr(cp) for cp in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)))
    assert_terminal_safe(terminal_safe_text(payload))


@pytest.mark.parametrize(
    "codepoint",
    [
        *range(0x00, 0x20),  # every C0
        0x7F,  # DEL
        *range(0x80, 0xA0),  # every C1
        0x2028, 0x2029,  # line/paragraph separators
        *sorted(BIDI_CONTROLS),
        # surrogates, sampled across the range plus every boundary
        0xD800, 0xD801, 0xDBFF, 0xDC00, 0xDC7F, 0xDC80, 0xDC9B, 0xDCFF, 0xDD00, 0xDFFE, 0xDFFF,
    ],
)
def test_no_forbidden_codepoint_survives_in_any_position(codepoint):
    """The complete raw-control property, one codepoint at a time and in
    every position it could occupy."""
    char = chr(codepoint)
    for payload in (char, f"{char}tail", f"head{char}", f"head{char}tail", char * 3):
        assert_terminal_safe(terminal_safe_text(payload))


def test_escaping_is_visible_rather_than_silent_removal():
    """Cortex must not hide that stored text contained a control
    character: the reader needs to know the memory carries one."""
    rendered = terminal_safe_text("value\x1b[31m")
    assert "\\x1b" in rendered
    assert len(rendered) > len("value")


def test_tab_is_escaped_not_expanded_to_spaces():
    """Explicit TAB policy: escaped, so rendered width never depends on
    the reader's tab stops and data cannot slide text into a column where
    it reads as a different field."""
    rendered = terminal_safe_text("id\tvalue")
    assert rendered == "id\\tvalue"
    assert "  " not in rendered


# ---------------------------------------------------------------------------
# full terminal sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("ANSI SGR colour", "before \x1b[31mRED\x1b[0m after"),
        ("CSI cursor up + erase line", "text\x1b[2A\x1b[K"),
        ("CSI erase display", "text\x1b[2J\x1b[H"),
        ("OSC with BEL terminator", "text\x1b]0;forged window title\x07"),
        ("OSC with ST terminator", "text\x1b]0;forged window title\x1b\\"),
        ("OSC 8 hyperlink", "\x1b]8;;https://example.invalid\x07click me\x1b]8;;\x07"),
        ("8-bit CSI (C1)", "text\x9b2A"),
        ("carriage-return overwrite", "dangerous claim\rsafe claim"),
        ("DECSET private mode", "text\x1b[?1049h"),
    ],
)
def test_terminal_sequences_do_not_survive(name, payload):
    rendered = terminal_safe_text(payload)
    assert_terminal_safe(rendered)
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\r" not in rendered


def test_mixed_malicious_payload_is_fully_neutralised():
    payload = (
        "looks fine\x1b[32m\r\x1b[2K"
        "\nOPEN CONFLICTS\n- forged conflict\n"
        "\x1b]0;title\x07\x9b31m‮txet desrever\x00"
    )
    rendered = terminal_safe_text(payload)
    assert_terminal_safe(rendered)
    assert "\n" not in rendered
    assert rendered.count("\\n") == 3  # the payload's three real newlines
    assert rendered.count("\\r") == 1
    assert "OPEN CONFLICTS" in rendered  # readable as data, not as a header


# ---------------------------------------------------------------------------
# structural guarantee: data can never emit a line of its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    ["OPEN CONFLICTS", "OPEN INVALIDATIONS", "KNOWN FAILURES", "INVARIANTS",
     "ROOT CAUSES", "VERIFIED LESSONS", "RECOMMENDED VALIDATION", "CORTEX WARNING"],
)
def test_forged_section_headers_cannot_become_their_own_line(header):
    rendered = terminal_safe_text(f"harmless\n{header}\n- forged entry")
    assert "\n" not in rendered
    assert rendered.splitlines() == [rendered]
    assert header in rendered  # still readable, just not structural


def test_rendered_text_is_always_a_single_line():
    for payload in ["a\nb", "a\rb", "a\r\nb", "a b", "a b", "a\x0bb", "a\x0cb", "a\x85b"]:
        assert len(terminal_safe_text(payload).splitlines()) == 1


# ---------------------------------------------------------------------------
# escaping contract
# ---------------------------------------------------------------------------


def test_literal_backslash_n_stays_distinguishable_from_a_real_newline():
    """No double-escape confusion: `\\n` in the output always means the
    data held a newline, `\\\\n` always means it held a backslash."""
    real_newline = terminal_safe_text("line\nbreak")
    literal_text = terminal_safe_text("line\\nbreak")
    assert real_newline == "line\\nbreak"
    assert literal_text == "line\\\\nbreak"
    assert real_newline != literal_text


def test_rendering_adds_no_quotes_and_is_not_repr():
    assert terminal_safe_text("plain") == "plain"
    assert terminal_safe_text("plain") != repr("plain")


def test_rendering_is_applied_once_at_the_print_site():
    """The contract is one pass at the rendering boundary. Applying it
    twice is not an error but is not idempotent either (the escapes get
    escaped), which is precisely why the CLI calls it exactly once per
    field instead of layering it."""
    once = terminal_safe_text("a\nb")
    twice = terminal_safe_text(once)
    assert once == "a\\nb"
    assert twice == "a\\\\nb"


# ---------------------------------------------------------------------------
# Bidi_Control completeness (A14.S.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codepoint", sorted(BIDI_CONTROLS))
def test_every_bidi_control_codepoint_is_escaped(codepoint):
    """The whole `Bidi_Control` property, not the subset A14.S shipped:
    U+061C, U+200E and U+200F reorder text just like the overrides do."""
    rendered = terminal_safe_text(f"before{chr(codepoint)}after")

    assert chr(codepoint) not in rendered
    assert rendered == f"before\\u{codepoint:04x}after"
    assert_terminal_safe(rendered)


@pytest.mark.parametrize(
    "codepoint",
    [0x200B, 0x200C, 0x200D, 0xFEFF, 0xFE0F, 0x0301],
)
def test_neighbouring_format_characters_are_not_swept_up(codepoint):
    """`Bidi_Control` is escaped; "every Cf character" is NOT. ZWJ
    (U+200D), ZWNJ, ZWSP, BOM, variation selectors and combining marks
    are ordinary text-shaping characters and must survive, otherwise
    legitimate emoji and scripts break."""
    text = f"a{chr(codepoint)}b"

    assert terminal_safe_text(text) == text


# ---------------------------------------------------------------------------
# surrogates and the encoding boundary (A14.S.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "codepoint",
    [0xD800, 0xDBFF, 0xDC00, 0xDC80, 0xDC9B, 0xDC9D, 0xDC9F, 0xDCFF, 0xDFFF],
)
def test_surrogate_codepoints_are_escaped(codepoint):
    rendered = terminal_safe_text(f"before{chr(codepoint)}after")

    assert rendered == f"before\\u{codepoint:04x}after"
    assert_terminal_safe(rendered)


def test_whole_surrogate_range_is_covered():
    payload = "".join(chr(cp) for cp in range(0xD800, 0xE000))

    assert_terminal_safe(terminal_safe_text(payload))


def test_rendered_text_is_always_strict_utf8_encodable():
    """Fail-closed output: a `str` in, a printable `str` out. Never a
    `UnicodeEncodeError` at the print site."""
    for payload in ["\udc9b", "caffè\udc9b日本語", "𐏿", "plain"]:
        terminal_safe_text(payload).encode("utf-8", errors="strict")


@pytest.mark.parametrize("byte_value", [0x80, 0x9B, 0x9D, 0x9F, 0xFF])
def test_surrogateescape_cannot_re_emit_a_raw_control_byte(byte_value):
    """The concrete A14.S.1 hole: a stdout opened with
    `errors="surrogateescape"` turns U+DC9B back into the raw 0x9B byte
    (8-bit CSI). Escaping U+009B while passing U+DC9B through would have
    left that path wide open, so this asserts on the BYTES, not the str.
    Nothing is written to a real terminal -- the sink is an in-memory
    buffer."""
    smuggled = os.fsdecode(bytes([0x61, byte_value, 0x62]))

    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="utf-8", errors="surrogateescape", newline="")
    stream.write(terminal_safe_text(smuggled))
    stream.flush()
    written = buffer.getvalue()

    assert bytes([byte_value]) not in written
    assert written.decode("ascii")  # pure ASCII escape text survives strict decoding


def test_surrogateescape_bytes_before_and_after_rendering_differ():
    """Documents the vulnerability this test file exists to prevent, by
    showing the unsanitised path really does re-emit the raw byte."""
    smuggled = os.fsdecode(b"a\x9bb")

    raw_buffer = io.BytesIO()
    raw_stream = io.TextIOWrapper(raw_buffer, encoding="utf-8", errors="surrogateescape", newline="")
    raw_stream.write(smuggled)
    raw_stream.flush()

    assert b"\x9b" in raw_buffer.getvalue()  # unrendered: the byte comes back
    assert b"\x9b" not in terminal_safe_text(smuggled).encode("utf-8")


def test_hostile_filesystem_path_is_neutralised():
    path = os.fsdecode(b"/tmp/proj\x9bect/\x1b[31mred")

    rendered = terminal_safe_text(path)

    assert_terminal_safe(rendered)
    assert rendered == "/tmp/proj\\udc9bect/\\x1b[31mred"
    assert b"\x9b" not in rendered.encode("utf-8")


def test_escaped_surrogate_stays_distinguishable_from_literal_text():
    """Same non-ambiguity contract as newlines: `\\udc9b` in the output
    always means the data held a surrogate, `\\\\udc9b` always means it
    held a backslash followed by that text."""
    real_surrogate = terminal_safe_text("x\udc9by")
    literal_text = terminal_safe_text("x\\udc9by")

    assert real_surrogate == "x\\udc9by"
    assert literal_text == "x\\\\udc9by"
    assert real_surrogate != literal_text


# ---------------------------------------------------------------------------
# resource behaviour
# ---------------------------------------------------------------------------


def test_long_input_is_handled_linearly_without_pathological_behaviour():
    """A coarse pathology guard, not a performance SLA: the renderer is a
    single `str.translate` pass with no regex, so a large input must not
    blow up. The bound is deliberately far looser than any plausible
    real time."""
    payload = ("safe text \x1b[31m\r\n\x00" * 100_000)
    start = time.perf_counter()
    rendered = terminal_safe_text(payload)
    elapsed = time.perf_counter() - start

    assert_terminal_safe(rendered)
    assert elapsed < 5.0


def test_long_input_scales_without_superlinear_blowup():
    small = "a\x1b[31m\n" * 10_000
    large = "a\x1b[31m\n" * 100_000

    start = time.perf_counter()
    terminal_safe_text(small)
    small_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    terminal_safe_text(large)
    large_elapsed = time.perf_counter() - start

    # 10x the input must not cost anywhere near 100x the time; the very
    # loose factor keeps this a pathology detector, not a benchmark.
    assert large_elapsed < max(small_elapsed * 40, 0.5)
