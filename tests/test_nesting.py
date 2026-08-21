"""Prose inside containers.

Restricting the walker to top-level blocks left 93% of a bullet-heavy document
untranslated, so it now descends into lists, blockquotes, and footnote bodies.
Descending is only safe if the container's own structure survives, which is what
these tests pin down: the marker on the first line, the matching prefix on every
continuation line, and the fact that a block which grows a line during
translation does not escape its list.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ouroboros.backends.mock import MockBackend
from ouroboros.formats import base as formats
from ouroboros.persona import none_persona
from ouroboros.pipeline import RoundTrip
from ouroboros.rewrap import apply_continuation, strip_continuation

formats.load_builtins()

FIXTURES = Path(__file__).parent / "fixtures"


def _parse(name: str):
    fmt = "markdown" if name.endswith(".md") else "latex"
    source = (FIXTURES / name).read_text(encoding="utf-8")
    return source, formats.get(fmt).parse(source, path=name)


def _round_trip(name: str) -> str:
    _, doc = _parse(name)
    trip = RoundTrip(MockBackend(), pivot="fr", persona=none_persona(), cache=None)
    output, _ = trip.run(doc)
    return output


# ----------------------------------------------------------------- coverage


@pytest.mark.parametrize("name,minimum", [("nested.md", 75), ("gaps_min.md", 70)])
def test_container_prose_is_reached(name, minimum, tmp_path):
    if name == "gaps_min.md":
        # A document that is almost entirely list prose. Before descending into
        # containers this scored 7%.
        text = ("Intro paragraph.\n\n"
                "- First bullet with real prose that should be translated.\n"
                "- Second bullet with more real prose in it.\n")
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        doc = formats.get("markdown").parse(text, path=str(path))
        source = text
    else:
        source, doc = _parse(name)

    covered = sum(len(s) for s in doc.segments)
    pct = 100 * covered / len(source)
    assert pct >= minimum, f"only {pct:.0f}% of prose reached"


def test_latex_items_are_separate_segments():
    """Regression: \\item boundaries were merged into one translation unit.

    Two unrelated bullets handed over as a single paragraph invites the model to
    blend them together.
    """
    _, doc = _parse("nested.tex")
    texts = [s.text for s in doc.segments]
    assert "A short bullet with a citation \\cite{smith2020}." in texts
    assert "An inner bullet with its own prose." in texts
    # No segment may span two bullets.
    for text in texts:
        assert "\\item" not in text


# --------------------------------------------------------- structure survives


def test_markdown_list_markers_survive_a_round_trip():
    output = _round_trip("nested.md")
    assert output.count("\n- ") >= 4
    assert "\n  - " in output          # nested bullet
    assert "\n1. " in output and "\n2. " in output


def test_blockquote_marker_is_on_every_line():
    output = _round_trip("nested.md")
    quoted = [ln for ln in output.split("\n") if "blockquote" in ln.lower()]
    assert quoted
    for block in output.split("\n\n"):
        if not block.startswith(">"):
            continue
        for line in block.split("\n"):
            assert line.startswith(">"), f"blockquote line lost its marker: {line!r}"


def test_fenced_block_inside_a_list_is_untouched():
    output = _round_trip("nested.md")
    assert "  ```python\n  def f():\n      return 1\n  ```" in output


def test_table_inside_a_container_document_is_untouched():
    output = _round_trip("nested.md")
    for line in ("| Method | Score |", "|--------|------:|", "| Ours   |  94.7 |"):
        assert line in output


def test_footnote_definition_is_translated_but_its_label_is_not():
    _, doc = _parse("nested.md")
    bodies = [s for s in doc.segments if "footnote definition" in s.text]
    assert len(bodies) == 1
    assert not bodies[0].text.startswith("[^")

    output = _round_trip("nested.md")
    assert "[^1]: " in output


# ------------------------------------------------------- continuation prefix


def test_a_one_line_bullet_that_grows_keeps_its_list():
    """A single-line bullet has no second line to copy a prefix from.

    The prefix must be derived from the marker instead, or a bullet that becomes
    two lines during translation puts its second line outside the list.
    """
    text = "- A short bullet.\n"
    doc = formats.get("markdown").parse(text)
    assert doc.segments[0].meta["indent"] == "  "


def test_prefix_prefers_what_the_author_actually_wrote():
    """A footnote continued at four spaces stays at four, not the label's six."""
    _, doc = _parse("nested.md")
    body = next(s for s in doc.segments if "footnote definition" in s.text)
    assert body.meta["indent"] == "    "


def test_nested_bullet_prefix_accounts_for_depth():
    _, doc = _parse("nested.md")
    inner = next(s for s in doc.segments if s.text.startswith("An inner bullet"))
    assert inner.meta["indent"] == "    "


def test_blockquote_bullet_prefix_keeps_the_quote_marker():
    _, doc = _parse("nested.md")
    seg = next(s for s in doc.segments if s.text.startswith("A bullet inside"))
    assert seg.meta["indent"].startswith(">")


@pytest.mark.parametrize("prefix", ["  ", "> ", ">   ", "    "])
def test_strip_and_apply_continuation_round_trip(prefix):
    body = f"first line\n{prefix}second line\n{prefix}third line"
    stripped = strip_continuation(body, prefix)
    assert "\n" + prefix not in stripped
    assert apply_continuation(stripped, prefix) == body


def test_strip_continuation_handles_a_lazy_line():
    """A blockquote continuation may omit its marker; it must still dedent."""
    assert strip_continuation("first\nsecond", "> ") == "first\nsecond"


# -------------------------------------------------------------- edge cases


def test_task_list_checkbox_stays_at_the_start_of_the_item():
    """A checkbox is not CommonMark, so it arrives as ordinary prose.

    Left visible, a translator moves it to the end of the sentence and the task
    list stops being a task list.
    """
    output = _round_trip("nested.md")
    assert "- [ ] " in output
    assert "- [x] " in output
    assert "- [ ]  " not in output, "doubled space after the checkbox"


def test_hard_line_break_survives():
    """Two trailing spaces mean <br>. They are meaningful and invisible.

    Rewrapping a paragraph that contains one would silently delete it, so the
    block is reported as two spans with the break protected between them.
    """
    source, doc = _parse("nested.md")
    assert "hard break at the end of this line  \n" in source

    output = _round_trip("nested.md")
    assert "  \n" in output, "hard break was eaten"

    # Neither span may contain the break itself.
    hard_break = re.compile(r"(?:[ \t]{2,}|\\)\n")
    for seg in doc.segments:
        assert not hard_break.search(seg.text)


def test_setext_heading_underline_is_untouched():
    output = _round_trip("nested.md")
    assert "\n==============\n" in output


def test_echoed_list_marker_is_removed():
    """Regression: output came back as "- - A short bullet".

    The container's marker is outside the translated span, so the model never
    sees one. It writes one anyway, and splicing it under the real marker
    doubles it.
    """
    from ouroboros.cleanup import strip_echoed_markers

    out, fixed = strip_echoed_markers(
        "A short bullet with a citation [[0]].",
        "- A short bullet with a citation [[0]].",
    )
    assert out == "A short bullet with a citation [[0]]."
    assert fixed

    for bad in ("> quoted text", "## A heading", "1. an item", "\\item a bullet"):
        _, applied = strip_echoed_markers("plain text", bad)
        assert applied, f"{bad!r} was not recognised as an echoed marker"


def test_a_marker_the_source_really_had_is_kept():
    from ouroboros.cleanup import strip_echoed_markers

    out, fixed = strip_echoed_markers("- literal dash in source", "- literal dash in output")
    assert out == "- literal dash in output"
    assert not fixed
