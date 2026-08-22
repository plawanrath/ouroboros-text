"""Regions the author marked as off limits.

Some prose is not the author's to reword: a NeurIPS checklist's prescribed
questions, a licence notice, a quoted passage. Every structural test says it is
prose, and it must still come back byte-identical.
"""
from __future__ import annotations

import pytest

from ouroboros.backends.mock import MockBackend
from ouroboros.formats import base as formats
from ouroboros.persona import none_persona
from ouroboros.pipeline import RoundTrip
from ouroboros.regions import disabled_spans

formats.load_builtins()

LATEX = """\
\\begin{document}
This opening paragraph is ordinary prose and should be translated.

% ouroboros: off
Do the main claims made in the abstract accurately reflect the contributions?
% ouroboros: on

This closing paragraph is ordinary prose and should be translated.
\\end{document}
"""

MARKDOWN = """\
This opening paragraph is ordinary prose and should be translated.

<!-- ouroboros: off -->
Prescribed wording that the template mandates and nobody may reword.
<!-- ouroboros: on -->

This closing paragraph is ordinary prose and should be translated.
"""


def _round_trip(source: str, fmt: str) -> str:
    doc = formats.get(fmt).parse(source, path=f"t.{fmt}")
    trip = RoundTrip(MockBackend(), pivot="fr", persona=none_persona(), cache=None)
    output, _ = trip.run(doc)
    return output


@pytest.mark.parametrize("source,fmt,frozen", [
    (LATEX, "latex", "Do the main claims made in the abstract"),
    (MARKDOWN, "markdown", "Prescribed wording that the template mandates"),
])
def test_a_disabled_region_is_not_translated(source, fmt, frozen):
    doc = formats.get(fmt).parse(source, path=f"t.{fmt}")
    assert not [s for s in doc.segments if frozen in s.text]

    output = _round_trip(source, fmt)
    assert frozen in output, "the frozen text was reworded"


@pytest.mark.parametrize("source,fmt", [(LATEX, "latex"), (MARKDOWN, "markdown")])
def test_prose_outside_the_region_is_still_translated(source, fmt):
    """A do-not-translate marker must not quietly disable the whole document."""
    doc = formats.get(fmt).parse(source, path=f"t.{fmt}")
    texts = " ".join(s.text for s in doc.segments)
    assert "opening paragraph" in texts
    assert "closing paragraph" in texts

    output = _round_trip(source, fmt)
    assert output != source


@pytest.mark.parametrize("source,fmt", [(LATEX, "latex"), (MARKDOWN, "markdown")])
def test_the_markers_themselves_survive(source, fmt):
    """They are comments, so they cost nothing to leave in permanently."""
    output = _round_trip(source, fmt)
    assert "ouroboros: off" in output
    assert "ouroboros: on" in output


def test_an_unclosed_region_runs_to_the_end_of_the_file():
    """Forgetting to re-enable gives less translation, which is the harmless
    direction to be wrong in."""
    source = ("\\begin{document}\nTranslate this paragraph of prose.\n\n"
              "% ouroboros: off\nDo not touch this one.\n\nNor this one.\n"
              "\\end{document}\n")
    doc = formats.get("latex").parse(source, path="t.tex")
    texts = " ".join(s.text for s in doc.segments)
    assert "Translate this" in texts
    assert "Do not touch" not in texts
    assert "Nor this one" not in texts


def test_a_stray_second_off_does_not_swallow_the_document():
    """Markers do not nest. Counting them would mean a duplicated `off`
    silently disabled everything after it."""
    source = ("\\begin{document}\n"
              "% ouroboros: off\nFrozen text here.\n% ouroboros: off\n"
              "% ouroboros: on\nThis paragraph must still be translated.\n"
              "\\end{document}\n")
    doc = formats.get("latex").parse(source, path="t.tex")
    texts = " ".join(s.text for s in doc.segments)
    assert "must still be translated" in texts
    assert "Frozen text here" not in texts


@pytest.mark.parametrize("marker", [
    "% ouroboros: off", "%ouroboros:off", "%% ouroboros:  OFF",
])
def test_marker_spelling_is_forgiving(marker):
    source = f"\\begin{{document}}\n{marker}\nFrozen.\n\\end{{document}}\n"
    assert disabled_spans(source, "latex")


def test_a_document_without_markers_is_unaffected():
    source = "\\begin{document}\nOrdinary prose here.\n\\end{document}\n"
    assert disabled_spans(source, "latex") == []


def test_a_percent_sign_in_prose_is_not_a_marker():
    """An escaped percent is data, and \\SI{95}{\\percent} style text is common."""
    source = "\\begin{document}\nWe report 95\\% accuracy overall.\n\\end{document}\n"
    assert disabled_spans(source, "latex") == []
    doc = formats.get("latex").parse(source, path="t.tex")
    assert doc.segments
