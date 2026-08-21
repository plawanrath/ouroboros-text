"""Wrapping restoration.

The subtlety worth testing is that wrapping runs while placeholders are still in
place. A restored fragment such as ``$O(n \\log n)$`` contains spaces, so a
wrapper that measured the masked text naively would produce lines that overflow
once the fragment is restored, and one that ran after restoration could split
the fragment in half.
"""
from __future__ import annotations

from ouroboros.rewrap import (
    detect_width,
    match_source_wrapping,
    rewrap,
    tighten_placeholder_spacing,
)


def test_detect_width_reads_a_hard_wrapped_paragraph():
    text = ("This is a plain paragraph of prose that should be translated.\n"
            "It mentions a citation and some inline math and a bare URL\n"
            "plus a footnote.")
    assert detect_width(text) == 61


def test_detect_width_ignores_a_single_line():
    assert detect_width("One long line with no wrapping at all.") is None


def test_detect_width_ignores_hand_shaped_text():
    """Wildly uneven lines were shaped deliberately and must not be reflowed."""
    assert detect_width("Short\nA very much longer line than the one above it.") is None


def test_rewrap_respects_the_width():
    text = " ".join(["word"] * 40)
    for line in rewrap(text, 40).split("\n"):
        assert len(line) <= 40


def test_rewrap_measures_placeholders_at_their_restored_length():
    """A placeholder is short; what it stands for may not be."""
    mapping = {"[[0]]": "$\\mathcal{L}(\\theta) = \\sum_{i=1}^{N} \\log p(y_i)$"}
    text = "The loss [[0]] is minimised over the training set by gradient descent."

    wrapped = rewrap(text, 40, mapping)
    restored = wrapped
    for sentinel, fragment in mapping.items():
        restored = restored.replace(sentinel, fragment)

    # The line holding the long fragment necessarily overflows, since the token
    # cannot be split, but every other line must respect the width.
    others = [ln for ln in restored.split("\n") if mapping["[[0]]"] not in ln]
    assert others, "expected more than one line"
    for line in others:
        assert len(line) <= 40


def test_rewrap_never_splits_a_placeholder():
    mapping = {"[[0]]": "\\cite{a,b,c}"}
    text = "Prior work [[0]] established this baseline result for the task."
    for line in rewrap(text, 20, mapping).split("\n"):
        assert "[[0" not in line or "[[0]]" in line


def test_match_source_wrapping_leaves_unwrapped_sources_alone():
    original = "A single line."
    translated = "A single line, translated, and now considerably longer than before."
    assert match_source_wrapping(original, translated) == translated


def test_tighten_removes_a_space_the_source_did_not_have():
    original = "and a footnote[^1]."
    mapping = {"[[0]]": "[^1]"}
    assert tighten_placeholder_spacing(original, "and a footnote [[0]].", mapping) == \
        "and a footnote[[0]]."


def test_tighten_keeps_a_space_the_source_did_have():
    original = "see [@smith2020] for details"
    mapping = {"[[0]]": "[@smith2020]"}
    translated = "see [[0]] for details"
    assert tighten_placeholder_spacing(original, translated, mapping) == translated


def test_rewrap_keeps_a_bracketed_label_on_one_line():
    """A link label must not be split, even though the break would be legal."""
    mapping = {"[[0]]": "[", "[[1]]": "](https://arxiv.org/abs/1234.5678)"}
    groups = [("[[0]]", "[[1]]")]
    text = "Previous work established the baseline. See [[0]]the survey[[1]] for more."

    wrapped = rewrap(text, 60, mapping, groups)
    for line in wrapped.split("\n"):
        assert not (("[[0]]" in line) ^ ("[[1]]" in line)), \
            f"label split across lines: {wrapped!r}"


def test_fuse_groups_survives_a_missing_closing_placeholder():
    """A model that dropped one half must not crash the wrapper."""
    from ouroboros.rewrap import _fuse_groups

    tokens = _fuse_groups("See [[0]]the survey for more.", [("[[0]]", "[[1]]")])
    assert "survey" in " ".join(tokens)


# ---------------------------------------------------------- pivot artifacts


def test_french_narrow_nbsp_before_punctuation_is_removed():
    """Regression: "\\textbf{bold} ." came back from French with U+202F.

    French typography puts a space before ; : ! ?, using a narrow no-break
    space. It survives the return trip and is wrong in English, and the ASCII
    space rules never matched it.
    """
    from ouroboros.cleanup import strip_pivot_artifacts

    out, fixed = strip_pivot_artifacts("and others are bold .")
    assert out == "and others are bold."
    assert fixed

    out, _ = strip_pivot_artifacts("really ? yes !")
    assert out == "really? yes!"


def test_cleanup_leaves_newlines_alone():
    """Newlines are structural in both formats and must survive cleanup."""
    from ouroboros.cleanup import strip_pivot_artifacts

    text = "first line\nsecond line\n\nnew paragraph"
    out, _ = strip_pivot_artifacts(text)
    assert out == text


def test_fused_group_keeps_attached_punctuation():
    """Regression: wrapping reintroduced the space cleanup had just removed.

    "[[0]]bold[[1]]." was tokenised as the group plus a separate ".", and
    rejoining with spaces produced "\\textbf{bold} ." all over again.
    """
    from ouroboros.rewrap import _fuse_groups

    tokens = _fuse_groups("others are [[0]]bold[[1]]. Next.", [("[[0]]", "[[1]]")])
    assert "[[0]]bold[[1]]." in tokens
    assert "." not in [t for t in tokens if t == "."]

    wrapped = rewrap("others are [[0]]bold[[1]]. Next sentence here.", 80,
                     {"[[0]]": "\\textbf{", "[[1]]": "}"}, [("[[0]]", "[[1]]")])
    assert "[[1]] ." not in wrapped


def test_a_paragraph_whose_last_line_is_longest_is_still_wrapped():
    """Regression: such paragraphs were reflowed into one long line.

    Prose wrapped by hand or by an editor often ends on a line a few characters
    longer than the ones above it. Requiring the last line to fit within the
    others' width rejected those, and the whole paragraph came back unwrapped.
    """
    text = ("Beta section elaborating a second wholly novel protocol, distinct in wording\n"
            "from the first, and equally absent from any cache this tool has ever written.")
    assert detect_width(text) == 77
