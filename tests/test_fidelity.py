"""Fidelity checks: did the round trip change what the text claims?

The design note that matters is in fidelity.py, but it is worth restating here
because it is why this module exists in the shape it does. Cosine similarity on
sentence embeddings, measured on this project's own output, scores a negated
claim at 0.976 and a passage with 256 swapped for 512 at 0.996, while a correct
round trip of a short bullet scores 0.676. Embeddings encode topic, not
assertion. Ranking by them would surface the good bullets and pass the broken
claim, so the checks here are exact instead.
"""
from __future__ import annotations

import pytest

from ouroboros.fidelity import compare, digits_of, identifiers, numbers
from ouroboros.validate import check_numbers

BASE = ("Each bit you remove halves the number of representable values: "
        "8-bit gives 256 levels, 4-bit gives 16. We evaluate Llama-3-8B.")


def kinds(source: str, output: str) -> set[str]:
    return {i.kind for i in compare(source, output)}


# ------------------------------------------------------- corruption is caught


def test_a_changed_number_is_caught():
    """The case embeddings score at 0.996, i.e. cannot see at all."""
    changed = BASE.replace("256", "512").replace("gives 16", "gives 64")
    assert "numbers" in kinds(BASE, changed)


def test_a_dropped_clause_is_caught():
    assert kinds(BASE, "Each bit you remove halves the number of representable values.")


def test_a_negation_is_caught():
    negated = BASE.replace("halves", "does not halve")
    assert "negation" in kinds(BASE, negated)


def test_a_removed_negation_is_caught():
    source = "The model does not acquire new beliefs."
    assert "negation" in kinds(source, "The model acquires new beliefs.")


def test_an_added_hedge_is_caught():
    """A round trip that turns a claim into a suggestion has changed it."""
    hedged = BASE.replace("halves", "may halve")
    assert "hedging" in kinds(BASE, hedged)


def test_a_removed_hedge_is_caught():
    source = "This may indicate that quantization harms alignment."
    assert "hedging" in kinds(source, "This indicates that quantization harms alignment.")


def test_a_swapped_identifier_is_caught():
    """Q5_K_M to Q5_K_S carries the same digits, so numbers alone cannot see it."""
    issues = kinds("quantized at Q5_K_M today", "quantized at Q5_K_S today")
    assert "identifiers" in issues


def test_a_swapped_model_name_is_caught():
    assert kinds(BASE, BASE.replace("Llama-3-8B", "Llama-2-7B"))


# ------------------------------------------------- correct output stays quiet


def test_a_faithful_paraphrase_is_not_flagged():
    """The whole value of the check is that it stays quiet when nothing broke."""
    paraphrase = ("Every bit removed halves how many values can be represented: "
                  "8-bit yields 256 levels, 4-bit yields 16. We evaluate Llama-3-8B.")
    assert compare(BASE, paraphrase) == []


@pytest.mark.parametrize("english,french", [
    ("we saw 1,000.5 units", "nous avons vu 1 000,5 unites"),
    ("a total of 1,000 runs", "un total de 1.000 essais"),
])
def test_locale_number_formatting_is_not_a_change(english, french):
    """French writes 1 000,5 for 1,000.5. Comparing separators would flag every
    such pair, and a check that cries wolf gets switched off."""
    assert "numbers" not in kinds(english, french)


def test_digits_of_ignores_separators():
    assert digits_of("1,000.5") == digits_of("1 000,5") == "10005"
    assert digits_of("256") != digits_of("512")


def test_placeholders_are_ignored():
    """Placeholders are identical on both sides and must not contribute."""
    assert compare("see [[0]] and [[1]] for 8 results",
                   "voir [[1]] et [[0]] pour 8 resultats") == []


def test_identifiers_do_not_capture_trailing_punctuation():
    assert "ResNet50" in identifiers("we used ResNet50.")


def test_numbers_ignores_ordinary_words():
    assert numbers("no digits at all here") == numbers("none here either")


# --------------------------------------------------- the blocking check only


def test_numbers_is_the_only_check_that_blocks():
    """A wrong digit is a false claim, so it costs a retry and then a fallback.

    A shifted hedge count is worth a glance and would be maddening as a hard
    failure, so it is reported instead.
    """
    bad = check_numbers("gives 256 levels", "gives 512 levels")
    assert not bad.ok
    assert "512" in bad.detail

    good = check_numbers("gives 256 levels", "yields 256 levels")
    assert good.ok


def test_severity_marks_numbers_and_identifiers_as_high():
    severe = {i.kind for i in compare(BASE, BASE.replace("256", "512"))
              if i.severity == "high"}
    assert "numbers" in severe

    soft = {i.kind for i in compare(BASE, BASE.replace("halves", "may halve"))
            if i.severity == "low"}
    assert "hedging" in soft


# --------------------------------------------------- end to end through the pipeline


class _Corrupting:
    """A backend that damages the text in the ways that actually matter.

    Passes the outbound leg through untouched and corrupts the return leg, so
    each segment gets exactly one realistic injury.
    """

    name = "corrupting"

    def __init__(self):
        self.n = 0

    def generate(self, system, user, **kw):
        self.n += 1
        if self.n % 2 == 1:
            return user                      # outbound leg
        if "256" in user:
            return user.replace("256", "512")
        if "is faster" in user:
            return user.replace("is faster", "may be faster")
        return user


def _run_corrupting():
    from ouroboros.formats import base as formats
    from ouroboros.persona import none_persona
    from ouroboros.pipeline import RoundTrip

    formats.load_builtins()
    source = (
        "Each bit you remove halves the count: 8-bit gives 256 levels.\n\n"
        "The quantized model is faster than the baseline on every device.\n"
    )
    doc = formats.get("markdown").parse(source, path="drift.md")
    trip = RoundTrip(_Corrupting(), pivot="fr", persona=none_persona(),
                     cache=None, max_attempts=1)
    return source, *trip.run(doc)


def test_a_changed_number_never_reaches_the_output():
    """The blocking tier. Corruption costs a fallback, not a corrupted file."""
    _, output, report = _run_corrupting()

    assert "512" not in output
    assert "256" in output
    assert any("numbers" in f.reason for f in report.fallbacks)


def test_a_softened_claim_is_allowed_but_reported():
    """The reporting tier. Too fuzzy to fail a run over, too real to hide."""
    _, output, report = _run_corrupting()

    assert "may be faster" in output          # not blocked
    flagged = report.flagged
    assert len(flagged) == 1
    assert flagged[0].issues[0]["kind"] == "hedging"


def test_the_report_records_issues_as_json():
    import json

    _, _, report = _run_corrupting()
    data = json.loads(report.to_json())

    assert data["segments_with_issues"] == 1
    assert data["segments_fallback"] == 1
    issue = next(s["issues"][0] for s in data["segments"] if s["issues"])
    assert issue["severity"] == "low"
    assert "hedge" in issue["detail"]


# ------------------------------------------------- escapes break the build


def test_monospace_content_is_never_shown_to_the_model():
    r"""Regression from a real paper, and the worst kind: the output stopped
    compiling.

    \texttt was treated as prose, so its label was exposed. The model returned
    \texttt{c3\_scope} as \texttt{c3_scope}, and a bare underscore in text mode
    is a fatal LaTeX error. 51 of 83 escapes in that paper were destroyed.
    """
    from ouroboros.masking import Masker, rules_for

    masker = Masker(rules_for("latex"))
    text = r"the validator \texttt{c3\_scope.validate(s)} returns true"
    masked, mapping = masker.mask(text)

    assert "c3" not in masked
    assert "\\_" not in masked
    assert masker.unmask(masked, mapping) == text


def test_a_label_containing_an_escape_is_hidden_whole():
    """Even for a genuinely prose-bearing macro. An exposed escape is lost."""
    from ouroboros.masking import Masker, rules_for

    masker = Masker(rules_for("latex"))
    text = r"an \emph{escaped \& label} here"
    masked, mapping = masker.mask(text)

    assert "\\&" not in masked
    assert masker.unmask(masked, mapping) == text


def test_an_ordinary_label_is_still_translatable():
    """The guard must not make every emphasis opaque."""
    from ouroboros.masking import Masker, rules_for

    masked, _ = Masker(rules_for("latex")).mask(r"some \emph{ordinary emphasis} here")
    assert "ordinary emphasis" in masked


def test_a_lost_escape_blocks_the_segment():
    from ouroboros.validate import check_escapes

    bad = check_escapes(r"see \texttt{a\_b}", r"see \texttt{a_b}")
    assert not bad.ok
    assert "_" in bad.detail
    assert check_escapes(r"a \_ b", r"a \_ b").ok


def test_a_lost_escape_is_reported_at_high_severity():
    severe = {i.kind for i in compare(r"a \_ b", r"a _ b") if i.severity == "high"}
    assert "escapes" in severe


# ------------------------------------------- checks that need one language


def test_negation_is_not_checked_against_the_pivot_language():
    """Regression: 47 good segments were failed on a real paper.

    The negation lexicon is English. French negates with "ne ... pas" and
    "aucun", so "no human judgment" becomes "aucun jugement humain": one cue
    becomes zero and a correct translation is rejected.
    """
    from ouroboros.validate import validate

    verdict = validate("no human judgment is required",
                       "aucun jugement humain n'est requis",
                       same_language=False)
    assert verdict.ok, verdict.summary


def test_negation_is_checked_once_the_text_is_english_again():
    """The inversion this check exists for must still be caught."""
    from ouroboros.validate import validate

    verdict = validate(r"does \emph{not} generalize uniformly",
                       r"does not \emph{not} generalize uniformly",
                       same_language=True)
    assert not verdict.ok
    assert "negation" in verdict.summary


def test_language_independent_checks_run_on_every_leg():
    """Digits and escapes mean the same thing in any language."""
    from ouroboros.validate import validate

    assert not validate("gives 256 levels", "donne 512 niveaux",
                        same_language=False).ok
    assert not validate(r"see \_ here", "voir _ ici", same_language=False).ok
