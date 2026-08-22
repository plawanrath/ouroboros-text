"""Learning which terms a round trip cannot carry.

Two attempts at guessing this from shape both failed on a real paper, so the
method measures instead: for every candidate term, how often did the output
still contain it? These tests pin down the two judgement calls that make the
measurement usable, which are what counts as a candidate and how much evidence
is enough.
"""
from __future__ import annotations

from ouroboros.mining import measure, mine, render


def test_a_term_that_never_survives_is_mined():
    pairs = [(f"the {t} matters here", "le truc importe ici")
             for t in ["constraint-layer"] * 4]
    assert [s.term for s in mine(pairs)] == ["constraint-layer"]


def test_a_term_that_always_survives_is_left_alone():
    pairs = [("we use BPE-boundary alignment", "we use BPE-boundary alignment")] * 5
    assert mine(pairs) == []


def test_one_loss_is_not_enough_evidence():
    """A term seen once and lost once is an anecdote, not a pattern."""
    pairs = [("the per-step budget", "the budget per step")]
    assert mine(pairs) == []


def test_an_ordinary_verb_is_not_a_candidate():
    """Regression: mining plain words scored "ships" at 0 survivals out of 6,
    purely because "new dialects ship per domain" becomes "are shipped".

    That is correct English, not a defect. A survival rate cannot tell a term of
    art from a verb that inflects, so the shape filter does that job.
    """
    pairs = [("new dialects ships per domain", "new dialects are shipped per domain")] * 6
    assert [s.term for s in mine(pairs)] == []


def test_capitalised_phrases_count_as_coined():
    pairs = [("see Responsible Release for detail", "see responsible publication")] * 4
    assert any("Responsible Release" in s.term for s in mine(pairs))


def test_survival_is_measured_case_insensitively():
    """A term returning with different capitalisation still survived."""
    stats = measure([("the In-Line decoder", "the in-line decoder")] * 3)
    assert stats["In-Line"].survived == 3


def test_the_threshold_is_a_ratio_not_a_count():
    """A common term losing a few times must not outrank a rare one always lost."""
    pairs = ([("use per-step budget", "use per-step budget")] * 8
             + [("use per-step budget", "use budget per step")] * 2)
    assert mine(pairs) == []          # survived 8/10


def test_the_glossary_records_its_evidence():
    pairs = [("the constraint-layer stack", "the stack of layers")] * 4
    text = render(mine(pairs))
    assert "constraint-layer" in text
    assert "survived 0/4" in text
    assert text.lstrip().startswith("#")


def test_a_mined_glossary_protects_its_terms():
    """The point of mining: the second pass masks what the first pass lost."""
    from ouroboros.masking import Masker, rules_for
    from ouroboros.terms import Glossary

    pairs = [("the constraint-layer stack", "the stack of layers")] * 4
    terms = [s.term for s in mine(pairs)]
    masker = Masker(rules_for("latex", Glossary(terms=terms)))

    masked, mapping = masker.mask("the constraint-layer stack is fixed")
    assert "constraint-layer" not in masked
    assert masker.unmask(masked, mapping) == "the constraint-layer stack is fixed"
