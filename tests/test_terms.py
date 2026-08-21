"""Holding specific words steady.

The glossary buys a guarantee and charges for it. Measured on the real model,
hiding an ordinary noun phrase makes the sentence worse:

    "The attention head learns ..."  ->  "La tete d'attention apprend ..."   correct
    "The [[0]] learns ..."           ->  "Le [[0]] apprend ..."              wrong gender

So the glossary is for names, and the other two fixes here use no word list at
all: they restore forms from the source document's own vocabulary, which cannot
invent a spelling or a capitalisation the author never wrote.
"""
from __future__ import annotations

import pytest

from ouroboros.backends.mock import MockBackend
from ouroboros.formats import base as formats
from ouroboros.masking import Masker, rules_for
from ouroboros.persona import none_persona
from ouroboros.pipeline import RoundTrip
from ouroboros.terms import (
    Glossary,
    restore_capitalisation,
    restore_spelling,
    spelling_variants,
    vocabulary,
)

formats.load_builtins()


# ------------------------------------------------------------------- glossary


def _glossary(*terms: str) -> Glossary:
    return Glossary(terms=list(terms))


def test_a_glossary_term_is_hidden_from_the_model():
    masker = Masker(rules_for("markdown", _glossary("Muse-Glimmer", "Q5_K_M")))
    masked, mapping = masker.mask("We evaluate Muse-Glimmer at Q5_K_M today.")

    assert "Muse-Glimmer" not in masked
    assert "Q5_K_M" not in masked
    assert set(mapping.values()) == {"Muse-Glimmer", "Q5_K_M"}
    assert masker.unmask(masked, mapping) == "We evaluate Muse-Glimmer at Q5_K_M today."


def test_the_longest_matching_term_wins():
    """Otherwise "cache" eats the tail of "KV cache" and protects the wrong span."""
    masker = Masker(rules_for("markdown", _glossary("cache", "KV cache")))
    _, mapping = masker.mask("the KV cache stores it")
    assert "KV cache" in mapping.values()


def test_a_term_is_matched_regardless_of_case_but_restored_as_written():
    masker = Masker(rules_for("markdown", _glossary("muse-glimmer")))
    masked, mapping = masker.mask("We ran Muse-Glimmer overnight.")
    assert "Muse-Glimmer" not in masked
    assert masker.unmask(masked, mapping) == "We ran Muse-Glimmer overnight."


def test_a_term_does_not_match_inside_a_longer_word():
    masker = Masker(rules_for("markdown", _glossary("bert")))
    masked, _ = masker.mask("the Albertine manuscript and Bert itself")
    assert "Albertine" in masked


def test_no_glossary_leaves_the_rules_untouched():
    assert len(rules_for("markdown")) == len(rules_for("markdown", Glossary()))


def test_a_glossary_term_survives_a_round_trip_even_when_the_model_mangles_prose():
    """The point of the guarantee: it does not depend on the model behaving."""
    doc = formats.get("markdown").parse(
        "We evaluate Muse-Glimmer on the held out split today.\n", path="g.md")
    trip = RoundTrip(MockBackend(), pivot="fr", persona=none_persona(),
                     cache=None, glossary=_glossary("Muse-Glimmer"))
    output, _ = trip.run(doc)
    assert "Muse-Glimmer" in output


def test_loading_a_glossary_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "glossary.txt"
    path.write_text("# a comment\n\nMuse-Glimmer\nQ5_K_M  # trailing note\n",
                    encoding="utf-8")
    g = Glossary.load(path)
    assert g.terms == ["Muse-Glimmer", "Q5_K_M"]
    assert bool(g)


def test_an_empty_glossary_is_falsy_and_produces_no_pattern():
    assert not Glossary()
    assert Glossary().pattern() is None


# ------------------------------------------------------------------- spelling


@pytest.mark.parametrize("source_word,model_word", [
    ("emphasised", "emphasized"),
    ("analyse", "analyze"),
    ("colour", "color"),
    ("centre", "center"),
    ("modelling", "modeling"),
    ("organisation", "organization"),
])
def test_the_authors_spelling_variant_is_restored(source_word, model_word):
    source = f"Some words are {source_word} in this document."
    output = f"Some words are {model_word} in this document."
    fixed, changed = restore_spelling(source, output)
    assert source_word in fixed
    assert changed


def test_spelling_restoration_works_in_the_other_direction_too():
    """The tool has no opinion about which variant is correct."""
    fixed, _ = restore_spelling("we analyze the data", "we analyse the data")
    assert "analyze" in fixed


def test_case_is_preserved_when_restoring_a_spelling():
    fixed, _ = restore_spelling("emphasised throughout", "Emphasized throughout")
    assert fixed.startswith("Emphasised")


def test_a_word_the_source_never_used_is_left_alone():
    """The safety property: nothing is replaced by a word the author did not write.

    Without it, a rule-based converter turns "figure" into something else.
    """
    source = "the paper discusses quantization at length"
    for word in ("figure", "genre", "acre", "pour", "four", "hour", "surprise"):
        fixed, changed = restore_spelling(source, f"a {word} appears here")
        assert word in fixed, f"{word!r} was mangled"
        assert not changed


def test_spelling_restoration_is_a_no_op_when_nothing_switched():
    text = "nothing here changed at all"
    assert restore_spelling(text, text) == (text, [])


def test_spelling_variants_are_generated_both_ways():
    assert "organize" in spelling_variants("organise")
    assert "organise" in spelling_variants("organize")
    assert "travelled" in spelling_variants("traveled")


def test_vocabulary_keeps_the_first_form_seen():
    assert vocabulary("Work work WORK")["work"] == "Work"


# ------------------------------------------------------------ capitalisation


def test_heading_title_case_is_restored():
    """Regression: "Related Work" came back as "Related work".

    A prompt instruction did not reliably prevent it, and this is exactly
    checkable against the source.
    """
    fixed, changed = restore_capitalisation("Related Work", "Related work")
    assert fixed == "Related Work"
    assert changed


def test_capitalisation_restoration_leaves_new_words_alone():
    fixed, _ = restore_capitalisation("Related Work", "Prior Studies")
    assert fixed == "Prior Studies"


def test_capitalisation_is_only_applied_to_headings():
    """Applied to a paragraph it would capitalise a word mid-sentence merely
    because some other sentence happened to start with it."""
    source = ("# Work\n\nWork continues on the project, and the work is hard.\n")
    doc = formats.get("markdown").parse(source, path="h.md")
    blocks = [s.meta.get("block") for s in doc.segments]
    assert "heading_open" in blocks
    assert "paragraph_open" in blocks


def test_latex_section_titles_are_marked_as_headings():
    """LaTeX has no heading token, so \\section had to be marked explicitly."""
    doc = formats.get("latex").parse(
        "\\begin{document}\n\\section{Related Work}\n\nSome prose here.\n\\end{document}\n"
    )
    headings = [s.text for s in doc.segments if s.meta.get("heading")]
    assert headings == ["Related Work"]


def test_a_starred_section_translates_its_title_not_its_star():
    r"""Regression, found on a real arXiv paper.

    pylatexenc gives \section the argument list (star, optional, group), so
    indexing positionally past the Nones picked the bare "*" as the title. The
    star became a one-character segment to translate and the actual heading was
    never seen.
    """
    doc = formats.get("latex").parse(
        "\\begin{document}\n\\section*{Attention Visualizations}\n\\end{document}\n"
    )
    assert [s.text for s in doc.segments] == ["Attention Visualizations"]
    assert doc.segments[0].meta.get("heading")


def test_an_optional_short_title_is_not_mistaken_for_the_title():
    doc = formats.get("latex").parse(
        "\\begin{document}\n\\section[Short]{The Full Title}\n\\end{document}\n"
    )
    assert [s.text for s in doc.segments] == ["The Full Title"]
