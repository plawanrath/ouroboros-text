"""Holding specific words steady across the round trip.

Three problems that all reduce to the same question: which form did the author
actually use?

A **glossary** names terms that must survive verbatim. They become placeholders,
so their survival is a property of the pipeline rather than a hope about the
model. That guarantee is not free, and the cost is worth stating plainly.
Measured on this project's own model, hiding an ordinary noun phrase makes the
sentence around it worse:

    "The attention head learns ..."  ->  "La tete d'attention apprend ..."   correct
    "The [[0]] learns ..."           ->  "Le [[0]] apprend ..."              wrong gender

With the noun hidden the model has nothing to agree with, guesses, and nearby
words drift too. A product name has no grammatical role and pays none of that,
so the glossary is for names and identifiers, not for vocabulary the model
already handles.

**Spelling variants** and **heading capitalisation** are handled without any
word list at all, by restoring from the source's own vocabulary. If the output
says "emphasized" and the author wrote "emphasised", the author's form wins.
Nothing is ever replaced by a word the author did not use, which is what makes
this safe without a dictionary and without knowing which side of the Atlantic
anyone is on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")

#: Regular spelling differences between English variants. Applied in both
#: directions, and only ever to reach a form the source document already used.
_ENDINGS = (
    ("isation", "ization"), ("isations", "izations"),
    ("ise", "ize"), ("ised", "ized"), ("ises", "izes"), ("ising", "izing"),
    ("yse", "yze"), ("ysed", "yzed"), ("yses", "yzes"), ("ysing", "yzing"),
    ("our", "or"), ("ours", "ors"),
    ("re", "er"), ("res", "ers"),
    ("ogue", "og"), ("ogues", "ogs"),
    ("ce", "se"),
    ("aeon", "eon"), ("oedema", "edema"),
)

_DOUBLED = re.compile(r"ll(ed|ing|er|ers|est)$")
_SINGLE = re.compile(r"(?<!l)l(ed|ing|er|ers|est)$")

DEFAULT_GLOSSARY = Path("glossary.txt")


# ------------------------------------------------------------------- glossary


@dataclass
class Glossary:
    """Terms that must come back exactly as they went out."""

    terms: list[str] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> Glossary:
        path = Path(path)
        terms: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                terms.append(line)
        return cls(terms=terms, path=path)

    @classmethod
    def find(cls, explicit: Path | str | None = None) -> Glossary:
        """Load the named glossary, or a glossary.txt sitting in the way."""
        if explicit:
            return cls.load(explicit)
        if DEFAULT_GLOSSARY.is_file():
            return cls.load(DEFAULT_GLOSSARY)
        return cls()

    def __bool__(self) -> bool:
        return bool(self.terms)

    def pattern(self) -> re.Pattern | None:
        """One alternation over every term, longest first.

        Longest first so that "KV cache" wins over "cache" and the shorter term
        does not eat the start of the longer one. Matching is case-insensitive
        but the source text is what gets stored and restored, so the author's
        own capitalisation comes back untouched.
        """
        if not self.terms:
            return None
        ordered = sorted(self.terms, key=len, reverse=True)
        body = "|".join(re.escape(t) for t in ordered)
        # Lookarounds rather than \b: a term may begin or end with punctuation,
        # where \b would not match at all.
        return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


# ------------------------------------------------------ source-form restoration


def vocabulary(text: str) -> dict[str, str]:
    """Map each lowercased word to the form the author actually wrote."""
    forms: dict[str, str] = {}
    for word in _WORD.findall(text):
        forms.setdefault(word.lower(), word)
    return forms


def spelling_variants(word: str) -> set[str]:
    """Plausible other-variant spellings of a lowercased word."""
    out: set[str] = set()
    for a, b in _ENDINGS:
        if word.endswith(a):
            out.add(word[: -len(a)] + b)
        if word.endswith(b):
            out.add(word[: -len(b)] + a)

    if m := _DOUBLED.search(word):
        out.add(word[: m.start()] + "l" + m.group(1))
    if m := _SINGLE.search(word):
        out.add(word[: m.start()] + "ll" + m.group(1))

    out.discard(word)
    return out


def _match_case(form: str, like: str) -> str:
    """Give ``form`` the capitalisation of ``like``."""
    if like.isupper() and len(like) > 1:
        return form.upper()
    if like[:1].isupper():
        return form[:1].upper() + form[1:]
    return form


def restore_spelling(source: str, output: str) -> tuple[str, list[str]]:
    """Put back the source's spelling variant wherever the output switched.

    Only ever substitutes a form the source document itself contains, so no
    dictionary is needed and no unrelated word can be mangled. "emphasized"
    becomes "emphasised" when the author wrote it that way, and stays put
    otherwise.
    """
    forms = vocabulary(source)
    if not forms:
        return output, []

    changed: list[str] = []

    def fix(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if lower in forms:
            return word
        for candidate in spelling_variants(lower):
            if candidate in forms:
                replacement = _match_case(forms[candidate], word)
                changed.append(f"{word} -> {replacement}")
                return replacement
        return word

    return _WORD.sub(fix, output), changed


def restore_capitalisation(source: str, output: str) -> tuple[str, list[str]]:
    """Give every word the capitalisation the source used for it.

    Meant for headings, where "Related Work" comes back as "Related work" and a
    prompt instruction does not reliably prevent it. Restricted to headings on
    purpose: applied to a paragraph it would capitalise a word mid-sentence
    merely because a sentence elsewhere began with it.
    """
    forms = vocabulary(source)
    if not forms:
        return output, []

    changed: list[str] = []

    def fix(m: re.Match) -> str:
        word = m.group(0)
        original = forms.get(word.lower())
        if original and original != word:
            changed.append(f"{word} -> {original}")
            return original
        return word

    return _WORD.sub(fix, output), changed
