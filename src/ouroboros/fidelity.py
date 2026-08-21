"""Did the round trip change what the text claims?

The obvious way to answer this is to embed both versions and compare them. That
does not work, and it is worth recording why, because the failure is not subtle.
Measured with nomic-embed-text on this project's own output:

    meaning negated  ("does not halve")     0.976
    numbers changed  (256 -> 512, 16 -> 64) 0.996
    a correct round trip of a short bullet  0.676

Sentence embeddings encode what a passage is *about*, not what it *asserts*. A
falsified number is invisible to them and a negation nearly so, while a short
correct paraphrase scores lower than either. Ranking by cosine similarity would
surface the good bullets and pass the broken claim.

The things that actually go wrong in a technical document are not fuzzy. A
number changes, a negation appears or vanishes, a hedge is firmed up, a model
identifier is altered, a clause is dropped. Every one of those is exactly
checkable against the source, so that is what this module does. No model, no
threshold to tune, and it runs in CI.

Numbers are the severe case and are wired into the validation ladder, where a
mismatch costs a retry and then a fallback. The rest are reported rather than
enforced: a shifted hedge count is worth a human glance and is too noisy to
block on.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .masking import SENTINEL_RE

#: A run of digits with optional separators, e.g. 256, 94.7, 1,000, 1 000,5.
_NUMBER = re.compile(r"\d[\d\s.,]*\d|\d")

#: Words that flip or weaken a claim. Kept small and high-signal on purpose:
#: a long list produces noise, and noise is what makes a report get ignored.
_NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "without",
    "fails", "fail", "unable", "absent",
}
_HEDGES = {
    "may", "might", "could", "possibly", "perhaps", "arguably", "seems",
    "seem", "appears", "appear", "suggests", "suggest", "likely", "probably",
    "generally", "typically", "often", "somewhat", "relatively",
}

#: An identifier that must survive verbatim: Llama-3-8B, GPT-4, ResNet50, Q5_K_M.
#: A token that starts with a letter and contains a digit somewhere, hyphens and
#: underscores included. The digit requirement is what keeps ordinary words out;
#: spanning hyphens is what catches Llama-3-8B, which \w alone stops at "Llama".
_IDENTIFIER = re.compile(r"\b[A-Za-z][\w.-]*\d[\w.-]*")

_WORD = re.compile(r"[A-Za-z']+")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass(frozen=True)
class Issue:
    kind: str
    detail: str
    #: "high" issues are corruption. "low" issues are worth a look.
    severity: str = "low"

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def _strip(text: str) -> str:
    """Drop placeholders, which are identical on both sides and carry nothing."""
    return SENTINEL_RE.sub(" ", text)


def digits_of(token: str) -> str:
    """Canonical form of a number: its digits, with all separators removed.

    Locale-proof by construction. French writes 1 000,5 where English writes
    1,000.5, and comparing separators would flag every such pair as a change
    when nothing changed at all. Comparing digit sequences catches the case that
    matters, 256 becoming 512, without inventing failures out of punctuation.

    The cost is that 1.5 and 15 look alike. That is a real blind spot, and it is
    a much rarer error than a digit changing.
    """
    return re.sub(r"[^\d]", "", token)


def numbers(text: str) -> Counter:
    return Counter(
        d for m in _NUMBER.finditer(_strip(text)) if (d := digits_of(m.group(0)))
    )


def identifiers(text: str) -> Counter:
    # The trailing character class happily swallows a sentence-ending period,
    # which would make "ResNet50" and "ResNet50." look like different models.
    return Counter(
        m.group(0).rstrip(".-") for m in _IDENTIFIER.finditer(_strip(text))
    )


def _lexicon_count(text: str, lexicon: set[str]) -> int:
    words = [w.lower() for w in _WORD.findall(_strip(text))]
    # Contractions such as "doesn't" survive tokenisation as "doesn't".
    return sum(1 for w in words if w in lexicon or w.endswith("n't"))


def sentence_count(text: str) -> int:
    return max(1, len(_SENTENCE_END.findall(_strip(text).strip())))


def compare(source: str, output: str) -> list[Issue]:
    """Every way the output's claims differ from the source's."""
    issues: list[Issue] = []

    want, got = numbers(source), numbers(output)
    if want != got:
        lost = sorted((want - got).elements())
        gained = sorted((got - want).elements())
        parts = []
        if lost:
            parts.append(f"lost {lost}")
        if gained:
            parts.append(f"gained {gained}")
        issues.append(Issue("numbers", ", ".join(parts), "high"))

    want, got = identifiers(source), identifiers(output)
    if want != got:
        lost = sorted((want - got).elements())
        gained = sorted((got - want).elements())
        parts = []
        if lost:
            parts.append(f"lost {lost}")
        if gained:
            parts.append(f"gained {gained}")
        issues.append(Issue("identifiers", ", ".join(parts), "high"))

    before, after = _lexicon_count(source, _NEGATIONS), _lexicon_count(output, _NEGATIONS)
    if before != after:
        issues.append(
            Issue("negation", f"{before} negation cue(s) became {after}", "high")
        )

    before, after = _lexicon_count(source, _HEDGES), _lexicon_count(output, _HEDGES)
    if before != after:
        direction = "firmed up" if after < before else "softened"
        issues.append(Issue("hedging", f"{before} hedge(s) became {after}, {direction}"))

    before, after = sentence_count(source), sentence_count(output)
    if abs(before - after) > max(1, before // 3):
        issues.append(Issue("structure", f"{before} sentence(s) became {after}"))

    return issues


def worst_first(results) -> list:
    """Order segment results so the most suspicious come first."""
    def rank(seg):
        issues = getattr(seg, "issues", None) or []
        high = sum(1 for i in issues if _severity(i) == "high")
        return (-high, -len(issues))

    return sorted(results, key=rank)


def _severity(issue) -> str:
    return issue.severity if isinstance(issue, Issue) else issue.get("severity", "low")
