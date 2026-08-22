"""Acceptance checks for a translated segment.

Every check answers the same question: is this output safe to splice into the
document? Nothing here judges translation quality, because quality is a
judgement call and corruption is not.

A segment that fails is retried and then, if it keeps failing, left in the
original language. Leaving one paragraph untranslated is a visible, harmless
outcome. Splicing in a paragraph that quietly lost a citation is neither.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .masking import Masker

#: Openers that betray a model answering the request instead of performing it.
_CHATTER = re.compile(
    r"\A\s*(here (is|are)\b|sure[,!]|certainly[,!]|of course[,!]|"
    r"the translation\b|translation:|voici\b|bien s.r[,!]|"
    r"i (have|'ve) translated\b|below is\b)",
    re.IGNORECASE,
)


@dataclass
class Check:
    ok: bool
    name: str
    detail: str = ""


@dataclass
class Verdict:
    ok: bool
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def summary(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)


def check_sentinels(source: str, output: str) -> Check:
    """Placeholders must survive exactly: same set, same multiplicity.

    Order is deliberately not checked. Word order legitimately changes between
    languages, and a citation moving to the other end of a clause is correct
    behaviour, not corruption.
    """
    want = Counter(Masker.sentinels(source))
    got = Counter(Masker.sentinels(output))
    if want == got:
        return Check(True, "sentinels")

    missing = want - got
    extra = got - want
    bits = []
    if missing:
        bits.append(f"lost {sorted(missing.elements())}")
    if extra:
        bits.append(f"invented {sorted(extra.elements())}")
    return Check(False, "sentinels", ", ".join(bits))


def check_nonempty(source: str, output: str) -> Check:
    if output.strip():
        return Check(True, "nonempty")
    return Check(False, "nonempty", "model returned nothing")


def check_chatter(source: str, output: str) -> Check:
    """Reject a model that answered the request instead of performing it.

    Only when the source did not open the same way. A paragraph that genuinely
    begins "Here is the algorithm" translates to "Voici l'algorithme", and
    flagging that would throw away a correct translation. The check is for
    openers the model introduced, not openers it preserved.
    """
    m = _CHATTER.match(output)
    if not m:
        return Check(True, "chatter")
    if _CHATTER.match(source):
        return Check(True, "chatter")
    return Check(False, "chatter", f"conversational opener {m.group(0).strip()!r}")


def check_length_ratio(source: str, output: str, lo: float = 0.5, hi: float = 2.0) -> Check:
    """Catch truncation and runaway generation.

    Short strings are exempt: a four-word heading can legitimately double in
    length, and the ratio carries no signal at that size.
    """
    if len(source) < 40:
        return Check(True, "length")
    ratio = len(output) / max(len(source), 1)
    if lo <= ratio <= hi:
        return Check(True, "length")
    return Check(False, "length", f"ratio {ratio:.2f} outside [{lo}, {hi}]")


def check_no_control_tokens(source: str, output: str) -> Check:
    """Leaked ATEM markers mean the channel parser missed a block."""
    leaked = re.findall(r"<\|[^|]*\|>", output)
    if not leaked:
        return Check(True, "control_tokens")
    return Check(False, "control_tokens", f"leaked {sorted(set(leaked))}")


def check_numbers(source: str, output: str) -> Check:
    """Every number in the source must come back, and no new ones may appear.

    This is the one fidelity check that blocks rather than reports. A changed
    digit in a paper is not a stylistic drift, it is a false claim, and it is
    the failure an embedding-based similarity score is least able to see: a
    passage with 256 swapped for 512 scores 0.996 against its original.

    Comparison is on digit sequences with separators removed, so the French
    convention of writing 1 000,5 for 1,000.5 does not register as a change.
    """
    from .fidelity import numbers

    want, got = numbers(source), numbers(output)
    if want == got:
        return Check(True, "numbers")

    lost = sorted((want - got).elements())
    gained = sorted((got - want).elements())
    parts = []
    if lost:
        parts.append(f"lost {lost}")
    if gained:
        parts.append(f"gained {gained}")
    return Check(False, "numbers", ", ".join(parts))


#: LaTeX escapes. Losing one is not a stylistic drift, it is a document that
#: no longer compiles: a bare _ or & in text mode is a fatal error.
_ESCAPE_RE = re.compile(r"\\[&%$#_{}~^]")


def check_escapes(source: str, output: str) -> Check:
    """Every backslash escape must come back, and no new ones may appear.

    A safety net rather than the primary defence. Masking should hide escapes
    before the model ever sees one, so this firing means a masking rule has a
    gap. It is cheap, and the failure it catches is severe: on a real paper a
    gap in the \\texttt rule destroyed 51 of 83 escapes and the output would
    not build.
    """
    want = Counter(_ESCAPE_RE.findall(source))
    got = Counter(_ESCAPE_RE.findall(output))
    if want == got:
        return Check(True, "escapes")

    lost = sorted((want - got).elements())
    gained = sorted((got - want).elements())
    parts = []
    if lost:
        parts.append(f"lost {lost}")
    if gained:
        parts.append(f"invented {gained}")
    return Check(False, "escapes", ", ".join(parts))


def check_negation(source: str, output: str) -> Check:
    """The count of negation cues must not change.

    A flipped negation is a reversed claim, which is the same order of failure
    as a changed digit and belongs in the same tier. It was report-only until a
    real paper came back with "does not \\emph{not} generalize" where the source
    said "does \\emph{not} generalize": the check caught the double negative and
    the inverted sentence was written to the file anyway.
    """
    from .fidelity import negation_count

    before, after = negation_count(source), negation_count(output)
    if before == after:
        return Check(True, "negation")
    return Check(False, "negation", f"{before} negation cue(s) became {after}")


DEFAULT_CHECKS = (
    check_nonempty,
    check_sentinels,
    check_no_control_tokens,
    check_chatter,
    check_length_ratio,
    check_numbers,
    check_escapes,
    check_negation,
)

#: Checks that compare words, so they only mean anything when both sides are in
#: the same language. Counting English negation cues in French text finds none,
#: because French negates with "ne ... pas" and "aucun": "no human judgment"
#: becomes "aucun jugement humain", one cue becomes zero, and the check fails a
#: perfectly good translation. That mistake failed 47 segments on a real paper
#: before it was caught.
#:
#: Everything else here counts digits, escapes or placeholders, which mean the
#: same thing in any language and are checked at every hop.
LANGUAGE_DEPENDENT = frozenset({check_negation})


def validate(source: str, output: str, checks=DEFAULT_CHECKS,
             same_language: bool = True) -> Verdict:
    """Check an output against its source.

    ``same_language`` is False on the outbound leg, where the output is in the
    pivot language and word-level comparisons against an English source are
    meaningless.
    """
    results = [
        c(source, output) for c in checks
        if same_language or c not in LANGUAGE_DEPENDENT
    ]
    return Verdict(ok=all(r.ok for r in results), checks=results)
