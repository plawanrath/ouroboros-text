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


DEFAULT_CHECKS = (
    check_nonempty,
    check_sentinels,
    check_no_control_tokens,
    check_chatter,
    check_length_ratio,
)


def validate(source: str, output: str, checks=DEFAULT_CHECKS) -> Verdict:
    results = [c(source, output) for c in checks]
    return Verdict(ok=all(r.ok for r in results), checks=results)
