"""Deterministic fake backend.

The structural guarantee this project makes is that protected content survives
untouched. That is a property of the segmentation and splicing layers, not of
the model, so it should be testable without loading 19 GB of weights. This
backend makes the full pipeline runnable in CI in milliseconds.

It mangles prose visibly while preserving placeholders, which is exactly the
behaviour a real translation leg has: the words change, the sentinels do not.
"""
from __future__ import annotations

import re

from ..masking import SENTINEL_RE
from .base import register

_TOKEN = re.compile(r"(\[\[\d+\]\]|\s+|[^\s]+)")


@register("mock")
class MockBackend:
    """Rotates word order per line, leaving sentinels in their original slots.

    Rotation rather than reversal, deliberately. Reversal is an involution, so a
    two-hop round trip through it returns the input unchanged and a test that
    asserts "the prose changed" would pass while exercising nothing.
    """

    def __init__(self, *, fail_sentinels: bool = False, **kwargs) -> None:
        #: When set, drops a sentinel so the validation ladder can be tested.
        self.fail_sentinels = fail_sentinels
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        self.calls.append((system, user))

        out_lines = []
        for line in user.split("\n"):
            words = [w for w in _TOKEN.findall(line) if w.strip()]
            sentinels = [w for w in words if SENTINEL_RE.fullmatch(w)]
            plain = [w for w in words if not SENTINEL_RE.fullmatch(w)]
            plain = plain[1:] + plain[:1]

            # Re-interleave so sentinels keep their relative order but move
            # position, mimicking a real translator reordering a clause.
            merged, si, pi = [], 0, 0
            for w in words:
                if SENTINEL_RE.fullmatch(w):
                    merged.append(sentinels[si]); si += 1
                else:
                    merged.append(plain[pi]); pi += 1
            if self.fail_sentinels and sentinels:
                merged = [w for w in merged if w != sentinels[0]]
            out_lines.append(" ".join(merged))

        return "\n".join(out_lines)
