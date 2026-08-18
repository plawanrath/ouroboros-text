"""The core data model: a document is its original source text plus a list of
character spans that hold translatable prose.

The single most important property of this design is what it *omits*. We never
enumerate protected content and we never re-render the document from a parse
tree. We enumerate only the prose spans, translate those, and splice the results
back into the original string. Everything we failed to identify as prose --
tables, figures, captions, math, citations, comments, and anything a parser
handed back that we did not understand -- is therefore preserved byte for byte
by construction rather than by care.

That inverts the usual failure mode. A parser gap makes us translate less, never
corrupt more.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Span = tuple[int, int]


@dataclass(frozen=True)
class Segment:
    """A run of translatable prose, located by character offset in the source."""

    span: Span
    text: str
    kind: str = "prose"
    #: Free-form provenance for debugging and reporting, e.g. {"block": "paragraph"}.
    meta: dict = field(default_factory=dict)

    @property
    def start(self) -> int:
        return self.span[0]

    @property
    def end(self) -> int:
        return self.span[1]

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class Document:
    """Original source plus the prose spans discovered inside it."""

    source: str
    segments: list[Segment]
    path: str | None = None
    fmt: str | None = None

    def __post_init__(self) -> None:
        self.segments.sort(key=lambda s: s.span)
        self._validate()

    def _validate(self) -> None:
        """Segments must be disjoint, in order, and quote the source exactly.

        A violation here means a format module is buggy in a way that would
        silently corrupt output, so it is worth failing loudly and early.
        """
        prev_end = -1
        for seg in self.segments:
            if seg.start < prev_end:
                raise ValueError(f"overlapping segments at {seg.span}")
            if seg.start > seg.end:
                raise ValueError(f"inverted span {seg.span}")
            quoted = self.source[seg.start:seg.end]
            if quoted != seg.text:
                raise ValueError(
                    f"segment at {seg.span} does not quote the source:\n"
                    f"  source: {quoted[:60]!r}\n"
                    f"  text:   {seg.text[:60]!r}"
                )
            prev_end = seg.end

    @property
    def protected_spans(self) -> list[Span]:
        """The complement of the prose spans -- everything we must not touch."""
        out: list[Span] = []
        cur = 0
        for seg in self.segments:
            if seg.start > cur:
                out.append((cur, seg.start))
            cur = seg.end
        if cur < len(self.source):
            out.append((cur, len(self.source)))
        return out

    def render(self, replacements: dict[Span, str]) -> str:
        """Rebuild the document, substituting new text for the given prose spans.

        Spans absent from ``replacements`` keep their original text, which is how
        a segment that fails validation falls back safely.
        """
        parts: list[str] = []
        cur = 0
        for seg in self.segments:
            parts.append(self.source[cur:seg.start])
            parts.append(replacements.get(seg.span, seg.text))
            cur = seg.end
        parts.append(self.source[cur:])
        return "".join(parts)

    def unchanged(self) -> str:
        """Round-trip with no replacements. Must equal the source; used in tests."""
        return self.render({})


def merge_adjacent(spans: Iterable[Span], source: str, gap_pattern: str = " \t") -> list[Span]:
    """Join spans separated only by characters in ``gap_pattern``.

    Inline parsing tends to fragment a sentence into several text runs; merging
    them back gives the model a whole clause instead of a shred of one.
    """
    out: list[Span] = []
    for span in sorted(spans):
        if out and all(c in gap_pattern for c in source[out[-1][1]:span[0]]):
            out[-1] = (out[-1][0], span[1])
        else:
            out.append(span)
    return out
