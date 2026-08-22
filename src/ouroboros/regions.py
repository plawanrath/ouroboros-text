"""Regions the author has marked as off limits.

Some text in a document is not the author's to reword. A NeurIPS checklist
carries sixteen questions whose wording the template prescribes; a licence
notice, a quoted passage, a reviewer's own words, an author list: all of them
are prose by every structural test this tool applies, and all of them must come
back exactly as they went out.

The markers are comments, so they are invisible in the rendered document and
cost nothing to leave in permanently:

    % ouroboros: off
    ... text that must not change ...
    % ouroboros: on

    <!-- ouroboros: off -->
    ... text that must not change ...
    <!-- ouroboros: on -->

Filtering happens after a format has found its prose, by dropping whole
segments that fall inside a disabled region. That is deliberate. It reuses the
property the whole design rests on, which is that anything not reported as prose
is passed through untouched, so a disabled region needs no special handling
anywhere else in the pipeline: it simply stops being translatable.

An unclosed ``off`` runs to the end of the file. Someone who disables a region
and forgets to re-enable it gets less translation than they meant, which is the
harmless direction to be wrong in.
"""
from __future__ import annotations

import re

#: A marker inside a comment. The comment syntax differs per format, but the
#: payload is the same, so each format supplies only its own comment shape.
LATEX_MARKER = re.compile(r"^[ \t]*%+[ \t]*ouroboros:[ \t]*(off|on)\b.*$", re.MULTILINE | re.IGNORECASE)
MARKDOWN_MARKER = re.compile(
    r"^[ \t]*<!--[ \t]*ouroboros:[ \t]*(off|on)[ \t]*-->[ \t]*$", re.MULTILINE | re.IGNORECASE
)

MARKERS = {"latex": LATEX_MARKER, "markdown": MARKDOWN_MARKER}


def disabled_spans(source: str, fmt: str) -> list[tuple[int, int]]:
    """Character ranges the author marked as off limits.

    Nested or repeated ``off`` markers do not stack: the first one opens a
    region and the next ``on`` closes it. Treating them as a counter would mean
    a stray second ``off`` silently swallowed the rest of the document.
    """
    pattern = MARKERS.get(fmt)
    if pattern is None:
        return []

    spans: list[tuple[int, int]] = []
    start: int | None = None

    for m in pattern.finditer(source):
        if m.group(1).lower() == "off":
            if start is None:
                start = m.start()
        elif start is not None:
            spans.append((start, m.end()))
            start = None

    if start is not None:
        spans.append((start, len(source)))
    return spans


def enabled(segments, source: str, fmt: str):
    """Drop every segment lying inside a disabled region."""
    spans = disabled_spans(source, fmt)
    if not spans:
        return list(segments)

    return [
        seg for seg in segments
        if not any(lo <= seg.start and seg.end <= hi for lo, hi in spans)
    ]
