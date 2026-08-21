"""Markdown prose discovery, driven by markdown-it-py source maps.

Every block token carries a line range, which converts to a character span. We
keep the spans of paragraph-like blocks and drop everything else. Anything the
parser does not emit a token for at all -- link reference definitions are the
notable case -- falls outside every span and is therefore preserved untouched,
which is exactly the behaviour we want and the reason this module reports spans
instead of rewriting an AST.

Prose inside containers counts. A blog post is often half bullet list, and an
earlier version of this module looked only at top-level blocks, which left 93%
of such a document untranslated. Descending costs one complication: the block's
first line carries the container's markers, and its continuation lines carry a
matching prefix that must be reproduced or the list falls apart. Both are
handled by reporting a span that starts after the marker and recording the
prefix that continuation lines need, rather than by rewriting anything.
"""
from __future__ import annotations

import re

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

from ..document import Document, Segment
from .base import register

#: Block tokens whose content is running prose we want translated.
PROSE_BLOCKS = {"paragraph_open", "heading_open"}

#: Everything else is protected; listed here only to document intent.
PROTECTED_BLOCKS = {
    "front_matter", "fence", "code_block", "table_open", "math_block",
    "html_block", "hr",
}

#: Container markers that may precede prose on a block's first line: any number
#: of blockquote markers, then at most one list bullet or ordered marker.
_CONTAINER_MARKER = re.compile(
    r"\A(?:[ \t]*>)*[ \t]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?"
)

#: Footnote definition label, e.g. "[^1]: ". The label is an identifier and the
#: text after it is prose.
_FOOTNOTE_LABEL = re.compile(r"\A\[\^[^\]]*\]:[ \t]*")

#: A hard line break: two or more trailing spaces, or a trailing backslash.
#: It renders as <br> and is meaningful, but it is invisible whitespace that
#: rewrapping would silently eat.
_HARD_BREAK = re.compile(r"(?:[ \t]{2,}|\\)\n")


def _parser() -> MarkdownIt:
    return (
        MarkdownIt("commonmark")
        .enable("table")
        .enable("strikethrough")
        .use(front_matter_plugin)
        .use(footnote_plugin)
        .use(dollarmath_plugin, allow_labels=True, double_inline=True)
    )


def _line_offsets(source: str) -> list[int]:
    offsets, acc = [], 0
    for line in source.splitlines(keepends=True):
        offsets.append(acc)
        acc += len(line)
    offsets.append(acc)
    return offsets


def _is_image_only(inline_token) -> bool:
    """A paragraph holding nothing but an image is a figure, not prose.

    Its alt text is a caption, and captions are protected by requirement.
    """
    children = [c for c in (inline_token.children or []) if c.type != "text" or c.content.strip()]
    return bool(children) and all(c.type in {"image", "softbreak"} for c in children)


def _narrow_to_inline(source: str, lo: int, hi: int, inline) -> tuple[int, int] | None:
    """Locate a heading's inline content inside its block span, excluding markup."""
    if inline is None or inline.type != "inline":
        return None
    content = inline.content.strip()
    if not content:
        return None
    at = source.find(content, lo, hi)
    if at < 0:
        return None
    return at, at + len(content)


def _content_start(source: str, line_start: int, block_end: int) -> int:
    """Offset of the first prose character, past any container markers.

    Deliberately regex-driven rather than derived from the inline token's text.
    The token has escapes already resolved, so locating it in the source fails
    on any paragraph containing a backslash escape, and failing means losing the
    paragraph. The marker grammar is small and fixed, so matching it is both
    more robust and easier to reason about.
    """
    line_end = source.find("\n", line_start)
    if line_end < 0 or line_end > block_end:
        line_end = block_end

    first_line = source[line_start:line_end]
    at = line_start + _CONTAINER_MARKER.match(first_line).end()

    # A footnote definition's label is an identifier, not prose.
    label = _FOOTNOTE_LABEL.match(source[at:line_end])
    if label:
        at += label.end()
    return at


def _continuation_prefix(source: str, line_start: int, content_start: int,
                         block_text: str) -> str:
    """The prefix every continuation line of this block must carry.

    Prefer what the author actually used. If the block already spans more than
    one line, its second line shows the intended prefix directly, and copying it
    avoids rewriting indentation that was already correct: a footnote continued
    with four spaces should stay at four, not be normalised to the six that its
    "[^1]: " label would imply.

    A single-line block has nothing to sample, so the prefix is derived from the
    first line's markers instead, by keeping blockquote markers and blanking
    everything else. "- " becomes two spaces and "> - " becomes ">" plus three.
    That case matters: a one-line bullet may need two lines after translation,
    and without a derived prefix the second line would escape its list.
    """
    lines = block_text.split("\n")
    if len(lines) > 1:
        sampled = re.match(r"[ \t>]*", lines[1]).group(0)
        if sampled:
            return sampled

    marker = source[line_start:content_start]
    return "".join(c if c == ">" else " " for c in marker)


def _split_hard_breaks(source: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split a block at hard line breaks, leaving the breaks outside every span.

    A hard break is two trailing spaces, which is both semantically meaningful
    and invisible. Rewrapping a paragraph that contains one would silently
    delete it. Reporting the lines on either side as separate spans leaves the
    break itself in protected territory, where nothing can touch it, and still
    gets both lines translated.
    """
    pieces: list[tuple[int, int]] = []
    cur = start
    for m in _HARD_BREAK.finditer(source, start, end):
        if source[cur:m.start()].strip():
            pieces.append((cur, m.start()))
        cur = m.end()
        # The next line reopens with its container's prefix, which belongs
        # outside the span exactly as the first line's marker does.
        while cur < end and source[cur] in " \t>":
            cur += 1
    if source[cur:end].strip():
        pieces.append((cur, end))
    return pieces


class MarkdownFormat:
    name = "markdown"
    extensions = (".md", ".markdown", ".mdown")

    def parse(self, source: str, path: str | None = None, **kwargs) -> Document:
        # Markdown has no include mechanism to speak of, so format-specific
        # options such as `fragment` do not apply and are accepted and ignored.
        md = _parser()
        tokens = md.parse(source)
        starts = _line_offsets(source)

        segments: list[Segment] = []
        for i, tok in enumerate(tokens):
            if tok.type not in PROSE_BLOCKS or not tok.map:
                continue

            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            if inline is None or inline.type != "inline":
                continue
            if _is_image_only(inline):
                continue  # figure, leave alone

            lo, hi = starts[tok.map[0]], starts[tok.map[1]]
            text = source[lo:hi].rstrip("\n")
            hi = lo + len(text)
            if not text.strip():
                continue

            if tok.type == "heading_open":
                # Exclude the "#" markup, which the model would otherwise be
                # free to drop or reword. If the content cannot be located
                # exactly, protect the whole heading rather than mangle it.
                narrowed = _narrow_to_inline(source, lo, hi, inline)
                if narrowed is None:
                    continue
                start, end = narrowed
                prefix = ""
            else:
                start = _content_start(source, lo, hi)
                end = hi
                prefix = _continuation_prefix(source, lo, start, source[start:end])

            if start >= end:
                continue

            for piece_start, piece_end in _split_hard_breaks(source, start, end):
                segments.append(
                    Segment(
                        span=(piece_start, piece_end),
                        text=source[piece_start:piece_end],
                        meta={
                            "block": tok.type,
                            "lines": tuple(tok.map),
                            "indent": prefix,
                        },
                    )
                )

        return Document(source=source, segments=segments, path=path, fmt=self.name)


register(MarkdownFormat())
