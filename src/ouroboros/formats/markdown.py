"""Markdown prose discovery, driven by markdown-it-py source maps.

Every block token carries a line range, which converts to a character span. We
keep the spans of paragraph-like blocks and drop everything else. Anything the
parser does not emit a token for at all -- link reference definitions are the
notable case -- falls outside every span and is therefore preserved untouched,
which is exactly the behaviour we want and the reason this module reports spans
instead of rewriting an AST.
"""
from __future__ import annotations

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
    "html_block", "blockquote_open", "hr", "footnote_block_open",
}


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


class MarkdownFormat:
    name = "markdown"
    extensions = (".md", ".markdown", ".mdown")

    def parse(self, source: str, path: str | None = None) -> Document:
        md = _parser()
        tokens = md.parse(source)
        starts = _line_offsets(source)

        segments: list[Segment] = []
        depth = 0
        for i, tok in enumerate(tokens):
            if tok.nesting == -1:
                depth -= 1

            # Only top-level blocks. Nested prose (list items, blockquotes) is
            # deliberately out of scope for now: the containers carry markers
            # that are easy to damage, and skipping them is the safe default.
            if depth == 0 and tok.map and tok.type in PROSE_BLOCKS:
                inline = tokens[i + 1] if i + 1 < len(tokens) else None
                if inline is not None and inline.type == "inline" and _is_image_only(inline):
                    pass  # figure, leave alone
                else:
                    lo, hi = starts[tok.map[0]], starts[tok.map[1]]
                    # Trim the trailing newline(s) so the block separator itself
                    # is never handed to the model.
                    text = source[lo:hi].rstrip("\n")
                    hi = lo + len(text)

                    if tok.type == "heading_open":
                        # Exclude the "#" markup, which the model would otherwise
                        # be free to drop or reword. If the content cannot be
                        # located exactly, protect the whole heading rather than
                        # risk mangling it.
                        narrowed = _narrow_to_inline(source, lo, hi, inline)
                        if narrowed is None:
                            if tok.nesting == 1:
                                depth += 1
                            continue
                        lo, hi = narrowed
                        text = source[lo:hi]

                    segments.append(
                        Segment(
                            span=(lo, hi),
                            text=text,
                            meta={"block": tok.type, "lines": tuple(tok.map)},
                        )
                    )

            if tok.nesting == 1:
                depth += 1

        return Document(source=source, segments=segments, path=path, fmt=self.name)


register(MarkdownFormat())
