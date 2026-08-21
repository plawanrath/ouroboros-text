"""LaTeX prose discovery via pylatexenc node positions.

We deliberately do not try to understand LaTeX. We walk the node tree and
collect the character runs that are unambiguously prose, treating every
construct we do not recognise as protected. The preamble is skipped entirely;
only the body of ``document`` is considered, and within it a short allowlist of
environments and macros is descended into.

Adjacent prose runs separated by blank lines are split into paragraphs, which is
the unit handed to the model: large enough to carry the context French agreement
needs, small enough that the model does not start summarising.
"""
from __future__ import annotations

import re

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec

from ..document import Document, Segment
from .base import register

#: Environments whose body is prose and should be descended into.
PROSE_ENVIRONMENTS = {
    "document", "abstract", "itemize", "enumerate", "description", "quote",
    "quotation", "center", "sloppypar",
}

#: Environments that are protected wholesale, including anything nested inside.
#: Anything not in PROSE_ENVIRONMENTS is protected anyway; this set exists to
#: make the important cases explicit and greppable.
OPAQUE_ENVIRONMENTS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", "multline",
    "multline*", "eqnarray", "eqnarray*", "displaymath", "math", "array",
    "table", "table*", "tabular", "tabularx", "longtable", "figure", "figure*",
    "verbatim", "lstlisting", "minted", "algorithm", "algorithmic", "tikzpicture",
    "thebibliography",
}

#: Macros whose n-th argument is prose. Everything else is opaque.
PROSE_MACRO_ARGS = {
    "section": 0, "subsection": 0, "subsubsection": 0, "paragraph": 0,
    "subparagraph": 0, "chapter": 0, "part": 0, "title": 0,
    "emph": 0, "textit": 0, "textbf": 0, "textsc": 0, "texttt": 0,
    "underline": 0, "footnote": 0, "text": 0,
}

#: Sectioning commands. Their argument is a title, so the author's
#: capitalisation of it is meaningful and is restored after translation.
HEADING_MACROS = {
    "section", "subsection", "subsubsection", "paragraph", "subparagraph",
    "chapter", "part", "title",
}

#: Captions are protected by requirement, so \caption is intentionally absent
#: from PROSE_MACRO_ARGS. Named here to document that the omission is a choice.
CAPTION_MACROS = {"caption", "captionof", "subcaption"}

_PARA_SPLIT = re.compile(r"\n[ \t]*\n")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_ITEM_RE = re.compile(r"\\item\b")

#: Macros pylatexenc's default database does not know, whose arguments are
#: identifiers or paths. Without a spec the walker leaves the braces as a bare
#: group, and the identifier inside it looks exactly like prose.
_EXTRA_MACROS = {
    "bibliographystyle": "{", "citep": "*[[{", "citet": "*[[{",
    "citeauthor": "{", "citeyear": "{", "autoref": "{", "cref": "{",
    "Cref": "{", "nocite": "{", "bibliographystyleplain": "{",
    "acresetall": "", "toprule": "", "midrule": "", "bottomrule": "",
}


def _latex_context():
    db = get_default_latex_context_db()
    db.add_context_category(
        "ouroboros",
        macros=[MacroSpec(name, args) for name, args in _EXTRA_MACROS.items()],
        prepend=True,
    )
    return db


def _merge_runs(runs: list[tuple[int, int]], source: str) -> list[tuple[int, int]]:
    """Rejoin prose runs that a macro merely interrupted.

    The walker ends a character run at every macro, so a single sentence
    containing \\cite and \\ref arrives as three fragments. Handing those to a
    translator separately produces nonsense: "and cites prior work" has no
    subject, and French agreement needs the whole clause.

    So consecutive runs are rejoined when the text between them is inline
    material. The resulting span covers the interrupting macros too, which is
    correct, because the masking layer hides \\cite and \\ref before the model
    sees them. A gap that crosses a blank line, an environment boundary, or a
    comment ends the paragraph and is never merged across.
    """
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(set(runs)):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            continue
        if merged and _joinable(source[merged[-1][1]:lo]):
            merged[-1] = (merged[-1][0], hi)
            continue
        merged.append((lo, hi))
    return merged


def _joinable(gap: str) -> bool:
    if _BLANK_LINE.search(gap):
        return False
    # An environment boundary separates blocks, never words in a sentence.
    if "\\begin{" in gap or "\\end{" in gap:
        return False
    # \item starts a new list entry. Merging across one would hand the model two
    # unrelated bullets as a single paragraph and invite it to blend them.
    if _ITEM_RE.search(gap):
        return False
    # Swallowing a comment into a translatable span would work, since masking
    # would hide it, but keeping comments outside prose entirely is simpler to
    # reason about and costs nothing.
    return not re.search(r"(?<!\\)%", gap)


class LatexFormat:
    name = "latex"
    extensions = (".tex", ".latex")

    def parse(self, source: str, path: str | None = None,
              fragment: bool = False) -> Document:
        """Find the prose spans in a LaTeX source.

        ``fragment`` marks a file that is included by another rather than
        standing on its own. Such a file has no ``\\begin{document}``, and the
        default behaviour of reporting no prose at all would silently skip every
        section of a multi-file paper. Traversal sets this; a file opened
        directly does not get it, because a stray .sty is far likelier than a
        fragment someone meant to translate on its own.
        """
        nodes, _, _ = LatexWalker(source, latex_context=_latex_context()).get_latex_nodes()

        body = nodes if fragment else self._document_body(nodes)
        runs: list[tuple[int, int]] = []
        headings: list[tuple[int, int]] = []
        self._collect(body, source, runs, headings)
        runs = _merge_runs(runs, source)

        segments: list[Segment] = []
        for lo, hi in runs:
            is_heading = any(h_lo >= lo and h_hi <= hi for h_lo, h_hi in headings)
            for seg in self._paragraphs(source, lo, hi):
                if is_heading:
                    seg.meta["heading"] = True
                segments.append(seg)

        return Document(source=source, segments=segments, path=path, fmt=self.name)

    # ------------------------------------------------------------------ walk

    def _document_body(self, nodes) -> list:
        """Return the contents of \\begin{document}, or [] if there is none.

        A file with no document environment is treated as having no prose at
        all rather than guessing -- it is most likely an included fragment or a
        style file, and translating a preamble would be destructive.
        """
        for n in nodes:
            if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document":
                return n.nodelist or []
        return []

    def _collect(self, nodes, source: str, out: list[tuple[int, int]],
                 headings: list[tuple[int, int]] | None = None,
                 in_heading: bool = False) -> None:
        for n in nodes or []:
            if n is None:
                continue

            if isinstance(n, LatexCharsNode):
                raw = source[n.pos:n.pos + n.len]
                if raw.strip():
                    # Trim to the non-whitespace extent. A run that carries its
                    # leading newlines would hide a paragraph break inside
                    # itself, and _merge_runs would then join across it.
                    lead = len(raw) - len(raw.lstrip())
                    trail = len(raw) - len(raw.rstrip())
                    span = (n.pos + lead, n.pos + n.len - trail)
                    out.append(span)
                    if in_heading and headings is not None:
                        headings.append(span)

            elif isinstance(n, LatexEnvironmentNode):
                if n.environmentname in PROSE_ENVIRONMENTS:
                    self._collect(n.nodelist, source, out, headings)
                # else: opaque, contributes nothing

            elif isinstance(n, LatexMacroNode):
                idx = PROSE_MACRO_ARGS.get(n.macroname)
                if idx is None:
                    continue

                # Count only brace-delimited arguments. pylatexenc gives
                # \section the argument list (star, optional, group), so
                # \section*{Title} puts a bare "*" where a positional index
                # expects the title: the "*" became a one-character segment to
                # translate and the real heading was never seen at all. The
                # optional [short title] is skipped for the same reason, being
                # a running-head variant rather than the title itself.
                braces = [
                    a for a in (n.nodeargd.argnlist if n.nodeargd else [])
                    if isinstance(a, LatexGroupNode)
                    and getattr(a, "delimiters", None) == ("{", "}")
                ]
                if idx < len(braces):
                    # Descend into the group's contents so the surrounding
                    # braces stay outside the translated span.
                    heading = in_heading or n.macroname in HEADING_MACROS
                    self._collect(braces[idx].nodelist, source, out, headings, heading)

            # A bare group is deliberately NOT descended into. In a document
            # body it is almost always the argument of a macro this parser does
            # not have a spec for, and its contents are then an identifier or a
            # path rather than prose. Treating it as protected costs at most a
            # missed translation; treating it as prose corrupts a \label.

            # LatexMathNode, LatexCommentNode and anything else: protected.

    # ------------------------------------------------------------ paragraphs

    def _paragraphs(self, source: str, lo: int, hi: int):
        """Split a prose run on blank lines, trimming surrounding whitespace."""
        text = source[lo:hi]
        cur = 0
        for part in _PARA_SPLIT.split(text):
            at = text.index(part, cur)
            cur = at + len(part)
            start = lo + at
            stripped = part.strip()
            if not stripped:
                continue
            offset = part.index(stripped)
            span = (start + offset, start + offset + len(stripped))
            yield Segment(span=span, text=stripped, meta={"block": "paragraph"})


register(LatexFormat())
