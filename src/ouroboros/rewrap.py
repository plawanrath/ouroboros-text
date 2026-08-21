"""Restore the source's line wrapping to a translated paragraph.

The model returns one long line. Papers are usually hard-wrapped and tracked in
git, where reflowing a paragraph turns a one-word change into a whole-paragraph
diff, so the original shape is worth putting back.

Wrapping has to happen while the placeholders are still in place. After
restoration a paragraph contains fragments like ``$O(n \\log n)$`` and
``\\cite{a,b}`` that hold spaces but must never be split across lines, and a
whitespace-splitting wrapper would break them. Wrapping the masked text solves
that, but the placeholder and the fragment it stands for have different widths,
so each token is measured at its restored length and laid out accordingly.
"""
from __future__ import annotations

import re

from .masking import SENTINEL_RE

_WS = re.compile(r"\s+")


def strip_continuation(text: str, prefix: str) -> str:
    """Remove a container's continuation prefix from every line but the first.

    The model should see a paragraph, not a bullet's indentation. A line that
    does not carry the expected prefix is left-stripped instead, which covers
    lazy continuation lines in blockquotes.
    """
    if not prefix or "\n" not in text:
        return text

    first, *rest = text.split("\n")
    out = [first]
    for line in rest:
        if line.startswith(prefix):
            out.append(line[len(prefix):])
        else:
            out.append(line.lstrip("> \t"))
    return "\n".join(out)


def apply_continuation(text: str, prefix: str) -> str:
    """Put the container's continuation prefix back on every line but the first."""
    if not prefix or "\n" not in text:
        return text

    first, *rest = text.split("\n")
    return "\n".join([first] + [prefix + line for line in rest])


def detect_width(text: str) -> int | None:
    """Infer the wrap width of a hard-wrapped paragraph, or None if it is not.

    A single-line paragraph, or one whose lines vary wildly, is left alone: the
    author either did not wrap it or wrapped it by hand for a reason.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None

    widths = [len(ln) for ln in lines]
    width = max(widths)

    # Too narrow to be a wrap column, so the line break means something else.
    if width < 20:
        return None

    # In a wrapped paragraph every line but the last runs close to the column.
    # That is what separates one from two lines that merely sit next to each
    # other, where a short opening line would otherwise be read as the column.
    #
    # Measuring against the maximum rather than against the non-final lines
    # matters: prose wrapped by hand or by an editor often ends on a line a few
    # characters longer than the ones above it, and treating that as
    # disqualifying left whole paragraphs reflowed into one long line.
    if min(widths[:-1]) < width - 24:
        return None

    return width


def _restored_length(token: str, mapping: dict[str, str]) -> int:
    """Width this token will occupy once placeholders are restored."""
    def sub(m: re.Match) -> str:
        return mapping.get(m.group(0), m.group(0))

    return len(SENTINEL_RE.sub(sub, token))


def _fuse_groups(text: str, groups: list[tuple[str, str]] | None) -> list[str]:
    """Tokenise on whitespace, keeping each bracketed label as one token.

    "[[0]]the survey[[1]]" is a link. Breaking the line between "the" and
    "survey" is legal in both Markdown and LaTeX but reads as damage, so the
    whole construct is treated as a single unbreakable token.
    """
    if not groups:
        return [t for t in _WS.split(text.strip()) if t]

    spans: list[tuple[int, int]] = []
    for opening, closing in groups:
        start = text.find(opening)
        if start < 0:
            continue
        end = text.find(closing, start + len(opening))
        if end < 0:
            continue

        lo, hi = start, end + len(closing)
        # Absorb whatever is attached to the construct without a space, so the
        # trailing period in "[[0]]bold[[1]]." stays part of the same token.
        # Splitting it off and rejoining with spaces would reintroduce the gap
        # the cleanup pass just removed.
        while lo > 0 and not text[lo - 1].isspace():
            lo -= 1
        while hi < len(text) and not text[hi].isspace():
            hi += 1
        spans.append((lo, hi))

    spans.sort()
    tokens: list[str] = []
    cur = 0
    for lo, hi in spans:
        if lo < cur:
            continue  # overlapping or already consumed
        tokens += [t for t in _WS.split(text[cur:lo]) if t]
        tokens.append(_WS.sub(" ", text[lo:hi]).strip())
        cur = hi
    tokens += [t for t in _WS.split(text[cur:]) if t]
    return tokens


def rewrap(text: str, width: int, mapping: dict[str, str] | None = None,
           groups: list[tuple[str, str]] | None = None) -> str:
    """Greedily wrap ``text`` to ``width``, measuring placeholders at full size.

    Never splits a token, so a placeholder standing in for a multi-word fragment
    stays whole even when the fragment is wider than the target width.
    """
    mapping = mapping or {}
    tokens = _fuse_groups(text, groups)
    if not tokens:
        return text

    lines: list[list[str]] = []
    current: list[str] = []
    used = 0

    for token in tokens:
        size = _restored_length(token, mapping)
        if current and used + 1 + size > width:
            lines.append(current)
            current, used = [token], size
        else:
            used += (1 + size) if current else size
            current.append(token)

    if current:
        lines.append(current)
    return "\n".join(" ".join(line) for line in lines)


def match_source_wrapping(original: str, translated: str,
                          mapping: dict[str, str] | None = None,
                          groups: list[tuple[str, str]] | None = None) -> str:
    """Wrap ``translated`` the way ``original`` was wrapped, if it was."""
    width = detect_width(original)
    if width is None:
        return translated
    return rewrap(translated, width, mapping, groups)


def tighten_placeholder_spacing(original: str, translated: str,
                                mapping: dict[str, str]) -> str:
    """Remove a space the model inserted before a placeholder that had none.

    Markdown footnote references and LaTeX punctuation attach directly to the
    preceding word, and "footnote [^1]" is not what the author wrote. Only
    spacing the source did not have is removed, so a placeholder that legitimately
    follows a space is untouched.
    """
    for sentinel, fragment in mapping.items():
        # A fragment that already carries its own trailing space, such as a task
        # list's "[ ] ", would otherwise end up with two.
        if fragment and fragment[-1] in " \t":
            translated = re.sub(rf"{re.escape(sentinel)}[ \t]+", sentinel, translated)

        at = original.find(fragment)
        if at <= 0 or original[at - 1].isspace():
            continue  # the source had a space here, or the fragment leads
        translated = re.sub(rf"[ \t]+{re.escape(sentinel)}", sentinel, translated)
    return translated
