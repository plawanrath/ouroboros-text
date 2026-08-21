"""Removing traces of the pivot language from the returned text.

These are not style preferences and they are not the persona's business. They
are artifacts of having gone through another language, they are wrong in English
regardless of who wrote the document, and they are all mechanically detectable.
So they are stripped from every segment, on every run, before the persona is
consulted at all.

The one that motivated this module: French typography puts a space before ``;``,
``:``, ``!``, and ``?``, and it uses a narrow no-break space to do it. That
character survives the return trip and lands in English as "bold ." The ASCII
space rules never see it, because it is U+202F rather than U+0020.
"""
from __future__ import annotations

import re

#: Space characters French typography uses that English does not.
UNICODE_SPACES = {
    " ": " ",   # no-break space
    " ": " ",   # narrow no-break space
    " ": " ",   # thin space
    " ": " ",   # figure space
}

#: Whitespace that is not a newline. Newlines are structural in both formats.
_INLINE_WS = r"[^\S\n]"

_SPACE_BEFORE_PUNCT = re.compile(rf"{_INLINE_WS}+([.,;:!?%\)\]])")
_SPACE_AFTER_OPEN = re.compile(rf"([(\[]){_INLINE_WS}+")
_REPEATED_SPACE = re.compile(rf"{_INLINE_WS}{{2,}}")

#: Structural markers a model sometimes writes back even though they were never
#: in its input, because the text it was handed reads like a list item.
_ECHOED = (
    (re.compile(r"\A(?:[-*+]|\d{1,9}[.)])[ \t]+"), "list marker"),
    (re.compile(r"\A>[ \t]*"), "blockquote marker"),
    (re.compile(r"\A#{1,6}[ \t]+"), "heading marker"),
    (re.compile(r"\A\\item\b[ \t]*"), "latex item"),
)


def strip_echoed_markers(original: str, translated: str) -> tuple[str, list[str]]:
    """Remove a structural marker the model invented.

    The container's marker lives outside the translated span, so the model never
    sees one. It writes one anyway often enough to matter: handed "A short
    bullet with a citation", it returns "- A short bullet with a citation", and
    splicing that back under the real marker yields "- - A short bullet".

    Only markers absent from the input are removed. If the source genuinely
    began with one, it stays.
    """
    fixed: list[str] = []
    for pattern, label in _ECHOED:
        if pattern.match(translated) and not pattern.match(original):
            translated = pattern.sub("", translated, count=1)
            fixed.append(f"echoed {label}")
    return translated, fixed


def strip_pivot_artifacts(text: str) -> tuple[str, list[str]]:
    """Normalise spacing the pivot language introduced.

    Returns the cleaned text and a list of what was fixed, so a run can report
    that it happened rather than changing the document silently.
    """
    fixed: list[str] = []

    for bad, good in UNICODE_SPACES.items():
        if bad in text:
            text = text.replace(bad, good)
            fixed.append(f"unicode space U+{ord(bad):04X}")

    text, n = _SPACE_BEFORE_PUNCT.subn(r"\1", text)
    if n:
        fixed.append("space before punctuation")

    text, n = _SPACE_AFTER_OPEN.subn(r"\1", text)
    if n:
        fixed.append("space after opening bracket")

    text, n = _REPEATED_SPACE.subn(" ", text)
    if n:
        fixed.append("repeated space")

    return text, fixed
