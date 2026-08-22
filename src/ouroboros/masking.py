"""Inline protection: hide non-prose fragments behind opaque placeholders.

Block-level classification (formats/) keeps whole tables and equations away from
the model. This module handles the other half of the problem -- the citation,
the inline equation, the code span sitting *inside* a sentence we do want
translated. Those become sentinels before the model sees the text and are
restored afterwards.

The sentinel is a contract, not a hint. If a placeholder does not come back
exactly as it left, the translation is rejected rather than repaired: a segment
that loses ``[[3]]`` has lost a citation, and silently dropping it is far worse
than leaving the sentence in English.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Chosen empirically. Four candidate formats were round-tripped through French
#: and all four survived intact, so the tiebreaker was cost: [[0]] ran roughly
#: 40% faster than the Unicode-bracket and angle-bracket alternatives, because
#: ASCII digits in brackets tokenize into fewer pieces.
#:
#: A replacement must survive a round trip without being translated, reordered
#: into nonsense, spaced out, or normalised away by the tokenizer.
SENTINEL_FMT = "[[{}]]"
SENTINEL_RE = re.compile(r"\[\[(\d+)\]\]")


@dataclass(frozen=True)
class MaskResult:
    """Masked text, the placeholder mapping, and any bracketed visible labels.

    Unpacks as ``(text, mapping)`` so the common case reads as a plain tuple,
    while callers that care about layout can reach for ``groups``.
    """

    text: str
    mapping: dict[str, str]
    #: (opening, closing) placeholder pairs that surround still-visible text,
    #: such as a Markdown link label. The wrapper keeps these on one line: a
    #: line break inside "[the survey](url)" is legal but reads as damage.
    groups: list[tuple[str, str]] = field(default_factory=list)

    def __iter__(self):
        return iter((self.text, self.mapping))


@dataclass(frozen=True)
class Rule:
    """A named pattern whose matches are replaced by sentinels."""

    name: str
    pattern: re.Pattern
    #: When set, the text of this capture group stays visible to the model and
    #: only the surrounding syntax is hidden -- used for link labels, where the
    #: URL must be preserved but the human-readable text should be translated.
    keep_group: str | None = None


def _r(name: str, pattern: str, keep_group: str | None = None, flags: int = 0) -> Rule:
    return Rule(name, re.compile(pattern, flags), keep_group)


#: A label containing a backslash is not plain prose: it carries an escape such
#: as \_ or \&, or a nested macro. Exposing one to the model loses it. That is
#: how \texttt{c3\_scope} came back as \texttt{c3_scope}, which is a fatal
#: LaTeX error rather than a cosmetic drift, so any such label is hidden whole
#: rather than shown.
_LABEL_HAS_ESCAPE = re.compile(r"\\")


# Order matters: earlier rules win over later ones on overlapping matches.
MARKDOWN_RULES: list[Rule] = [
    # A task list checkbox opens the item's text. It is not CommonMark, so the
    # parser hands it over as ordinary prose, and a translator will happily move
    # it to the end of the sentence or drop the space after it.
    _r("task_marker", r"\A\[[ xX]\][ \t]+"),
    _r("code", r"`+[^`]*`+"),
    _r("math", r"\$\$?(?:\\.|[^$\\])+\$\$?"),
    _r("autolink", r"<https?://[^>\s]+>"),
    _r("image", r"!\[[^\]]*\]\([^)]*\)"),
    # Link: hide the syntax and the URL, expose the label for translation.
    _r("link", r"\[(?P<label>[^\]]*)\]\((?P<url>[^)]*)\)", keep_group="label"),
    _r("refstyle_link", r"\[(?P<label>[^\]]*)\]\[[^\]]*\]", keep_group="label"),
    _r("citation", r"\[[-@][^\]]*\]"),       # pandoc: [@smith2020], [-@smith2020]
    _r("footnote_ref", r"\[\^[^\]]*\]"),
    _r("bare_url", r"https?://\S+"),
    _r("html", r"</?[A-Za-z][^>]*>"),
    _r("entity", r"&[A-Za-z]+;|&#\d+;"),
]

#: Macros whose arguments are identifiers or paths, never prose.
_OPAQUE_MACROS = (
    "cite|citep|citet|citeauthor|citeyear|ref|eqref|autoref|cref|Cref|pageref|label"
    r"|includegraphics|input|include|bibliography|bibliographystyle|url|nocite|hyperref"
    # Monospace is code: identifiers, paths, flags. Never translate it, and in
    # particular never expose its escapes to the model.
    r"|texttt|lstinline|path|verb|code|mintinline"
)

LATEX_RULES: list[Rule] = [
    _r("comment", r"(?<!\\)%.*$", flags=re.MULTILINE),
    _r("display_math", r"\\\[(?:.|\n)*?\\\]|\$\$(?:.|\n)*?\$\$"),
    _r("inline_math", r"(?<!\\)\$(?:\\.|[^$\\])+\$|\\\((?:.|\n)*?\\\)"),
    _r("opaque_macro", rf"\\(?:{_OPAQUE_MACROS})\s*(?:\[[^\]]*\])?(?:\{{[^{{}}]*\}})+"),
    # \href{url}{text}: hide the URL, translate the visible label.
    _r("href", r"\\href\s*\{[^{}]*\}\s*\{(?P<label>[^{}]*)\}", keep_group="label"),
    # Inline formatting wraps prose. Matching the whole construct yields two
    # sentinels around a visible label; leaving it to the generic macro and
    # brace rules below would yield three, and every extra placeholder is one
    # more thing for the model to misplace.
    _r(
        "text_macro",
        r"\\(?:emph|textit|textbf|textsc|textrm|textsf|underline|text|mbox)"
        r"\s*\{(?P<label>[^{}]*)\}",
        keep_group="label",
    ),
    _r("escape", r"\\[&%$#_{}~^\\]"),
    # Bare control sequences with no arguments, e.g. \ours, \linewidth, \\.
    _r("bare_macro", r"\\[A-Za-z@]+\*?"),
    _r("brace", r"[{}]"),
]


class Masker:
    """Replaces protected fragments with sentinels and restores them afterwards."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules

    def mask(self, text: str) -> MaskResult:
        """Mask protected fragments, returning text, mapping, and label groups."""
        matches: list[tuple[int, int, str, str | None]] = []
        claimed: list[tuple[int, int]] = []

        for rule in self.rules:
            for m in rule.pattern.finditer(text):
                lo, hi = m.span()
                if lo == hi or any(lo < c_hi and c_lo < hi for c_lo, c_hi in claimed):
                    continue  # an earlier, higher-priority rule already owns this
                keep = m.group(rule.keep_group) if rule.keep_group else None
                if keep is not None and _LABEL_HAS_ESCAPE.search(keep):
                    keep = None   # hide the whole construct instead
                matches.append((lo, hi, m.group(0), keep))
                claimed.append((lo, hi))

        matches.sort()
        out: list[str] = []
        mapping: dict[str, str] = {}
        groups: list[tuple[str, str]] = []
        cur = n = 0

        for lo, hi, frag, keep in matches:
            out.append(text[cur:lo])
            if keep is None:
                sent = SENTINEL_FMT.format(n)
                mapping[sent] = frag
                out.append(sent)
                n += 1
            else:
                # Split the fragment around the visible label so the label stays
                # translatable while the syntax around it is hidden.
                at = frag.index(keep)
                pieces = []
                bracket: list[str] = []
                for part, hide in ((frag[:at], True), (keep, False), (frag[at + len(keep):], True)):
                    if not part:
                        continue
                    if hide:
                        sent = SENTINEL_FMT.format(n)
                        mapping[sent] = part
                        pieces.append(sent)
                        bracket.append(sent)
                        n += 1
                    else:
                        pieces.append(part)
                if len(bracket) == 2:
                    groups.append((bracket[0], bracket[1]))
                out.append("".join(pieces))
            cur = hi

        out.append(text[cur:])
        return MaskResult("".join(out), mapping, groups)

    @staticmethod
    def unmask(text: str, mapping: dict[str, str]) -> str:
        for sent, frag in mapping.items():
            text = text.replace(sent, frag)
        return text

    @staticmethod
    def sentinels(text: str) -> list[str]:
        return SENTINEL_RE.findall(text)


def rules_for(fmt: str, glossary=None) -> list[Rule]:
    """The masking rules for a format, with any glossary terms taking priority.

    Glossary terms go first so a protected name is claimed before a generic
    rule can take part of it, and so a term containing punctuation is not split
    by the escape or brace rules.
    """
    rules = list({"markdown": MARKDOWN_RULES, "latex": LATEX_RULES}[fmt])
    if glossary is not None:
        pattern = glossary.pattern()
        if pattern is not None:
            rules.insert(0, Rule("glossary", pattern))
    return rules
