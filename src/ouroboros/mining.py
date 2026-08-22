"""Learning which terms a round trip cannot carry.

A glossary protects terms that must survive verbatim, but writing one by hand
means guessing which words a translator will mangle, and guessing badly is easy.
Two attempts at guessing from shape both failed here: hyphenated compounds turned
out to appear in 82% of the segments that came back fine and 6% of the ones that
drifted, and a word-count rule fixed headings while leaving terminology
untouched.

So this does not guess. It reads a finished run and asks a much simpler question
of every candidate term: when the source contained it, how often did the output
still contain it? A term of art that French has no word for comes back as a
synonym nearly every time. An ordinary word that merely happens to be reworded
in one sentence survives in the next twenty.

That ratio is the whole method. It needs no dictionary, no stopword list, and no
assumption about which languages are involved, and it improves as the corpus
grows, because every run is more evidence about the same vocabulary.

The output is a glossary file, so the second pass protects those terms by
masking them, exactly as if they had been written by hand.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: Terms worth considering: coined vocabulary, recognised by its shape rather
#: than by a dictionary. A hyphenated compound, a capitalised multi-word phrase,
#: or a token carrying internal capitals or digits is something an author built.
_HYPHENATED = re.compile(r"\b[A-Za-z][\w]*(?:-[\w]+)+\b")
_CAPPHRASE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
_COINED = re.compile(r"\b[A-Za-z]+(?:[A-Z]|[0-9])[\w]*\b")

#: Ordinary words are deliberately excluded, and the reason is worth recording.
#: Mining them found "ladder" and "retry", which were genuinely terms of art in
#: one paper, but alongside "ships" and "through". "ships" scored 0 survivals
#: out of 6 purely because "new dialects ship per domain" becomes "are shipped",
#: which is correct English and not a defect at all. A survival rate cannot tell
#: a term of art from a verb that simply inflects, so the shape filter does that
#: job instead. The cost is recall: a single-word term of art is missed, and has
#: to be added to the glossary by hand.

#: A term must be seen this many times before its survival rate means anything.
#: One loss is an anecdote.
MIN_OCCURRENCES = 3

#: Protect a term when it fails to survive more often than this.
MAX_SURVIVAL = 0.5


@dataclass
class TermStat:
    term: str
    seen: int
    survived: int

    @property
    def survival(self) -> float:
        return self.survived / self.seen if self.seen else 1.0

    def __str__(self) -> str:
        return (f"{self.term}  (survived {self.survived}/{self.seen}, "
                f"{100 * self.survival:.0f}%)")


def _candidates(text: str) -> set[str]:
    out = {m.group(0) for m in _HYPHENATED.finditer(text)}
    out |= {m.group(0) for m in _CAPPHRASE.finditer(text)}
    out |= {m.group(0) for m in _COINED.finditer(text)}
    # A capitalised phrase subsumes its words; keep the longest form only, so
    # the glossary protects "Responsible Release" rather than "Release" alone.
    return {t for t in out if not any(t != o and t in o for o in out)}


def measure(pairs) -> dict[str, TermStat]:
    """Count how often each candidate term survived, over (source, output) pairs."""
    seen: Counter = Counter()
    survived: Counter = Counter()

    for source, output in pairs:
        lowered = output.lower()
        for term in _candidates(source):
            seen[term] += 1
            if term.lower() in lowered:
                survived[term] += 1

    return {
        term: TermStat(term, n, survived[term])
        for term, n in seen.items()
    }


def mine(pairs, min_occurrences: int = MIN_OCCURRENCES,
         max_survival: float = MAX_SURVIVAL) -> list[TermStat]:
    """Terms that a round trip usually loses, worst first."""
    stats = measure(pairs)
    poor = [
        s for s in stats.values()
        if s.seen >= min_occurrences and s.survival <= max_survival
    ]
    return sorted(poor, key=lambda s: (s.survival, -s.seen))


def pairs_from_reports(directory: Path | str):
    """Read (source, output) pairs from the report.json files a run writes."""
    out = []
    for report in sorted(Path(directory).rglob("*.report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            if seg.get("translated") and seg["final"] != seg["original"]:
                out.append((seg["original"], seg["final"]))
    return out


def render(stats: list[TermStat], source: str = "a completed run",
           min_occurrences: int = MIN_OCCURRENCES,
           max_survival: float = MAX_SURVIVAL) -> str:
    """A glossary file, with the evidence for every entry kept alongside it."""
    lines = [
        "# Terms this round trip could not carry, mined from " + source + ".",
        "#",
        (f"# Each was seen at least {min_occurrences} times and survived at "
         f"most {100 * max_survival:.0f}% of them."),
        "#",
        "# Protecting a term means the model never sees it, so review the list:",
        "# a term that should be translated does not belong here.",
        "",
    ]
    for s in stats:
        lines.append(f"{s.term}    # survived {s.survived}/{s.seen}")
    return "\n".join(lines) + "\n"
