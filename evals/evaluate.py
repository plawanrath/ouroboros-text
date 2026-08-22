"""Evaluate a round trip against its source, treating one paper as a dataset.

Three questions, answered separately because they fail in different ways.

STRUCTURE is a hard constraint. Every macro, environment, cross-reference key,
math span, citation and number must be identical. A single difference here is a
failure regardless of how good the prose reads: a paper that lost a \\cite is
broken even if every sentence is beautiful.

MEANING is what the round trip is supposed to preserve. Checked exactly where
exact checking works (numbers, identifiers, negation, hedging) and by content
overlap otherwise. Content overlap is a coarse signal and is used for ranking
which segments a human should read, never as a verdict on its own.

QUALITY is whether the returned English is worth having. Measured as the rate of
segments that had to fall back to the original, plus the drift distribution.

Usage: python adhoc/evaluate.py adhoc/original adhoc/translated
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from ouroboros.fidelity import identifiers, numbers  # noqa: E402

MACRO = re.compile(r"\\[A-Za-z@]+\*?")
ENVIRON = re.compile(r"\\(?:begin|end)\s*\{([^{}]*)\}")
KEYED = re.compile(
    r"\\(label|ref|eqref|cite|citep|citet|citealp|autoref|cref|Cref)\s*\{([^{}]*)\}"
)
MATH = re.compile(r"(?<!\\)\$[^$]*\$|\\\[[\s\S]*?\\\]")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

#: Words carrying no content. Overlap on these says nothing about meaning.
STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "that", "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "being", "as", "by", "at", "from", "it", "its", "we", "our", "which", "than",
    "then", "so", "such", "can", "may", "also", "has", "have", "had", "not",
    "do", "does", "did", "if", "when", "while", "each", "all", "both", "more",
}

ESCAPE = re.compile(r"\\[&%$#_{}~^]")

#: Checks marked FATAL produce a document that does not build. Reporting those
#: as a percentage was the mistake the first run of this harness made: it
#: scored 77.8/100 for a paper whose output pdflatex refuses outright, and a
#: number that high reads like drift rather than breakage.
STRUCTURE_CHECKS = (
    ("macros", lambda t: Counter(MACRO.findall(t)), True),
    ("environments", lambda t: Counter(ENVIRON.findall(t)), True),
    ("escapes", lambda t: Counter(ESCAPE.findall(t)), True),
    ("math spans", lambda t: Counter(MATH.findall(t)), True),
    ("reference keys", lambda t: Counter(KEYED.findall(t)), False),
    ("numbers", numbers, False),
    ("identifiers", identifiers, False),
)


def content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if w.lower() not in STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main(original_dir: Path, translated_dir: Path) -> int:
    print("=" * 78)
    print("STRUCTURE  (hard constraint: any difference is a failure)")
    print("=" * 78)

    failures: list[str] = []
    fatal: list[str] = []
    files_compared = files_changed = 0

    for translated in sorted(translated_dir.rglob("*.tex")):
        rel = translated.relative_to(translated_dir)
        source = original_dir / rel
        if not source.is_file():
            failures.append(f"{rel}: no counterpart in the original")
            continue

        a = source.read_text(encoding="utf-8")
        b = translated.read_text(encoding="utf-8")
        files_compared += 1
        if a != b:
            files_changed += 1

        for label, extract, is_fatal in STRUCTURE_CHECKS:
            want, got = extract(a), extract(b)
            if want != got:
                lost = sorted(str(x) for x in (want - got).elements())[:5]
                gained = sorted(str(x) for x in (got - want).elements())[:5]
                n_lost = sum((want - got).values())
                line = (f"{rel}: {label} lost {n_lost} "
                        f"(e.g. {lost}) gained {gained}")
                failures.append(line)
                if is_fatal:
                    fatal.append(line)

    print(f"  {files_compared} .tex files compared, {files_changed} translated")
    if fatal:
        print()
        print("  " + "!" * 68)
        print("  BUILD BROKEN. These differences stop the document compiling.")
        print("  " + "!" * 68)
        for f in fatal:
            print(f"    FATAL  {f}")
    if failures:
        print(f"\n  {len(failures)} structural difference(s) in total:")
        for f in failures[:20]:
            print(f"    {f}")
    if not failures:
        print("  PASS: every macro, environment, reference key, math span,")
        print("        number and identifier is byte-identical.")

    # ---------------------------------------------------------------- meaning
    print()
    print("=" * 78)
    print("MEANING  (per segment, from the run reports)")
    print("=" * 78)

    segments = []
    for report in sorted(translated_dir.rglob("*.report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        for seg in data["segments"]:
            seg["_file"] = Path(data["path"]).name
            segments.append(seg)

    # `changed` is a property on SegmentResult, so it is absent from the
    # serialised report. Derive it.
    translated_segs = [s for s in segments
                       if s["translated"] and s["final"] != s["original"]]
    fallbacks = [s for s in segments if not s["translated"]]
    flagged = [s for s in segments if s.get("issues")]

    for seg in translated_segs:
        seg["_sim"] = jaccard(content_words(seg["original"]), content_words(seg["final"]))

    sims = sorted(s["_sim"] for s in translated_segs)
    if sims:
        mean = sum(sims) / len(sims)
        median = sims[len(sims) // 2]
        print(f"  {len(segments)} segments: {len(translated_segs)} changed, "
              f"{len(segments) - len(translated_segs) - len(fallbacks)} unchanged, "
              f"{len(fallbacks)} kept in English")
        print(f"  content-word overlap: mean {mean:.3f}, median {median:.3f}, "
              f"min {sims[0]:.3f}")
        buckets = [(0.9, "0.90+"), (0.8, "0.80-0.90"), (0.7, "0.70-0.80"),
                   (0.6, "0.60-0.70"), (0.0, "below 0.60")]
        prev = 1.01
        for lo, label in buckets:
            n = sum(1 for s in sims if lo <= s < prev)
            bar = "#" * round(40 * n / max(len(sims), 1))
            print(f"    {label:<12} {n:>4}  {bar}")
            prev = lo

    print(f"\n  exact fidelity checks flagged {len(flagged)} segment(s)")
    for seg in flagged[:10]:
        for issue in seg["issues"]:
            print(f"    [{issue['severity']}] {seg['_file']}: {issue['kind']}: "
                  f"{issue['detail']}")

    if fallbacks:
        print(f"\n  {len(fallbacks)} segment(s) kept in English:")
        reasons = Counter(s["reason"].split(":")[-1].strip()[:60] for s in fallbacks)
        for reason, n in reasons.most_common(8):
            print(f"    {n:>3}x  {reason}")

    # ---------------------------------------------------------------- quality
    print()
    print("=" * 78)
    print("LOWEST-OVERLAP SEGMENTS  (read these; low overlap is a hint, not a verdict)")
    print("=" * 78)
    for seg in sorted(translated_segs, key=lambda s: s["_sim"])[:8]:
        print(f"\n  overlap {seg['_sim']:.3f}  {seg['_file']}  {tuple(seg['span'])}")
        print(f"    -  {' '.join(seg['original'].split())[:260]}")
        print(f"    +  {' '.join(seg['final'].split())[:260]}")

    # ----------------------------------------------------------------- scores
    print()
    print("=" * 78)
    print("SCORES")
    print("=" * 78)

    if fatal:
        structure_verdict = "FAIL (does not build)"
    elif failures:
        structure_verdict = f"FAIL ({len(failures)} difference(s))"
    else:
        structure_verdict = "PASS"
    severe = sum(1 for s in flagged
                 for i in s["issues"] if i["severity"] == "high")
    total_changed = max(len(translated_segs), 1)
    meaning_score = 100.0 * (1 - severe / total_changed)
    completion = 100.0 * (len(segments) - len(fallbacks)) / max(len(segments), 1)

    print(f"  structure preserved   {structure_verdict}")
    print(f"  meaning preserved     {meaning_score:6.1f} / 100   "
          f"({severe} severe issue(s) across {total_changed} changed segments)")
    print(f"  translation completed {completion:6.1f} / 100   "
          f"({len(fallbacks)} fell back)")
    if sims:
        print(f"  mean content overlap  {100 * mean:6.1f} / 100   "
              f"(lower is expected: rewording is the point)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
