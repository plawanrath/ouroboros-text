"""Ask the model whether two passages assert the same thing.

The exact checks in fidelity.py catch what is mechanically checkable, and
content overlap ranks what to look at, but neither can answer the actual
question: did a claim change? This asks directly.

Judging with the same model that produced the translation is not independent,
and the result should be read with that in mind. It is still worth doing,
because "do these assert the same thing" is a different task from "translate
this", and the failure being hunted here is one the translator would have to
make consistently in both roles to hide.

Usage: python adhoc/judge.py adhoc/translated [how_many]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from ouroboros.backends.llamacpp import LlamaCppBackend  # noqa: E402
from ouroboros.modelstore import resolve  # noqa: E402

SYSTEM = """\
You are checking whether a paragraph survived a translation round trip with its
meaning intact. You will be shown the ORIGINAL and the RESULT.

Answer with one word on the first line:
SAME     - every claim, number, entity and qualification is preserved, even if
           the wording differs completely
WEAKER   - a claim was softened, hedged, strengthened, or made vaguer
CHANGED  - a claim, number, entity or relationship is different, or something
           was added or dropped

Then one short line explaining what differs. Wording changes alone are SAME.
Judge the content, never the style."""

VERDICT = re.compile(r"\b(SAME|WEAKER|CHANGED)\b", re.I)


def main(translated_dir: Path, limit: int = 25) -> int:
    segments = []
    for report in sorted(translated_dir.rglob("*.report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        for seg in data["segments"]:
            if seg["translated"] and seg["final"] != seg["original"]:
                seg["_file"] = Path(data["path"]).name
                segments.append(seg)

    # Longest first: a long paragraph makes more claims and has more to lose,
    # and a one-line heading cannot really change meaning.
    segments.sort(key=lambda s: len(s["original"]), reverse=True)
    sample = segments[:limit]

    backend = LlamaCppBackend(resolve().path, n_ctx=8192, effort="low")
    tally = {"SAME": 0, "WEAKER": 0, "CHANGED": 0, "UNCLEAR": 0}
    problems = []

    for i, seg in enumerate(sample, 1):
        prompt = (f"ORIGINAL:\n{seg['original']}\n\n"
                  f"RESULT:\n{seg['final']}")
        reply = backend.generate(SYSTEM, prompt, max_tokens=256)
        m = VERDICT.search(reply)
        verdict = m.group(1).upper() if m else "UNCLEAR"
        tally[verdict] += 1
        note = " ".join(reply.split())[:150]
        print(f"[{i:>3}/{len(sample)}] {verdict:<8} {seg['_file']} {tuple(seg['span'])}")
        if verdict != "SAME":
            problems.append((seg, verdict, note))
            print(f"          {note}")

    print("\n" + "=" * 78)
    print("JUDGE VERDICTS")
    print("=" * 78)
    total = max(len(sample), 1)
    for verdict, n in tally.items():
        print(f"  {verdict:<8} {n:>3}  ({100 * n / total:.0f}%)")

    if problems:
        print("\nSegments the judge did not call SAME:\n")
        for seg, verdict, note in problems:
            print(f"--- {verdict}  {seg['_file']}  {tuple(seg['span'])}")
            print(f"    note: {note}")
            print(f"    -  {' '.join(seg['original'].split())[:300]}")
            print(f"    +  {' '.join(seg['final'].split())[:300]}\n")

    print(f"meaning preserved on {100 * tally['SAME'] / total:.0f}% "
          f"of the {len(sample)} longest changed segments")
    return 0


if __name__ == "__main__":
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    raise SystemExit(main(Path(sys.argv[1]), limit))
