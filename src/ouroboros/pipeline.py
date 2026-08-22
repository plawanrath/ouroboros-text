"""The round trip: prose out through a pivot language and back.

One segment's journey is: mask, translate to the pivot, translate back, enforce
the persona's hard rules, validate, unmask, splice. The hops are a list rather
than a hardcoded there-and-back, so a longer cycle (en, fr, de, en) costs a
config change instead of a rewrite.

The failure policy is the important part. A segment that fails validation is
retried, and if it keeps failing it is left in the original language and
recorded. Leaving one paragraph untranslated is visible and harmless. Splicing
in a paragraph that quietly lost a citation is neither.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import cleanup, fidelity, prompts, rewrap, terms
from .document import Document, Span
from .masking import Masker, rules_for
from .persona import Persona, none_persona
from .validate import validate


@dataclass
class SegmentResult:
    span: Span
    original: str
    final: str
    hops: list[str] = field(default_factory=list)
    attempts: int = 1
    translated: bool = True
    reason: str = ""
    rules_applied: list[str] = field(default_factory=list)
    #: Ways the output's claims differ from the source's. Reported, not
    #: enforced: numbers are already blocked by the validation ladder, and the
    #: rest are worth a human glance but too noisy to fail a run over.
    issues: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def severe(self) -> int:
        return sum(1 for i in self.issues if i.get("severity") == "high")

    @property
    def changed(self) -> bool:
        return self.final != self.original


@dataclass
class RunReport:
    path: str
    fmt: str
    pivot: list[str]
    persona: str
    model: str
    segments: list[SegmentResult] = field(default_factory=list)
    seconds: float = 0.0
    #: True when the run was stopped part way. The output still holds every
    #: segment finished before the interrupt; the rest is the original English.
    interrupted: bool = False

    @property
    def translated(self) -> int:
        return sum(1 for s in self.segments if s.translated)

    @property
    def fallbacks(self) -> list[SegmentResult]:
        return [s for s in self.segments if not s.translated]

    @property
    def flagged(self) -> list[SegmentResult]:
        """Translated segments whose claims differ from the source's."""
        return [s for s in self.segments if s.translated and s.issues]

    def to_json(self) -> str:
        return json.dumps(
            {
                "path": self.path,
                "format": self.fmt,
                "pivot": self.pivot,
                "persona": self.persona,
                "model": self.model,
                "seconds": round(self.seconds, 1),
                "segments_total": len(self.segments),
                "segments_translated": self.translated,
                "segments_fallback": len(self.fallbacks),
                "segments_with_issues": len(self.flagged),
                "interrupted": self.interrupted,
                "segments": [asdict(s) for s in self.segments],
            },
            indent=2,
            ensure_ascii=False,
        )


class Cache:
    """Content-addressed store for single translation hops.

    Two calls per paragraph at roughly 25 seconds each makes a re-run of a long
    paper expensive, and most re-runs change one thing. Keying on the model, the
    hop, the persona, and the exact input text makes resuming free and keeps a
    changed persona from silently reusing results generated under the old one.
    """

    def __init__(self, directory: Path | str | None = ".ouroboros-cache") -> None:
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    def get(self, *parts: str) -> str | None:
        if not self.dir:
            return None
        path = self.dir / f"{self._key(*parts)}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def has(self, *parts: str) -> bool:
        return bool(self.dir) and (self.dir / f"{self._key(*parts)}.txt").exists()

    def put(self, value: str, *parts: str) -> None:
        if not self.dir:
            return
        (self.dir / f"{self._key(*parts)}.txt").write_text(value, encoding="utf-8")


class RoundTrip:
    def __init__(
        self,
        backend,
        *,
        pivot: list[str] | str = "fr",
        persona: Persona | None = None,
        max_attempts: int = 2,
        cache: Cache | None = None,
        max_tokens: int = 2048,
        model_id: str = "unknown",
        preserve_wrapping: bool = True,
        should_stop=None,
        glossary=None,
    ) -> None:
        self.backend = backend
        self.pivot = [pivot] if isinstance(pivot, str) else list(pivot)
        self.persona = persona or none_persona()
        self.max_attempts = max_attempts
        self.cache = cache
        self.max_tokens = max_tokens
        self.model_id = model_id
        #: Restore the source's hard wrapping, so a translated paragraph does
        #: not reflow into one long line and blow up the diff.
        self.preserve_wrapping = preserve_wrapping
        #: Asked between segments. Cooperative rather than exception-driven,
        #: because a signal arriving inside a 25-second call into llama.cpp is
        #: delivered at an arbitrary moment, and stopping cleanly at a segment
        #: boundary is what makes the partial output a valid document.
        self.should_stop = should_stop
        #: Terms that must survive verbatim. Masked, so their survival is a
        #: property of the pipeline rather than a hope about the model.
        self.glossary = glossary

    # ------------------------------------------------------------------ hops

    def _hop_chain(self) -> list[tuple[str, str]]:
        """The ordered (source, target) pairs, ending back at English."""
        langs = ["en", *self.pivot, "en"]
        return list(itertools.pairwise(langs))

    def _system_for(self, src: str, dst: str) -> str:
        # Persona applies only on the leg that produces the final English,
        # which is the only output a reader sees.
        if dst == "en":
            return prompts.back_system(src, "en", persona=self.persona)
        return prompts.forward_system(dst, src)

    def _translate_once(self, text: str, src: str, dst: str, temperature: float) -> str:
        system = self._system_for(src, dst)
        payload = prompts.user_message(text)

        # The system prompt is part of the key, not just the persona name.
        # Keying on the name alone would silently serve results generated under
        # an older prompt after the prompt or the persona body was edited, which
        # is the worst kind of cache bug: invisible and wrong.
        cache_parts = (self.model_id, src, dst, system, str(temperature), payload)

        if temperature == 0.0 and self.cache:
            hit = self.cache.get(*cache_parts)
            if hit is not None:
                return hit

        out = self.backend.generate(
            system, payload, max_tokens=self.max_tokens, temperature=temperature
        )

        if temperature == 0.0 and self.cache:
            self.cache.put(out, *cache_parts)
        return out

    # -------------------------------------------------------------- segments

    def translate_segment(self, masked: str) -> tuple[str, list[str], int, str]:
        """Run one masked segment through every hop.

        Returns ``(text, hop_outputs, attempts, failure_reason)``. A non-empty
        reason means the caller should fall back to the original.
        """
        for attempt in range(1, self.max_attempts + 1):
            # A retry at temperature 0 would reproduce the same failure, so the
            # second attempt samples instead.
            temperature = 0.0 if attempt == 1 else 0.3
            current = masked
            hops: list[str] = []
            reason = ""

            for src, dst in self._hop_chain():
                current = self._translate_once(current, src, dst, temperature)
                hops.append(current)
                # Word-level checks mean nothing against the pivot language,
                # so they wait for the leg that lands back in English.
                verdict = validate(masked, current, same_language=(dst == "en"))
                if not verdict.ok:
                    reason = f"{src}->{dst}: {verdict.summary}"
                    break

            if not reason:
                return current, hops, attempt, ""

        return masked, hops, self.max_attempts, reason

    def pending(self, doc: Document) -> tuple[int, int, int]:
        """Return ``(already_cached, total, uncached_chars)``.

        A run on a real paper takes hours, and the single most useful thing to
        know before starting is how much of it is already done. Because the
        cache is keyed on the exact input to each hop, that is answerable
        without running anything: walk the chain, and stop counting the moment
        a hop is missing, since the next hop's input is that hop's output and is
        therefore unknowable.
        """
        if not self.cache:
            chars = sum(len(s.text) for s in doc.segments)
            return 0, len(doc.segments), chars

        masker = Masker(rules_for(doc.fmt or "markdown", self.glossary))
        cached = total = uncached_chars = 0

        for seg in doc.segments:
            body = rewrap.strip_continuation(seg.text, seg.meta.get("indent", ""))
            masked = masker.mask(body).text
            if not prompts.long_enough(masked):
                continue
            total += 1

            current = masked
            for src, dst in self._hop_chain():
                system = self._system_for(src, dst)
                hit = self.cache.get(self.model_id, src, dst, system, "0.0", current)
                if hit is None:
                    break
                current = hit
            else:
                cached += 1
                continue
            uncached_chars += len(masked)

        return cached, total, uncached_chars

    def run(self, doc: Document, on_segment=None) -> tuple[str, RunReport]:
        masker = Masker(rules_for(doc.fmt or "markdown", self.glossary))
        report = RunReport(
            path=doc.path or "<string>",
            fmt=doc.fmt or "",
            pivot=self.pivot,
            persona=self.persona.name,
            model=self.model_id,
        )

        replacements: dict[Span, str] = {}
        started = time.time()

        try:
            self._translate_all(doc, masker, report, replacements, on_segment)
        except KeyboardInterrupt:
            # Stop between segments rather than at an arbitrary point inside
            # one. Everything finished so far is spliced and written, and the
            # cache means resuming re-does none of it.
            report.interrupted = True

        report.seconds = time.time() - started
        return doc.render(replacements), report

    def _translate_all(self, doc, masker, report, replacements, on_segment) -> None:
        for seg in doc.segments:
            if self.should_stop and self.should_stop():
                raise KeyboardInterrupt
            t0 = time.time()

            # A segment inside a list or blockquote carries its container's
            # continuation prefix on every line after the first. The model
            # should see a paragraph, so the prefix comes off here and goes
            # back on after wrapping.
            indent = seg.meta.get("indent", "")
            body = rewrap.strip_continuation(seg.text, indent)

            result = masker.mask(body)
            masked, mapping = result.text, result.mapping

            # A segment with no words left after masking is pure markup that the
            # block classifier let through. There is nothing to translate.
            if not prompts.long_enough(masked):
                report.segments.append(
                    SegmentResult(
                        span=seg.span, original=seg.text, final=seg.text,
                        translated=False,
                        reason=f"under {prompts.MIN_WORDS_TO_TRANSLATE} words, "
                               f"nothing to round-trip",
                        seconds=time.time() - t0,
                    )
                )
                if on_segment:
                    on_segment(report.segments[-1])
                continue

            translated, hops, attempts, reason = self.translate_segment(masked)

            if reason:
                report.segments.append(
                    SegmentResult(
                        span=seg.span, original=seg.text, final=seg.text, hops=hops,
                        attempts=attempts, translated=False, reason=reason,
                        seconds=time.time() - t0,
                    )
                )
                if on_segment:
                    on_segment(report.segments[-1])
                continue

            cleaned, applied = self.persona.enforce(translated)

            # After the persona, so it also tidies the spacing that replacing a
            # forbidden character can leave behind.
            cleaned, artifacts = cleanup.strip_pivot_artifacts(cleaned)
            applied += artifacts

            # The container's marker lives outside the span, so a marker in the
            # output is one the model invented. Left in, it splices under the
            # real one and yields "- - A short bullet".
            cleaned, echoed = cleanup.strip_echoed_markers(masked, cleaned)
            applied += echoed

            # Restore the author's own forms. Both only ever substitute words
            # the source itself contains, so neither needs a dictionary and
            # neither can invent a spelling the author never used.
            cleaned, respelled = terms.restore_spelling(masked, cleaned)
            applied += [f"spelling {c}" for c in respelled]

            if seg.meta.get("block") == "heading_open" or seg.meta.get("heading"):
                cleaned, recased = terms.restore_capitalisation(masked, cleaned)
                applied += [f"capitalisation {c}" for c in recased]

            # Both of these run while placeholders are still in place. After
            # restoration the text contains fragments with spaces inside them
            # that must not be split or trimmed.
            # Measured against the dedented body, so the detected width is the
            # prose column rather than the column plus the bullet's indent.
            cleaned = rewrap.tighten_placeholder_spacing(body, cleaned, mapping)
            if self.preserve_wrapping:
                cleaned = rewrap.match_source_wrapping(
                    body, cleaned, mapping, result.groups
                )
            cleaned = rewrap.apply_continuation(cleaned, indent)

            final = masker.unmask(cleaned, mapping)
            replacements[seg.span] = final

            # Compared while masked, so identical placeholders on both sides
            # contribute nothing and only the prose is judged.
            issues = [
                {"kind": i.kind, "detail": i.detail, "severity": i.severity}
                for i in fidelity.compare(masked, cleaned)
            ]

            report.segments.append(
                SegmentResult(
                    span=seg.span, original=seg.text, final=final, hops=hops,
                    attempts=attempts, translated=True, rules_applied=applied,
                    issues=issues, seconds=time.time() - t0,
                )
            )
            if on_segment:
                on_segment(report.segments[-1])


def _has_words(text: str) -> bool:
    """True if anything remains that a translator could act on."""
    from .masking import SENTINEL_RE

    stripped = SENTINEL_RE.sub("", text)
    return any(c.isalpha() for c in stripped)
