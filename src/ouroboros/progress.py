"""Telling the user how long this is going to take.

A paper is an hours-long job, which makes two questions worth answering before
anything starts: how much work is there, and how much of it is already done. The
second matters more than it sounds, because the cache makes a re-run of an
unchanged paragraph free, so the honest estimate after a crash or an edit is
usually far shorter than the first one.

Cost is modelled per character rather than per segment, because a segment is not
a unit of work. Fitted against this project's own timings:

    seconds = 18.6 + 0.0857 * characters

The intercept is real: a six-character heading still costs about thirteen
seconds, because two calls have to be set up and a reasoning block generated
whatever the length. The slope is the part that actually varies, and ignoring it
is what made an earlier version predict 68 minutes for a run that took four
hours. That document's paragraphs were three times longer than the fixtures the
flat rate had been calibrated on.

Working in characters also fixes the running estimate. Averaging seconds per
segment goes badly wrong on a document that opens with a run of short headings:
the pace looks fast, and then the long body paragraphs arrive.
"""
from __future__ import annotations

#: Fixed cost of translating one segment, whatever its length: two round-trip
#: calls, each with prompt setup and a reasoning block.
DEFAULT_OVERHEAD_SECONDS = 18.6

#: Marginal cost per character of prose, both legs combined.
DEFAULT_SECONDS_PER_CHAR = 0.0857

#: A cached segment costs a file read. It must never drag the measured pace for
#: real work downwards, so the two are counted apart.
_CACHED_THRESHOLD = 1.0


def human_duration(seconds: float) -> str:
    """A duration a person can act on, not a number of seconds."""
    seconds = max(0.0, seconds)
    if seconds < 1:
        return "instant"
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = int(minutes // 60)
    rest = int(minutes % 60)
    return f"{hours}h {rest:02d}m" if rest else f"{hours}h"


def predict(chars: int, segments: int = 1) -> float:
    """Predicted seconds for ``segments`` totalling ``chars`` characters."""
    return segments * DEFAULT_OVERHEAD_SECONDS + chars * DEFAULT_SECONDS_PER_CHAR


class Estimator:
    """Predicts remaining time, refining the model from observed pace."""

    def __init__(self, uncached: int, cached: int = 0, uncached_chars: int = 0) -> None:
        self.remaining_uncached = uncached
        self.remaining_cached = cached
        self.remaining_chars = uncached_chars
        self._elapsed = 0.0
        self._done_chars = 0
        self._done_segments = 0

    @property
    def measured(self) -> bool:
        return self._done_segments > 0

    @property
    def seconds_per_segment(self) -> float:
        """Observed average, for display. Not what the prediction is built on."""
        if not self._done_segments:
            return DEFAULT_OVERHEAD_SECONDS
        return self._elapsed / self._done_segments

    def record(self, seconds: float, chars: int = 0) -> None:
        """Note one finished segment, cached or not."""
        if seconds < _CACHED_THRESHOLD:
            self.remaining_cached = max(0, self.remaining_cached - 1)
            self.remaining_chars = max(0, self.remaining_chars - chars)
            return
        self._elapsed += seconds
        self._done_chars += chars
        self._done_segments += 1
        self.remaining_uncached = max(0, self.remaining_uncached - 1)
        self.remaining_chars = max(0, self.remaining_chars - chars)

    @property
    def remaining_seconds(self) -> float:
        if not self.remaining_uncached:
            return 0.0
        if not self.measured:
            return predict(self.remaining_chars, self.remaining_uncached)

        # Rescale the fitted model by how this machine is actually performing.
        # A ratio keeps the shape of the model, which matters when the segments
        # left are a different size from the ones already done.
        expected = predict(self._done_chars, self._done_segments)
        factor = self._elapsed / expected if expected > 0 else 1.0
        return factor * predict(self.remaining_chars, self.remaining_uncached)

    def describe(self) -> str:
        if not self.remaining_uncached:
            return "all cached, instant"
        eta = human_duration(self.remaining_seconds)
        pace = f"{self.seconds_per_segment:.0f}s/segment"
        return f"~{eta} ({pace}{'' if self.measured else ', estimated'})"


def summarise(total: int, cached: int, files: int, chars: int = 0) -> str:
    """The one line printed before a long run starts."""
    uncached = total - cached
    where = f" across {files} files" if files > 1 else ""

    if total == 0:
        return "nothing to translate"
    if uncached == 0:
        return f"{total} segments{where}, all cached, this will be instant"

    eta = human_duration(predict(chars, uncached))
    if cached:
        return (f"{total} segments{where}, {cached} already cached, "
                f"{uncached} to translate, ~{eta}")
    return f"{total} segments{where}, ~{eta}"
