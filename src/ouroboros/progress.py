"""Telling the user how long this is going to take.

A paper is an hours-long job at roughly fifty seconds a paragraph, which makes
two questions worth answering before anything starts: how much work is there,
and how much of it is already done. The second matters more than it sounds,
because the cache makes a re-run of an unchanged paragraph free, so the honest
estimate after a crash or an edit is usually far shorter than the first one.

The estimate is measured rather than assumed. A starting guess gets the first
line on screen, and every completed segment replaces a little more of that guess
with the machine's actual pace.
"""
from __future__ import annotations

#: Used only until real timings arrive, then replaced by the machine's actual
#: pace. Measured across this project's runs on an M4 Max, where whole files
#: averaged 12 to 29 seconds a segment: a paragraph costs two calls at roughly
#: 25s each, but headings and short bullets are far cheaper and pull the mean
#: down. An initial guess of 50 consistently overstated the wait by about half.
DEFAULT_SECONDS_PER_SEGMENT = 25.0

#: A cached segment costs a file read. It should never drag the average for
#: uncached work downwards, so the two are tracked apart.
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


class Estimator:
    """Predicts remaining time from the pace observed so far."""

    def __init__(self, uncached: int, cached: int = 0,
                 default_seconds: float = DEFAULT_SECONDS_PER_SEGMENT) -> None:
        self.remaining_uncached = uncached
        self.remaining_cached = cached
        self.default_seconds = default_seconds
        self._elapsed = 0.0
        self._done_uncached = 0

    @property
    def seconds_per_segment(self) -> float:
        if self._done_uncached == 0:
            return self.default_seconds
        return self._elapsed / self._done_uncached

    @property
    def measured(self) -> bool:
        return self._done_uncached > 0

    def record(self, seconds: float) -> None:
        """Note one finished segment, cached or not."""
        if seconds < _CACHED_THRESHOLD:
            self.remaining_cached = max(0, self.remaining_cached - 1)
            return
        self._elapsed += seconds
        self._done_uncached += 1
        self.remaining_uncached = max(0, self.remaining_uncached - 1)

    @property
    def remaining_seconds(self) -> float:
        return self.remaining_uncached * self.seconds_per_segment

    def describe(self) -> str:
        if self.remaining_uncached == 0:
            return "all cached, instant"
        eta = human_duration(self.remaining_seconds)
        pace = f"{self.seconds_per_segment:.0f}s/segment"
        return f"~{eta} ({pace}{'' if self.measured else ', estimated'})"


def summarise(total: int, cached: int, files: int) -> str:
    """The one line printed before a long run starts."""
    uncached = total - cached
    est = Estimator(uncached, cached)
    where = f" across {files} files" if files > 1 else ""

    if total == 0:
        return "nothing to translate"
    if uncached == 0:
        return f"{total} segments{where}, all cached, this will be instant"
    if cached:
        return (f"{total} segments{where}, {cached} already cached, "
                f"{uncached} to translate, {est.describe()}")
    return f"{total} segments{where}, {est.describe()}"
