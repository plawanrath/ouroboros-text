"""Surviving a run that takes hours.

Three things matter once a job stops fitting in a coffee break: knowing what it
will cost before starting, being able to stop it without losing the work, and
being able to check a sample rather than betting the afternoon on a guess.
"""
from __future__ import annotations

import pytest

from ouroboros.backends.mock import MockBackend
from ouroboros.formats import base as formats
from ouroboros.persona import none_persona
from ouroboros.pipeline import Cache, RoundTrip
from ouroboros.progress import Estimator, human_duration, summarise

formats.load_builtins()

SOURCE = "\n\n".join(f"Paragraph number {i} with enough prose to be real." for i in range(10))


def _doc(source: str = SOURCE):
    return formats.get("markdown").parse(source, path="long.md")


def _trip(backend=None, cache=None, **kw):
    return RoundTrip(backend or MockBackend(), pivot="fr",
                     persona=none_persona(), cache=cache, **kw)


# ------------------------------------------------------------------ interrupt


class _StopsAfter:
    """A backend that raises KeyboardInterrupt part way, as Ctrl-C would."""

    name = "stops"

    def __init__(self, after: int):
        self.after, self.calls = after, 0

    def generate(self, system, user, **kw):
        self.calls += 1
        if self.calls > self.after:
            raise KeyboardInterrupt
        return user.upper()


def test_an_interrupt_keeps_the_work_already_done():
    doc = _doc()
    # Two calls per segment, so this stops midway through the third paragraph.
    output, report = _trip(_StopsAfter(5)).run(doc)

    assert report.interrupted
    assert report.translated == 2
    assert "PARAGRAPH NUMBER 0" in output.upper()


def test_an_interrupt_leaves_the_rest_in_the_original_language():
    """Partial output is still a valid document, not a truncated one."""
    doc = _doc()
    output, _ = _trip(_StopsAfter(5)).run(doc)

    assert "Paragraph number 9 with enough prose to be real." in output
    # Every paragraph is still present; none were dropped.
    for i in range(10):
        assert f"number {i}" in output.lower() or f"NUMBER {i}" in output.upper()


def test_an_interrupt_is_recorded_in_the_report():
    import json

    _, report = _trip(_StopsAfter(3)).run(_doc())
    assert json.loads(report.to_json())["interrupted"] is True


def test_a_completed_run_is_not_marked_interrupted():
    _, report = _trip().run(_doc())
    assert report.interrupted is False


def test_interrupted_work_is_cached_so_resuming_is_free(tmp_path):
    """The resume story: the cache, not a checkpoint file."""
    cache = Cache(tmp_path / "cache")
    doc = _doc()

    _, first = _trip(_StopsAfter(5), cache=cache, model_id="m").run(doc)
    assert first.interrupted

    done, total, chars = _trip(MockBackend(), cache=cache, model_id="m").pending(doc)
    assert done == first.translated
    assert total == 10
    assert chars > 0, "work left should carry a character count for the estimate"


# ------------------------------------------------------------------- pending


def test_pending_counts_nothing_as_cached_on_a_cold_cache(tmp_path):
    done, total, chars = _trip(cache=Cache(tmp_path / "cache"), model_id="m").pending(_doc())
    assert (done, total) == (0, 10)
    assert chars > 0


def test_pending_counts_everything_after_a_full_run(tmp_path):
    cache = Cache(tmp_path / "cache")
    doc = _doc()
    trip = _trip(cache=cache, model_id="m")
    trip.run(doc)

    done, total, chars = trip.pending(doc)
    assert (done, total) == (10, 10)
    assert chars == 0, "nothing is left, so no characters remain to cost"


def test_pending_is_sensitive_to_the_persona(tmp_path):
    """A changed persona invalidates the cache, and the estimate must agree."""
    from ouroboros.persona import Persona

    cache = Cache(tmp_path / "cache")
    doc = _doc()
    RoundTrip(MockBackend(), pivot="fr", persona=none_persona(),
              cache=cache, model_id="m").run(doc)

    other = Persona(name="other", guidance="Write tersely.")
    trip = RoundTrip(MockBackend(), pivot="fr", persona=other,
                     cache=cache, model_id="m")
    assert trip.pending(doc)[0] == 0


# ------------------------------------------------------------------ sampling


def test_limit_translates_only_the_first_paragraphs():
    doc = _doc()
    from ouroboros.cli import _select

    narrowed = _select(doc, limit=3, sample=0)
    assert len(narrowed.segments) == 3
    assert narrowed.segments[0].text.startswith("Paragraph number 0")


def test_sample_spreads_across_the_document():
    """The first N paragraphs of a paper are all introduction.

    A spot check that only ever sees the opening tells you nothing about how the
    method section will fare.
    """
    from ouroboros.cli import _select

    narrowed = _select(_doc(), limit=0, sample=3)
    assert len(narrowed.segments) == 3
    numbers = [int(s.text.split()[2]) for s in narrowed.segments]
    assert numbers[0] == 0
    assert numbers[-1] >= 6, f"sample clustered at the start: {numbers}"


def test_a_subset_still_renders_the_whole_document():
    """Unselected paragraphs are passed through, not dropped."""
    from ouroboros.cli import _select

    doc = _doc()
    narrowed = _select(doc, limit=3, sample=0)
    assert narrowed.unchanged() == doc.source

    output, _ = _trip().run(narrowed)
    assert "Paragraph number 9 with enough prose to be real." in output


def test_selection_is_a_no_op_when_it_would_not_narrow():
    from ouroboros.cli import _select

    doc = _doc()
    assert len(_select(doc, limit=0, sample=0).segments) == 10
    assert len(_select(doc, limit=99, sample=0).segments) == 10


# -------------------------------------------------------------- the estimate


@pytest.mark.parametrize("seconds,expected", [
    (0.2, "instant"), (45, "45s"), (600, "10 min"), (7200, "2h"), (5400, "1h 30m"),
])
def test_durations_read_like_something_a_person_would_say(seconds, expected):
    assert human_duration(seconds) == expected


def test_the_estimate_starts_from_a_guess_and_then_measures():
    est = Estimator(uncached=10, uncached_chars=3000)
    assert not est.measured
    first = est.remaining_seconds

    for _ in range(3):
        est.record(10.0, 300)

    assert est.measured
    assert est.seconds_per_segment == pytest.approx(10.0)
    # Faster than the fitted model predicted, so the estimate comes down.
    assert est.remaining_seconds < first


def test_cached_segments_do_not_drag_the_pace_estimate():
    """A cached segment costs a file read and would otherwise imply the whole
    run is nearly instant."""
    est = Estimator(uncached=5, cached=5, uncached_chars=1500)
    est.record(0.01, 300)     # cached
    est.record(20.0, 300)     # real work

    assert est.seconds_per_segment == pytest.approx(20.0)
    assert est.remaining_uncached == 4


def test_the_summary_says_what_is_already_done():
    assert "all cached" in summarise(total=10, cached=10, files=1)
    assert "nothing to translate" == summarise(total=0, cached=0, files=1)

    line = summarise(total=10, cached=6, files=3, chars=2000)
    assert "6 already cached" in line
    assert "4 to translate" in line
    assert "across 3 files" in line


def test_the_estimate_scales_with_segment_length():
    """Regression: a flat per-segment rate predicted 68 minutes for a run that
    took four hours, because that paper's paragraphs were three times longer
    than the fixtures the rate was calibrated on."""
    short = Estimator(uncached=10, uncached_chars=1000)
    long_ = Estimator(uncached=10, uncached_chars=10000)
    assert long_.remaining_seconds > 2 * short.remaining_seconds


def test_a_run_of_short_segments_does_not_make_long_ones_look_fast():
    """Averaging seconds per segment goes wrong on a document that opens with
    headings: the pace looks quick, then the body paragraphs arrive."""
    est = Estimator(uncached=10, uncached_chars=100 * 5 + 9 * 1000)
    est.record(20.0, 100)          # one short heading, quickly done

    # Nine long paragraphs remain. The estimate must reflect their size, not
    # the pace of the heading.
    assert est.remaining_seconds > 9 * 20.0


# ------------------------------------------- segments too short to round-trip


def test_a_heading_is_left_alone():
    """Measured on a real paper: of 46 changed segments under seven words, not
    one came back better than it went out.

    A heading is a noun phrase with no sentence around it. Through another
    language it returns as a synonym ("Diagnosis." -> "Diagnostic"), and even
    when the words survive the shape does not: a run-in title that loses its
    trailing period stops rendering correctly.
    """
    doc = formats.get("markdown").parse("# Experimental Setup\n", path="h.md")
    backend = MockBackend()
    output, report = _trip(backend).run(doc)

    assert output == doc.source
    assert not backend.calls, "a heading was sent to the model"
    assert report.fallbacks and "round-trip" in report.fallbacks[0].reason


def test_real_prose_is_still_translated():
    """The threshold must not swallow ordinary sentences."""
    doc = _doc()
    output, report = _trip().run(doc)
    assert report.translated == 10
    assert output != doc.source


def test_the_threshold_counts_words_not_placeholders():
    """A citation-heavy line is not prose just because it is long."""
    from ouroboros.prompts import long_enough

    assert not long_enough("[[0]] [[1]] [[2]] [[3]] [[4]] [[5]] [[6]] [[7]]")
    assert long_enough("this sentence has more than seven ordinary words in it")
