"""Project settings from ouroboros.toml.

The only behaviour here that can surprise is precedence, so that is what most of
these pin down: a flag left at its default must not override the config file,
or every setting in the file would be dead on arrival.
"""
from __future__ import annotations

import pytest

from ouroboros.config import ConfigError, apply, find, load


def _write(directory, body: str):
    path = directory / "ouroboros.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_settings_are_read_from_a_bare_table(tmp_path):
    _write(tmp_path, 'pivot = "de"\nattempts = 3\n')
    settings = load(tmp_path / "ouroboros.toml")
    assert settings["pivot"] == "de"
    assert settings["attempts"] == 3


def test_settings_may_live_under_an_ouroboros_table(tmp_path):
    """So the file can sit alongside another tool's settings."""
    _write(tmp_path, '[ouroboros]\npivot = "es"\n')
    assert load(tmp_path / "ouroboros.toml")["pivot"] == "es"


def test_having_no_config_at_all_is_not_an_error(tmp_path, monkeypatch):
    """A fresh clone has no ouroboros.toml and must still run."""
    monkeypatch.chdir(tmp_path)
    assert find(tmp_path) is None
    assert load() == {}


def test_a_config_named_explicitly_but_missing_is_an_error(tmp_path):
    """Asking for a specific file and silently getting the defaults instead is
    the same trap as a missing persona: the user believed a setting was on."""
    with pytest.raises(ConfigError):
        load(tmp_path / "nope.toml")


def test_the_file_is_found_from_a_subdirectory(tmp_path):
    """People run commands from inside the paper, not above it."""
    _write(tmp_path, 'pivot = "de"\n')
    nested = tmp_path / "sections" / "deep"
    nested.mkdir(parents=True)
    assert find(nested) == tmp_path / "ouroboros.toml"


def test_an_unknown_key_is_reported(tmp_path):
    """Silently ignoring a setting the user believed was on is worse than
    refusing to start."""
    path = _write(tmp_path, 'pivott = "de"\n')
    with pytest.raises(ConfigError, match="pivott"):
        load(path)


def test_malformed_toml_is_reported(tmp_path):
    path = _write(tmp_path, "pivot = [unclosed\n")
    with pytest.raises(ConfigError):
        load(path)


# ----------------------------------------------------------------- precedence


def test_an_explicit_flag_beats_the_config():
    assert apply({"pivot": "de"}, "pivot", "es", "fr") == "es"


def test_the_config_beats_the_built_in_default():
    """The important one: a flag nobody typed still sits at its default, and
    must not be mistaken for an explicit choice."""
    assert apply({"pivot": "de"}, "pivot", "fr", "fr") == "de"


def test_the_default_survives_when_nothing_else_is_set():
    assert apply({}, "pivot", "fr", "fr") == "fr"


@pytest.mark.parametrize("configured,flag,default,expected", [
    (False, True, True, False),     # config turns a default-on flag off
    (True, False, True, False),     # explicit --no-x still wins
    (False, False, True, False),
])
def test_precedence_holds_for_boolean_flags(configured, flag, default, expected):
    assert apply({"cache": configured}, "cache", flag, default) == expected
