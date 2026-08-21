"""Project settings from ouroboros.toml.

A paper's settings belong with the paper. Which pivot language, which persona,
which glossary, whether to preserve wrapping: these are properties of a
document, and retyping them on every invocation is how they end up
inconsistent between runs.

Precedence is the usual one and is worth stating because it is the only thing
here that can surprise: an explicit command line flag beats the file, and the
file beats the built-in default. A flag left at its default does not silently
override the config, which is the mistake this would otherwise make.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULT_NAME = "ouroboros.toml"

#: Settings a config file may set, and nothing else. An unknown key is a typo,
#: and reporting it beats silently ignoring a setting the user believed was on.
KNOWN_KEYS = {
    "pivot", "persona", "glossary", "effort", "attempts", "backend", "model",
    "preserve_wrapping", "cache", "report", "follow_inputs", "copy_assets",
    "out", "n_ctx",
}


class ConfigError(RuntimeError):
    pass


def find(start: Path | str = ".") -> Path | None:
    """Look for ouroboros.toml here, then in each parent directory.

    Walking upwards means a config beside the paper is found when the command
    is run from a subdirectory of it, which is where people actually run things.
    """
    here = Path(start).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / DEFAULT_NAME
        if candidate.is_file():
            return candidate
    return None


def load(path: Path | str | None = None) -> dict[str, Any]:
    """Read a config file, or return {} when there is none."""
    path = Path(path) if path else find()
    if path is None:
        return {}

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e

    # A [ouroboros] table is accepted so the file can live alongside other
    # tools' settings without colliding with them.
    settings = data.get("ouroboros", data)
    if not isinstance(settings, dict):
        raise ConfigError(f"{path}: expected a table of settings")

    unknown = sorted(set(settings) - KNOWN_KEYS)
    if unknown:
        raise ConfigError(
            f"{path}: unknown setting(s) {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(KNOWN_KEYS))}"
        )

    settings["_path"] = str(path)
    return settings


def apply(settings: dict[str, Any], key: str, value, default):
    """Resolve one setting: an explicit flag wins, then the file, then default.

    ``value`` is what the command line produced. It counts as explicit only when
    it differs from the flag's own default, which is what stops a flag nobody
    typed from overriding the config file.
    """
    if value != default:
        return value
    return settings.get(key, default)
