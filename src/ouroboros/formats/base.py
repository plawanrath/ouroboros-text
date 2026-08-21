"""Format registry.

A format module's only job is to answer one question about a source string:
*which character ranges are prose?* It never rewrites anything. Adding support
for a new input type means adding one module and registering it here; nothing
downstream changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..document import Document


class Format(Protocol):
    name: str
    extensions: tuple[str, ...]

    def parse(self, source: str, path: str | None = None, **kwargs) -> Document:
        """Report which character ranges of ``source`` hold prose.

        Extra keyword arguments are format-specific and optional. LaTeX accepts
        ``fragment=True`` for a file that is included by another; a format that
        has no such notion simply ignores what it does not recognise.
        """
        ...


_REGISTRY: dict[str, Format] = {}
_BY_EXT: dict[str, Format] = {}


def register(fmt: Format) -> Format:
    _REGISTRY[fmt.name] = fmt
    for ext in fmt.extensions:
        _BY_EXT[ext] = fmt
    return fmt


def get(name: str) -> Format:
    if name not in _REGISTRY:
        raise KeyError(f"unknown format {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def for_path(path: str | Path) -> Format:
    ext = Path(path).suffix.lower()
    if ext not in _BY_EXT:
        raise KeyError(f"no format handles {ext!r}; known: {sorted(_BY_EXT)}")
    return _BY_EXT[ext]


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_BY_EXT))


def load_builtins() -> None:
    from . import latex, markdown  # noqa: F401  (registration side effect)
