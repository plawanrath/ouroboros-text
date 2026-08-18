"""Backend protocol and registry.

A backend turns (system prompt, user text) into a string. That is the entire
contract, and keeping it that narrow is what lets the test suite run the full
pipeline against a mock in milliseconds without a 19 GB file on disk.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    name: str

    def generate(self, system: str, user: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        ...


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def load_builtins() -> None:
    from . import llamacpp, mock, ollama  # noqa: F401  (registration side effect)
