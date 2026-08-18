"""Ollama backend.

The escape hatch. Ollama ships muse-glimmer in its own library, including an MLX
build tuned for Apple Silicon, so this is the fallback if a llama.cpp regression
ever breaks the in-process path, and the fast path on machines where MLX beats
Metal-via-llama.cpp.

Ollama keeps weights in its own store rather than in ``models/``. Pointing
OLLAMA_MODELS at the project directory keeps this project's promise that all
weights live in one gitignored place, at the cost of not sharing a cache with
any ollama models pulled outside the project.
"""
from __future__ import annotations

import os

from .base import register

DEFAULT_MODEL = "muse-glimmer:30b"


@register("ollama")
class OllamaBackend:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        models_dir: str | None = None,
        effort: str = "low",
        **kwargs,
    ) -> None:
        import ollama

        if models_dir:
            os.environ["OLLAMA_MODELS"] = str(models_dir)
        self.model = model
        self.effort = effort
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        # Ollama applies the model's own chat template, which reads the reasoning
        # directive out of the system text. Appending it here gives the same
        # effort control as the in-process path.
        system = f"{system}\n\nReasoning strength: {self.effort}."
        options = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed

        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=options,
        )
        return (response["message"]["content"] or "").strip()

    def __repr__(self) -> str:
        return f"OllamaBackend({self.model}, effort={self.effort})"
