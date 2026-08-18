"""In-process GGUF backend via llama-cpp-python.

This is the default. It reads the weights straight out of ``models/``, which
makes the directory the single source of truth for which models exist and
removes any daemon from the picture.

Requires llama-cpp-python >= 0.3.35. That is the first release whose vendored
llama.cpp registers the ``muse-glimmer`` architecture; 0.3.34 fails to load the
file at all, so the floor is pinned in pyproject.toml.
"""
from __future__ import annotations

from pathlib import Path

from . import atem
from .base import register


@register("llamacpp")
class LlamaCppBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        effort: str = "low",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        from llama_cpp import Llama

        self.model_path = str(model_path)
        self.effort = effort
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            **kwargs,
        )

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        # llama.cpp prepends BOS during tokenisation, so the rendered prompt
        # must not carry its own or the model sees it twice, which llama-cpp
        # warns about and which measurably degrades output.
        prompt = atem.render_prompt(system, user, effort=self.effort, include_bos=False)
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        result = self._llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=atem.STOP,
            **kwargs,
        )
        return atem.parse_response(result["choices"][0]["text"])

    def __repr__(self) -> str:
        return f"LlamaCppBackend({Path(self.model_path).name}, effort={self.effort})"
