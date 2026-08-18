"""Discovery of local model weights.

``models/`` is the source of truth. The rule the CLI promises is simple: if the
directory holds exactly one model, use it without being asked; if it holds
several, require the user to choose rather than guessing, because silently
picking a different quantisation would change output quality with no visible
cause.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path("models")

_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)$")

#: The quantisation to fetch when the user does not name one. Q5_K_M keeps
#: noticeably more translation quality than Q4 at 19 GB, which is affordable on
#: any machine that can run a 30B model at all.
DEFAULT_REPO = "unsloth/Muse-Glimmer-30B-GGUF"
DEFAULT_FILE = "Muse-Glimmer-30B-UD-Q5_K_M.gguf"


@dataclass(frozen=True)
class LocalModel:
    path: Path
    size_bytes: int

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


class NoModelsFound(RuntimeError):
    pass


class AmbiguousModel(RuntimeError):
    pass


def discover(models_dir: Path | str = DEFAULT_DIR) -> list[LocalModel]:
    """List usable GGUF files, newest-looking first.

    Multi-part GGUFs are represented by their first shard only, since that is
    the file llama.cpp is given and it loads the rest itself.
    """
    directory = Path(models_dir)
    if not directory.is_dir():
        return []

    found: list[LocalModel] = []
    for path in sorted(directory.rglob("*.gguf")):
        # Skip huggingface_hub's staging area and vision projectors, which are
        # not standalone models.
        if ".cache" in path.parts or path.name.startswith("mmproj"):
            continue
        # For sharded models keep only the first shard; llama.cpp finds the rest.
        shard = _SHARD_RE.search(path.stem)
        if shard and int(shard.group(1)) != 1:
            continue
        found.append(LocalModel(path=path, size_bytes=path.stat().st_size))
    return found


def resolve(name: str | None = None, models_dir: Path | str = DEFAULT_DIR) -> LocalModel:
    """Pick the model to use, applying the auto-default rule."""
    models = discover(models_dir)

    if not models:
        raise NoModelsFound(
            f"no .gguf files in {models_dir}/. Run 'ouroboros download' to fetch "
            f"{DEFAULT_FILE} ({DEFAULT_REPO})."
        )

    if name:
        # Accept a full path, a filename, or any unambiguous substring.
        exact = [m for m in models if name in (str(m.path), m.path.name, m.name)]
        if exact:
            return exact[0]
        partial = [m for m in models if name.lower() in m.name.lower()]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise NoModelsFound(
                f"no model matching {name!r} in {models_dir}/. "
                f"Available: {', '.join(m.name for m in models)}"
            )
        raise AmbiguousModel(
            f"{name!r} matches several models: {', '.join(m.name for m in partial)}"
        )

    if len(models) == 1:
        return models[0]

    raise AmbiguousModel(
        f"{len(models)} models in {models_dir}/, so --model is required. "
        f"Available: {', '.join(m.name for m in models)}"
    )


def download(
    repo: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILE,
    models_dir: Path | str = DEFAULT_DIR,
) -> Path:
    """Fetch a GGUF into ``models/``. Idempotent: an existing file is reused."""
    from huggingface_hub import hf_hub_download

    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)

    target = directory / filename
    if target.exists():
        return target

    out = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(directory),
    )
    return Path(out)
