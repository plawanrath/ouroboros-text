"""Persona profiles: restoring an author's style after the pivot flattens it.

A round trip through another language is a style solvent. French has its own
preferences about sentence length, connectives, and formality, and English
coming back out of it arrives normalised: contractions formalise, clauses
lengthen, and "Furthermore," appears where nothing was before. Semantics survive
the trip. Voice does not.

So voice is reconstructed explicitly on the return leg, by two mechanisms, and
the split between them is deliberate:

  Soft rules go in the prompt. Rhythm, sentence construction, when to reach for
  a colon: these need judgement, so the model does them.

  Hard rules are enforced in code. "Never use an em dash" is a predicate over
  the output string, and a regex satisfies it every time where a prompt merely
  usually does. Anything mechanically checkable belongs here rather than in the
  system prompt, where a 30B model is free to forget it.

Personas live in a gitignored directory because a persona describes a specific
person's writing, and this repository is public. The translation mechanics that
apply to everyone stay in prompts.py, which is checked in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DIR = Path("personas")
DEFAULT_NAME = "default"

#: A persona file is a system prompt fragment, so it is treated as untrusted
#: input even though it usually is not. Bounded to keep a runaway file from
#: crowding the context window and to make review of an unfamiliar one feasible.
MAX_BYTES = 64 * 1024

#: Persona names index a filename, so they are restricted to a safe slug rather
#: than sanitised after the fact. "../../etc/passwd" is not a name.
_SAFE_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

#: Model control tokens. Because the ATEM prompt is rendered by hand, text
#: containing these could close the system block and forge a conversation turn.
#: Stripped from every untrusted string that reaches a prompt, persona files and
#: document text alike.
CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{0,64}\|>")

_FRONTMATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)

NONE_NAMES = {"none", "neutral", "off", ""}


class PersonaError(RuntimeError):
    pass


def strip_control_tokens(text: str) -> str:
    """Remove model control tokens from text destined for a prompt."""
    return CONTROL_TOKEN_RE.sub("", text)


@dataclass
class Persona:
    """A named writing style: prompt guidance plus mechanically enforced rules."""

    name: str
    guidance: str
    description: str = ""
    #: Characters that must never appear, mapped to their replacement.
    forbid: dict[str, str] = field(default_factory=dict)
    #: Sentence openers stripped when they begin a sentence.
    banned_openers: list[str] = field(default_factory=list)
    path: Path | None = None

    @property
    def is_none(self) -> bool:
        return not self.guidance and not self.forbid and not self.banned_openers

    def enforce(self, text: str) -> tuple[str, list[str]]:
        """Apply the hard rules. Returns ``(cleaned_text, rules_applied)``."""
        applied: list[str] = []

        for bad, good in self.forbid.items():
            if bad and bad in text:
                text = text.replace(bad, good)
                applied.append(f"replaced {bad!r}")

        for opener in self.banned_openers:
            if not opener:
                continue
            pattern = re.compile(rf"(\A|(?<=[.!?])\s+){re.escape(opener)}\s*", re.IGNORECASE)
            new, n = pattern.subn(lambda m: m.group(1), text)
            if n:
                text = re.sub(
                    r"(\A|(?<=[.!?])\s+)([a-z])",
                    lambda m: m.group(1) + m.group(2).upper(),
                    new,
                )
                applied.append(f"dropped opener {opener!r}")

        # Tidy whitespace damage from the removals above. Newlines are left
        # alone: they are structural in both Markdown and LaTeX.
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        text = re.sub(r"[^\S\n]+([.,;:!?])", r"\1", text)
        return text, applied


# --------------------------------------------------------------- frontmatter


def _parse_scalar(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def _parse_frontmatter(meta: str) -> dict:
    """A deliberately small YAML subset: scalars, nested scalars, and lists.

    Personas need six keys. Depending on PyYAML for that would put an arbitrary
    deserialiser in the path of an untrusted file for no benefit.
    """
    out: dict = {}
    key: str | None = None

    for line in meta.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indented = line[0] in " \t"
        stripped = line.strip()

        if stripped.startswith("- ") and key:
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            continue
        k, _, v = stripped.partition(":")
        k, v = k.strip(), v.strip()

        # Trailing comments, but only outside a quoted value.
        if v and v[0] not in "\"'" and "#" in v:
            v = v.split("#", 1)[0].strip()

        if indented and key:
            out.setdefault(key, {})
            if isinstance(out[key], dict):
                out[key][_parse_scalar(k)] = _parse_scalar(v)
            continue

        key = k
        out[k] = _parse_scalar(v) if v else None

    return out


# -------------------------------------------------------------------- loading


def none_persona() -> Persona:
    return Persona(name="none", guidance="", description="No style shaping.")


def resolve_path(name: str, personas_dir: Path | str = DEFAULT_DIR) -> Path:
    """Map a persona name to a file, refusing anything that escapes the dir."""
    if not _SAFE_NAME.match(name):
        raise PersonaError(
            f"invalid persona name {name!r}: use letters, digits, dot, dash, "
            f"or underscore only"
        )

    directory = Path(personas_dir).resolve()
    path = (directory / f"{name}.md").resolve()

    # Belt and braces. _SAFE_NAME already forbids separators, but a symlink
    # inside the directory could still point outside it.
    if directory not in path.parents:
        raise PersonaError(f"persona {name!r} resolves outside {personas_dir}/")
    return path


def from_file(path: Path) -> Persona:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise PersonaError(
            f"persona {path.name} is {size} bytes, over the {MAX_BYTES} limit"
        )

    raw = strip_control_tokens(path.read_text(encoding="utf-8", errors="replace"))
    m = _FRONTMATTER.match(raw)
    if not m:
        return Persona(name=path.stem, guidance=raw.strip(), path=path)

    meta = _parse_frontmatter(m.group("meta"))
    forbid = meta.get("forbid")
    openers = meta.get("banned_openers")
    return Persona(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        guidance=m.group("body").strip(),
        forbid=forbid if isinstance(forbid, dict) else {},
        banned_openers=openers if isinstance(openers, list) else [],
        path=path,
    )


def load(
    name: str | None = None,
    personas_dir: Path | str = DEFAULT_DIR,
    *,
    required: bool | None = None,
) -> Persona:
    """Load a persona by name.

    A missing *default* persona is not an error. A fresh clone of a public repo
    has no personas at all, and the tool must still run, so the default degrades
    to no style shaping. A missing *explicitly requested* persona is an error,
    because the user asked for something specific and silently substituting
    something else would be worse than stopping.
    """
    explicit = name is not None
    name = (name or DEFAULT_NAME).strip()

    if name.lower() in NONE_NAMES:
        return none_persona()

    if required is None:
        required = explicit

    path = resolve_path(name, personas_dir)
    if not path.exists():
        if required:
            raise PersonaError(
                f"no persona {name!r} in {personas_dir}/. "
                f"Available: {', '.join(available(personas_dir)) or '(none)'}"
            )
        return none_persona()

    return from_file(path)


def available(personas_dir: Path | str = DEFAULT_DIR) -> list[str]:
    directory = Path(personas_dir)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.md") if p.stem != "README")
