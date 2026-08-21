"""Multi-file LaTeX projects.

A real paper is rarely one file. It is a root that sets up the document and a
handful of ``\\input`` or ``\\include`` lines pulling in a section each, so
translating only the file you were handed translates almost nothing.

Traversal is deliberately narrow. We follow the include graph to find *which
files to translate*, and nothing else: the ``\\input`` line itself stays
protected in the text, because it is an instruction about document assembly
rather than something to translate.

Two things here are load-bearing.

The first is the fragment problem. An included section has no
``\\begin{document}``, and the LaTeX format module treats a file without one as
having no prose at all. That default is right when someone points the tool at a
style file by mistake, and wrong for every file reached through traversal, so
traversal marks what it finds as a fragment and the parser treats the whole file
as body.

The second is containment. ``\\input`` takes a path, and a path can climb. Every
resolved file is checked to be inside the project directory before it is opened,
so a document cannot talk the tool into reading, translating, and writing out
something from elsewhere on the disk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Macros that pull in another source file. \subfile is from the subfiles
#: package and behaves like \input for our purposes.
_INCLUDE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]*)\}")

#: A comment runs to end of line. Scanning without stripping these would follow
#: an \input that the author had deliberately commented out.
_COMMENT = re.compile(r"(?<!\\)%[^\n]*")

#: Marks a file that stands on its own rather than being included.
_DOCUMENT = re.compile(r"\\begin\s*\{document\}")

DEFAULT_MAX_FILES = 500
DEFAULT_MAX_DEPTH = 32


@dataclass(frozen=True)
class ProjectFile:
    """One source file in a project, and how it was reached."""

    path: Path
    #: Path relative to the project root directory, used to mirror the output.
    relative: Path
    depth: int
    #: True when the file has no \begin{document} and so must be parsed as a
    #: body fragment rather than skipped.
    fragment: bool
    included_by: Path | None = None


class ProjectError(RuntimeError):
    pass


def strip_comments(source: str) -> str:
    return _COMMENT.sub("", source)


def is_standalone(source: str) -> bool:
    """True if the file carries its own \\begin{document}."""
    return bool(_DOCUMENT.search(strip_comments(source)))


def _candidates(name: str) -> list[str]:
    """TeX lets you write \\input{foo} for foo.tex."""
    name = name.strip()
    if not name:
        return []
    if name.endswith(".tex"):
        return [name]
    return [f"{name}.tex", name]


def _resolve(name: str, root_dir: Path, current_dir: Path) -> Path | None:
    """Locate an included file, refusing anything outside the project.

    TeX resolves relative to the main file's directory, so that is tried first,
    with the including file's own directory as a fallback for projects that nest
    their inputs.
    """
    for directory in (root_dir, current_dir):
        for candidate in _candidates(name):
            path = (directory / candidate)
            try:
                resolved = path.resolve()
            except OSError:
                continue

            # Containment check. \input{../../../etc/passwd} must not resolve.
            if root_dir not in resolved.parents and resolved != root_dir:
                continue
            if resolved.is_file():
                return resolved
    return None


def discover(
    root: Path | str,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[list[ProjectFile], list[str]]:
    """Walk the include graph from ``root``.

    Returns the files to translate in breadth-first order and a list of warnings
    about includes that could not be resolved, which is worth surfacing: a
    missing section is the difference between translating a paper and
    translating its title page.
    """
    root = Path(root).resolve()
    if not root.is_file():
        raise ProjectError(f"{root} is not a file")

    root_dir = root.parent
    warnings: list[str] = []

    first = ProjectFile(
        path=root,
        relative=Path(root.name),
        depth=0,
        fragment=not is_standalone(root.read_text(encoding="utf-8", errors="replace")),
    )

    found: list[ProjectFile] = [first]
    seen: set[Path] = {root}
    queue: list[ProjectFile] = [first]

    while queue:
        current = queue.pop(0)
        if current.depth >= max_depth:
            warnings.append(f"stopped at depth {max_depth} in {current.relative}")
            continue

        try:
            source = current.path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            warnings.append(f"cannot read {current.relative}: {e}")
            continue

        for m in _INCLUDE.finditer(strip_comments(source)):
            name = m.group(1)
            target = _resolve(name, root_dir, current.path.parent)

            if target is None:
                warnings.append(f"unresolved \\input{{{name}}} in {current.relative}")
                continue
            if target in seen:
                continue  # already queued, or a cycle
            if len(found) >= max_files:
                warnings.append(f"stopped after {max_files} files")
                return found, warnings

            seen.add(target)
            child = ProjectFile(
                path=target,
                relative=target.relative_to(root_dir),
                depth=current.depth + 1,
                # Anything reached by traversal is a body fragment unless it
                # declares its own document environment.
                fragment=not is_standalone(
                    target.read_text(encoding="utf-8", errors="replace")
                ),
                included_by=current.path,
            )
            found.append(child)
            queue.append(child)

    return found, warnings


#: Files copied alongside the translation so the output still builds. Excludes
#: are conservative: version control, virtualenvs, and build products.
_ASSET_SKIP_DIRS = {".git", ".svn", ".hg", ".venv", "venv", "__pycache__",
                    "node_modules", "_minted", ".build"}
_ASSET_SKIP_SUFFIXES = {".aux", ".log", ".out", ".toc", ".synctex", ".fls",
                        ".fdb_latexmk", ".blg", ".bbl", ".pyc"}


#: Source formats are never assets. A .tex or .md in the project directory is
#: either already being translated, or it is not part of this document at all,
#: and copying an untranslated one in beside the translated ones is worse than
#: leaving it out: it looks like output but is not.
_ASSET_SKIP_SOURCE = {".tex", ".latex", ".md", ".markdown", ".mdown"}


def asset_files(root_dir: Path | str, translated: set[Path]) -> list[Path]:
    """Every project file that is not translated and is worth carrying over.

    A directory of translated .tex files does not compile on its own: it needs
    the figures, the .bib, and any local .sty. Copying them is what makes the
    output a buildable paper rather than a pile of prose.
    """
    root_dir = Path(root_dir).resolve()
    out: list[Path] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if set(path.relative_to(root_dir).parts) & _ASSET_SKIP_DIRS:
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in _ASSET_SKIP_SUFFIXES:
            continue
        if path.suffix.lower() in _ASSET_SKIP_SOURCE:
            continue
        if path.resolve() in translated:
            continue
        out.append(path)
    return out
