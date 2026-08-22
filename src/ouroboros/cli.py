"""Command line interface.

Five commands: fetch a model, inspect what a document parses into, run the round
trip, review where it drifted, and list what is installed. The inspect command
exists because the most common failure is not a bad translation, it is a block
classified wrongly, and seeing the segmentation finds that in one second instead
of after a twenty minute run.
"""
from __future__ import annotations

import json
import shutil
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import click

from . import config as config_mod
from . import mining, modelstore, progress, project, terms
from . import persona as persona_mod
from .backends import base as backends
from .formats import base as formats
from .pipeline import Cache, RoundTrip

formats.load_builtins()


def _echo_err(msg: str) -> None:
    click.secho(msg, fg="red", err=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="ouroboros-text")
def app() -> None:
    """Round-trip LaTeX and Markdown through a pivot language with a local model."""


# ------------------------------------------------------------------ download


@app.command()
@click.option("--repo", default=modelstore.DEFAULT_REPO, show_default=True)
@click.option("--file", "filename", default=modelstore.DEFAULT_FILE, show_default=True)
@click.option("--models-dir", default=str(modelstore.DEFAULT_DIR), show_default=True)
def download(repo: str, filename: str, models_dir: str) -> None:
    """Download the model into models/ (one time, ~19 GB)."""
    click.echo(f"fetching {filename} from {repo}")
    path = modelstore.download(repo, filename, models_dir)
    size = path.stat().st_size / 1e9
    click.secho(f"ready: {path} ({size:.1f} GB)", fg="green")


# --------------------------------------------------------------------- list


@app.command(name="list")
@click.option("--models-dir", default=str(modelstore.DEFAULT_DIR), show_default=True)
@click.option("--personas-dir", default=str(persona_mod.DEFAULT_DIR), show_default=True)
def list_(models_dir: str, personas_dir: str) -> None:
    """Show installed models, personas, formats, and backends."""
    backends.load_builtins()

    click.secho("models", bold=True)
    found = modelstore.discover(models_dir)
    if not found:
        click.echo(f"  none in {models_dir}/  (run: ouroboros download)")
    for m in found:
        marker = "*" if len(found) == 1 else " "
        click.echo(f"  {marker} {m.name}  ({m.size_gb:.1f} GB)")
    if len(found) == 1:
        click.echo("  (* used by default)")

    click.secho("\npersonas", bold=True)
    names = persona_mod.available(personas_dir)
    if not names:
        click.echo(f"  none in {personas_dir}/  (cp {personas_dir}/example.md "
                   f"{personas_dir}/default.md)")
    for n in names:
        marker = "*" if n == persona_mod.DEFAULT_NAME else " "
        click.echo(f"  {marker} {n}")

    click.secho("\nformats", bold=True)
    click.echo("  " + " ".join(formats.supported_extensions()))
    click.secho("\nbackends", bold=True)
    click.echo("  " + " ".join(backends.available()))


# ------------------------------------------------------------------ inspect


@app.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--show-protected/--no-show-protected", default=True,
              help="Dim the regions that will never reach the model.")
@click.option("--follow-inputs/--no-follow-inputs", default=True, show_default=True,
              help="Follow LaTeX \\input and \\include from a root file.")
@click.option("--fragment", is_flag=True,
              help="Treat a .tex file with no \\begin{document} as a section.")
def inspect(paths: tuple[str, ...], show_protected: bool,
            follow_inputs: bool, fragment: bool) -> None:
    """Show what a document parses into, without running a model.

    Green is prose that will be translated. Dim is protected and passes through
    untouched. This is the fastest way to catch a misclassified block, and on a
    multi-file paper it is the fastest way to catch a section that traversal
    never reached.
    """
    plan = _plan(paths, follow_inputs=follow_inputs, fragment=fragment,
                 copy_assets=False)
    for warning in plan.warnings:
        click.secho(f"warning: {warning}", fg="yellow")

    total_prose = total_chars = 0
    for item in plan.items:
        path = item.source
        source = path.read_text(encoding="utf-8")
        fmt = formats.for_path(path)
        doc = fmt.parse(source, path=str(path), fragment=item.fragment)

        prose_chars = sum(len(s) for s in doc.segments)
        total_prose += prose_chars
        total_chars += len(source)
        pct = 100 * prose_chars / max(len(source), 1)
        suffix = "  (fragment)" if item.fragment else ""
        click.secho(f"\n{item.relative}  [{fmt.name}]{suffix}", bold=True)
        click.echo(f"  {len(doc.segments)} prose segments, "
                   f"{prose_chars}/{len(source)} chars ({pct:.0f}%)\n")

        cur = 0
        for seg in doc.segments:
            if show_protected and seg.start > cur:
                click.secho(source[cur:seg.start], dim=True, nl=False)
            click.secho(seg.text, fg="green", nl=False)
            cur = seg.end
        if show_protected and cur < len(source):
            click.secho(source[cur:], dim=True, nl=False)
        click.echo()

        # The guarantee, asserted rather than assumed.
        assert doc.unchanged() == source, "splice is not lossless"

    if len(plan.items) > 1:
        pct = 100 * total_prose / max(total_chars, 1)
        click.secho(
            f"\n{len(plan.items)} files, {total_prose}/{total_chars} chars "
            f"({pct:.0f}%) is prose",
            bold=True,
        )


# ------------------------------------------------------------------- report


@app.command()
@click.argument("reports", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-n", "--limit", default=10, show_default=True,
              help="How many flagged segments to show.")
@click.option("--all", "show_all", is_flag=True,
              help="Show every flagged segment, not just the worst.")
def report(reports: tuple[str, ...], limit: int, show_all: bool) -> None:
    """Show where a run's output stopped saying what the source said.

    Reads the report.json files a translation writes. Accepts a directory, so
    a whole multi-file project can be reviewed at once.
    """
    paths: list[Path] = []
    for raw in reports:
        p = Path(raw)
        paths += sorted(p.rglob("*.report.json")) if p.is_dir() else [p]

    if not paths:
        _echo_err("no report.json files found")
        sys.exit(1)

    flagged: list[tuple[str, dict]] = []
    totals = {"segments": 0, "translated": 0, "fallback": 0}

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        totals["segments"] += data.get("segments_total", 0)
        totals["translated"] += data.get("segments_translated", 0)
        totals["fallback"] += data.get("segments_fallback", 0)
        for seg in data.get("segments", []):
            if seg.get("issues"):
                flagged.append((data.get("path", str(path)), seg))

    def severity_rank(item):
        _, seg = item
        high = sum(1 for i in seg["issues"] if i.get("severity") == "high")
        return (-high, -len(seg["issues"]))

    flagged.sort(key=severity_rank)

    click.secho(
        f"{totals['segments']} segments, {totals['translated']} translated, "
        f"{totals['fallback']} kept in English, {len(flagged)} flagged",
        bold=True,
    )

    if not flagged:
        click.secho(
            "\nNo segment changed a number, an identifier, a negation, or a hedge.",
            fg="green",
        )
        return

    shown = flagged if show_all else flagged[:limit]
    for source_path, seg in shown:
        high = any(i.get("severity") == "high" for i in seg["issues"])
        click.secho(f"\n{source_path}  {tuple(seg['span'])}", bold=True)
        for issue in seg["issues"]:
            colour = "red" if issue.get("severity") == "high" else "yellow"
            click.secho(f"  {issue['kind']}: {issue['detail']}", fg=colour)
        click.secho(f"  -  {seg['original'].strip()[:300]}", fg="cyan")
        click.secho(f"  +  {seg['final'].strip()[:300]}", fg="green" if not high else "red")

    if not show_all and len(flagged) > limit:
        click.echo(f"\n... {len(flagged) - limit} more (--all to see them)")


# ------------------------------------------------------------------ glossary


@app.command()
@click.argument("outputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--out", "out_path", default=None, type=click.Path(),
              help="Write the glossary here instead of to standard output.")
@click.option("--min-occurrences", default=mining.MIN_OCCURRENCES, show_default=True,
              help="Ignore a term seen fewer times than this. One loss is an anecdote.")
@click.option("--max-survival", default=mining.MAX_SURVIVAL, show_default=True,
              help="Protect a term surviving at most this fraction of the time.")
def glossary(outputs: tuple[str, ...], out_path: str | None,
             min_occurrences: int, max_survival: float) -> None:
    """Learn which terms the round trip cannot carry, from a finished run.

    Reads the report.json files a translation writes and measures, for every
    coined term, how often the output still contained it. A term of art that the
    pivot language has no word for comes back as a synonym nearly every time; an
    ordinary word reworded in one sentence survives in the next twenty.

    Review the result before using it. Protecting a term means the model never
    sees it, so a word that should be translated does not belong in the file.
    """
    pairs = []
    for directory in outputs:
        pairs += mining.pairs_from_reports(directory)

    if not pairs:
        _echo_err("no translated segments found; run a translation first")
        sys.exit(1)

    stats = mining.mine(pairs, min_occurrences, max_survival)
    text = mining.render(stats, f"{len(pairs)} translated segments",
                         min_occurrences, max_survival)

    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        click.secho(f"wrote {len(stats)} term(s) to {out_path}", fg="green")
        for s in stats[:10]:
            click.echo(f"  {s}")
    else:
        click.echo(text, nl=False)


# ---------------------------------------------------------------- translate


@app.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--out", default="out", show_default=True,
              help="Output directory. Inputs are never modified in place.")
@click.option("--model", default=None,
              help="Model name or path. Defaults to the only model in models/.")
@click.option("--models-dir", default=str(modelstore.DEFAULT_DIR), show_default=True)
@click.option("--backend", "backend_name", default="llamacpp", show_default=True)
@click.option("--persona", "persona_name", default=None,
              help="Persona to write back in. Defaults to personas/default.md "
                   "if present, otherwise no style shaping. Use 'none' to disable.")
@click.option("--personas-dir", default=str(persona_mod.DEFAULT_DIR), show_default=True)
@click.option("--pivot", default="fr", show_default=True,
              help="Pivot language, or a comma-separated chain such as 'fr,de'.")
@click.option("--effort", default="low", show_default=True,
              type=click.Choice(["low", "medium", "high", "xhigh"]),
              help="Model reasoning strength. Translation rarely needs more than low.")
@click.option("--attempts", default=2, show_default=True,
              help="Attempts per segment before falling back to the original.")
@click.option("--n-ctx", default=8192, show_default=True)
@click.option("--cache/--no-cache", default=True, show_default=True)
@click.option("--preserve-wrapping/--reflow", default=True, show_default=True,
              help="Re-wrap output to the source's line width. --reflow lets "
                   "paragraphs become single long lines.")
@click.option("--report/--no-report", default=True, show_default=True,
              help="Write report.json alongside the output.")
@click.option("--follow-inputs/--no-follow-inputs", default=True, show_default=True,
              help="Follow LaTeX \\input and \\include from a root file and "
                   "translate the whole project, mirroring its directory tree.")
@click.option("--fragment", is_flag=True,
              help="Treat a .tex file with no \\begin{document} as an included "
                   "section rather than skipping it.")
@click.option("--copy-assets/--no-copy-assets", default=True, show_default=True,
              help="Copy figures, .bib, and .sty files into the output so the "
                   "translated project still builds.")
@click.option("--limit", default=0, show_default=False, metavar="N",
              help="Translate only the first N paragraphs of each file. For "
                   "spot-checking before committing to a long run.")
@click.option("--sample", default=0, show_default=False, metavar="N",
              help="Translate N paragraphs spread evenly through each file. "
                   "More representative than --limit, which only sees the intro.")
@click.option("--glossary", "glossary_path", default=None, type=click.Path(exists=True),
              help="File of terms that must survive verbatim, one per line. "
                   "Defaults to glossary.txt if present. Best for names and "
                   "identifiers: hiding an ordinary noun makes the sentence "
                   "around it worse, since the model loses what to agree with.")
@click.option("--dry-run", is_flag=True,
              help="Parse and mask, but do not load a model or translate.")
def translate(
    paths: tuple[str, ...], out: str, model: str | None, models_dir: str,
    backend_name: str, persona_name: str | None, personas_dir: str, pivot: str,
    effort: str, attempts: int, n_ctx: int, cache: bool, preserve_wrapping: bool,
    follow_inputs: bool, fragment: bool, copy_assets: bool,
    limit: int, sample: int, glossary_path: str | None,
    report: bool, dry_run: bool,
) -> None:
    """Round-trip documents through a pivot language and back."""
    backends.load_builtins()

    # Settings from ouroboros.toml, overridden by anything typed explicitly.
    try:
        settings = config_mod.load()
    except config_mod.ConfigError as e:
        _echo_err(str(e))
        sys.exit(1)
    if settings:
        click.echo(f"config: {settings['_path']}")

    def pick(key, value, default):
        return config_mod.apply(settings, key, value, default)

    out = pick("out", out, "out")
    pivot = pick("pivot", pivot, "fr")
    effort = pick("effort", effort, "low")
    attempts = pick("attempts", attempts, 2)
    backend_name = pick("backend", backend_name, "llamacpp")
    model = pick("model", model, None)
    persona_name = pick("persona", persona_name, None)
    glossary_path = pick("glossary", glossary_path, None)
    n_ctx = pick("n_ctx", n_ctx, 8192)
    cache = pick("cache", cache, True)
    report = pick("report", report, True)
    preserve_wrapping = pick("preserve_wrapping", preserve_wrapping, True)
    follow_inputs = pick("follow_inputs", follow_inputs, True)
    copy_assets = pick("copy_assets", copy_assets, True)

    plan = _plan(paths, follow_inputs=follow_inputs, fragment=fragment,
                 copy_assets=copy_assets)
    if not plan.items:
        _echo_err("no supported files found")
        sys.exit(1)
    for warning in plan.warnings:
        click.secho(f"warning: {warning}", fg="yellow")
    if len(plan.items) > 1:
        click.echo(f"{len(plan.items)} files to translate")

    try:
        persona = persona_mod.load(persona_name, personas_dir)
    except persona_mod.PersonaError as e:
        _echo_err(str(e))
        sys.exit(1)

    if persona.is_none:
        if persona_name is None:
            click.secho(
                f"no {personas_dir}/{persona_mod.DEFAULT_NAME}.md, running without "
                f"style shaping (cp {personas_dir}/example.md "
                f"{personas_dir}/{persona_mod.DEFAULT_NAME}.md to set one up)",
                fg="yellow",
            )
    else:
        click.echo(f"persona: {persona.name}")

    glossary = terms.Glossary.find(glossary_path)
    if glossary:
        click.echo(f"glossary: {len(glossary.terms)} term(s) from {glossary.path}")

    if dry_run:
        for item in plan.items:
            _dry_run(item.source, fragment=item.fragment, glossary=glossary)
        return

    backend_cls = backends.get(backend_name)

    if backend_name == "llamacpp":
        try:
            chosen = modelstore.resolve(model, models_dir)
        except (modelstore.NoModelsFound, modelstore.AmbiguousModel) as e:
            _echo_err(str(e))
            sys.exit(1)
        model_id = chosen.name
        click.echo(f"model: {chosen.name} ({chosen.size_gb:.1f} GB), backend llamacpp")
        backend = backend_cls(chosen.path, n_ctx=n_ctx, effort=effort)
    else:
        # A backend that keeps its own weights has no use for models/, and
        # requiring a local GGUF there would lock out anyone running this
        # through ollama alone. --model names the tag instead.
        model_id = model or getattr(backend_cls, "DEFAULT_MODEL", "default")
        click.echo(f"model: {model_id}, backend {backend_name}")
        try:
            backend = backend_cls(model_id, effort=effort)
        except Exception as e:  # noqa: BLE001 - surfaced to the user as-is
            _echo_err(f"could not start the {backend_name} backend: {e}")
            sys.exit(1)

    trip = RoundTrip(
        backend,
        pivot=[p.strip() for p in pivot.split(",") if p.strip()],
        persona=persona,
        max_attempts=attempts,
        cache=Cache() if cache else None,
        model_id=model_id,
        preserve_wrapping=preserve_wrapping,
        glossary=glossary,
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse everything up front and ask the cache what is already done. On a
    # paper this is the difference between "about four hours" and "about six
    # minutes", and it is worth knowing before committing rather than after.
    docs: list[tuple[WorkItem, object]] = []
    total = cached_total = total_chars = 0
    for item in plan.items:
        source = item.source.read_text(encoding="utf-8")
        doc = formats.for_path(item.source).parse(
            source, path=str(item.source), fragment=item.fragment
        )
        doc = _select(doc, limit, sample)
        done, count, chars = trip.pending(doc)
        total += count
        cached_total += done
        total_chars += chars
        docs.append((item, doc))

    click.secho(progress.summarise(total, cached_total, len(plan.items), total_chars),
                bold=True)
    if limit or sample:
        click.secho(
            "  partial run: unselected paragraphs are left in English", fg="yellow"
        )
    if total:
        click.echo("  press Ctrl-C to stop; finished paragraphs are kept and cached")

    estimator = progress.Estimator(total - cached_total, cached_total, total_chars)
    interrupted = False

    guard = InterruptGuard()
    trip.should_stop = lambda: guard.requested

    with guard:
      for item, doc in docs:
          label = f"\n{item.relative}: {len(doc.segments)} segments"
          if item.fragment:
              label += "  (fragment)"
          click.secho(label, bold=True)

          with click.progressbar(
              length=max(len(doc.segments), 1),
              label="  translating",
              item_show_func=lambda _: estimator.describe(),
          ) as bar:
              def advance(seg_result, bar=bar):
                  estimator.record(seg_result.seconds, len(seg_result.original))
                  bar.update(1)

              trip_result, run_report = trip.run(doc, on_segment=advance)

          target = out_dir / item.relative
          target.parent.mkdir(parents=True, exist_ok=True)
          target.write_text(trip_result, encoding="utf-8")

          fallbacks = run_report.fallbacks
          colour = "yellow" if fallbacks else "green"
          click.secho(
              f"  wrote {target}  "
              f"({run_report.translated}/{len(run_report.segments)} translated, "
              f"{progress.human_duration(run_report.seconds)})",
              fg=colour,
          )
          for f in fallbacks:
              click.secho(f"    kept original at {f.span}: {f.reason}", fg="yellow")

          if report:
              # Keep the full filename, extension included. with_suffix() maps
              # both paper.tex and paper.md onto paper.report.json, and the
              # second one silently overwrites the first one's report.
              report_path = out_dir / item.relative.with_name(
                  item.relative.name + ".report.json"
              )
              report_path.parent.mkdir(parents=True, exist_ok=True)
              report_path.write_text(run_report.to_json(), encoding="utf-8")

          if run_report.interrupted:
              interrupted = True
              break

    # Assets last, so a run interrupted partway leaves the translations rather
    # than a directory of figures.
    if plan.assets and not interrupted:
        written = {(out_dir / item.relative).resolve() for item, _ in docs}
        copied = 0
        for src, rel in plan.assets:
            dest = out_dir / rel
            # Never let an asset land on a file this run translated. Belt and
            # braces alongside the project check above, because the cost of
            # getting it wrong is silently replacing output with its input.
            if dest.resolve() in written:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
        if copied:
            click.echo(f"\ncopied {copied} asset(s) so the output still builds")

    if interrupted:
        click.secho(
            "\nstopped. Everything finished so far is written and cached, so "
            "re-running the same command picks up where this left off.",
            fg="yellow",
        )
        sys.exit(130)


@dataclass
class WorkItem:
    """One file to translate, and where its output belongs."""

    source: Path
    #: Output path relative to the output directory. Mirrors the project layout
    #: so sections/method.tex does not collide with another method.tex.
    relative: Path
    fragment: bool = False


@dataclass
class Plan:
    items: list[WorkItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: (absolute source, path relative to the output directory)
    assets: list[tuple[Path, Path]] = field(default_factory=list)


def _order_roots_first(paths: list[Path], follow_inputs: bool) -> list[Path]:
    """Put standalone LaTeX roots ahead of everything else.

    Handing over a directory expands to files in alphabetical order, which can
    put sections/introduction.tex before main.tex. The section would then be
    planned as a standalone file, and the root's traversal would skip it as
    already seen, so it would be parsed without the fragment flag and yield no
    prose. Visiting roots first makes the result independent of file names.
    """
    if not follow_inputs:
        return paths

    roots, rest = [], []
    for path in paths:
        is_root = False
        if path.suffix.lower() in (".tex", ".latex"):
            try:
                is_root = project.is_standalone(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                is_root = False
        (roots if is_root else rest).append(path)
    return roots + rest


def _plan(paths: tuple[str, ...], *, follow_inputs: bool, fragment: bool,
          copy_assets: bool) -> Plan:
    """Work out every file to translate, following LaTeX includes.

    A paper is a root file plus the sections it pulls in, so being handed
    main.tex means being asked to translate the project. Traversal happens here
    rather than inside the pipeline: what to translate is a question about
    files, and the pipeline's job starts once that is settled.
    """
    plan = Plan()
    seen: set[Path] = set()

    for path in _order_roots_first(_expand(paths), follow_inputs):
        resolved = path.resolve()
        if resolved in seen:
            continue

        source = path.read_text(encoding="utf-8", errors="replace")
        is_tex = path.suffix.lower() in (".tex", ".latex")

        if is_tex and follow_inputs and project.is_standalone(source):
            files, warnings = project.discover(path)
            plan.warnings += warnings
            root_dir = resolved.parent
            for pf in files:
                if pf.path in seen:
                    continue
                seen.add(pf.path)
                plan.items.append(
                    WorkItem(source=pf.path, relative=pf.relative, fragment=pf.fragment)
                )
            # Only a genuine multi-file project has assets. A lone .tex is not
            # a project, and treating its containing directory as one made every
            # neighbouring file an "asset" -- which then overwrote the tool's own
            # translated output with the untouched original.
            if copy_assets and len(files) > 1:
                translated = {i.source for i in plan.items}
                plan.assets += [
                    (a, a.relative_to(root_dir))
                    for a in project.asset_files(root_dir, translated)
                ]
            continue

        # A standalone file, or a fragment the user pointed at directly.
        if is_tex and not project.is_standalone(source) and not fragment:
            plan.warnings.append(
                f"{path} has no \\begin{{document}}, so nothing will be "
                f"translated. Pass --fragment to treat it as an included section."
            )
        seen.add(resolved)
        plan.items.append(
            WorkItem(source=path, relative=Path(path.name), fragment=fragment)
        )

    return plan


class InterruptGuard:
    """Turns Ctrl-C into "stop after this paragraph" instead of "stop now".

    A run is hours long, so the first press should end it tidily: finish the
    paragraph in flight, splice and write what is done, and leave the rest in
    English. A second press means the user has stopped being patient, so the
    default handler goes back on and the process dies immediately.

    Installing a handler explicitly, rather than relying on Python's default,
    also makes the behaviour testable: a background process started without job
    control inherits SIGINT as ignored, and would otherwise be unstoppable.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous = None

    def __enter__(self) -> Self:
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, signum, frame) -> None:
        if self.requested:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        self.requested = True
        click.secho(
            "\nstopping after the current paragraph. Press Ctrl-C again to "
            "abort immediately and lose it.",
            fg="yellow", err=True,
        )


def _select(doc, limit: int, sample: int):
    """Narrow a document to a subset of its paragraphs for a spot check.

    Unselected paragraphs are not lost: anything not reported as prose passes
    through untouched, so a partial run simply translates less.
    """
    n = len(doc.segments)
    if sample and sample < n:
        # Evenly spaced, because the first N paragraphs of a paper are all
        # introduction and tell you nothing about how the method section fares.
        step = n / sample
        return doc.subset(int(i * step) for i in range(sample))
    if limit and limit < n:
        return doc.subset(range(limit))
    return doc


def _dry_run(path: str | Path, fragment: bool = False, glossary=None) -> None:
    """Print each segment exactly as the model will receive it.

    That means after the container prefix is stripped and after masking, not
    before. Showing the raw source here would defeat the point of the command.
    """
    from .masking import Masker, rules_for
    from .rewrap import strip_continuation

    source = Path(path).read_text(encoding="utf-8")
    fmt = formats.for_path(path)
    doc = fmt.parse(source, path=str(path), fragment=fragment)
    masker = Masker(rules_for(fmt.name, glossary))

    click.secho(f"\n{path}  [{fmt.name}]  {len(doc.segments)} segments", bold=True)
    for i, seg in enumerate(doc.segments):
        indent = seg.meta.get("indent", "")
        masked, mapping = masker.mask(strip_continuation(seg.text, indent))

        label = seg.meta.get("block", "")
        if indent:
            label += f"  indent={indent!r}"
        click.echo(f"\n  [{i}] {seg.span} {label}")
        click.secho(f"      {masked}", fg="green")
        for sent, frag in mapping.items():
            click.echo(f"      {sent} = {frag!r}")


def _expand(paths: tuple[str, ...]) -> list[Path]:
    """Accept files or directories, keeping only supported extensions."""
    exts = set(formats.supported_extensions())
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
        elif p.suffix.lower() in exts:
            out.append(p)
    return out


if __name__ == "__main__":
    app()
