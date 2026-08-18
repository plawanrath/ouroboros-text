"""Command line interface.

Four commands: fetch a model, inspect what a document parses into, run the round
trip, and list what is installed. The inspect command exists because the most
common failure in this tool is not a bad translation, it is a block classified
wrongly, and seeing the segmentation is how you find that in one second instead
of after a twenty minute run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from . import modelstore
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
def inspect(paths: tuple[str, ...], show_protected: bool) -> None:
    """Show what a document parses into, without running a model.

    Green is prose that will be translated. Dim is protected and passes through
    untouched. This is the fastest way to catch a misclassified block.
    """
    for path in _expand(paths):
        source = Path(path).read_text(encoding="utf-8")
        fmt = formats.for_path(path)
        doc = fmt.parse(source, path=str(path))

        prose_chars = sum(len(s) for s in doc.segments)
        pct = 100 * prose_chars / max(len(source), 1)
        click.secho(f"\n{path}  [{fmt.name}]", bold=True)
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
@click.option("--dry-run", is_flag=True,
              help="Parse and mask, but do not load a model or translate.")
def translate(
    paths: tuple[str, ...], out: str, model: str | None, models_dir: str,
    backend_name: str, persona_name: str | None, personas_dir: str, pivot: str,
    effort: str, attempts: int, n_ctx: int, cache: bool, preserve_wrapping: bool,
    report: bool, dry_run: bool,
) -> None:
    """Round-trip documents through a pivot language and back."""
    backends.load_builtins()
    targets = _expand(paths)
    if not targets:
        _echo_err("no supported files found")
        sys.exit(1)

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

    if dry_run:
        for path in targets:
            _dry_run(path)
        return

    try:
        chosen = modelstore.resolve(model, models_dir)
    except (modelstore.NoModelsFound, modelstore.AmbiguousModel) as e:
        _echo_err(str(e))
        sys.exit(1)

    click.echo(f"model: {chosen.name} ({chosen.size_gb:.1f} GB), backend {backend_name}")
    backend_cls = backends.get(backend_name)
    backend = (
        backend_cls(chosen.path, n_ctx=n_ctx, effort=effort)
        if backend_name == "llamacpp"
        else backend_cls(effort=effort, models_dir=models_dir)
    )

    trip = RoundTrip(
        backend,
        pivot=[p.strip() for p in pivot.split(",") if p.strip()],
        persona=persona,
        max_attempts=attempts,
        cache=Cache() if cache else None,
        model_id=chosen.name,
        preserve_wrapping=preserve_wrapping,
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in targets:
        source = Path(path).read_text(encoding="utf-8")
        doc = formats.for_path(path).parse(source, path=str(path))
        click.secho(f"\n{path}: {len(doc.segments)} segments", bold=True)

        with click.progressbar(length=len(doc.segments), label="  translating") as bar:
            trip_result, run_report = _run_with_progress(trip, doc, bar)

        target = out_dir / Path(path).name
        target.write_text(trip_result, encoding="utf-8")

        fallbacks = run_report.fallbacks
        colour = "yellow" if fallbacks else "green"
        click.secho(
            f"  wrote {target}  "
            f"({run_report.translated}/{len(run_report.segments)} translated, "
            f"{run_report.seconds:.0f}s)",
            fg=colour,
        )
        for f in fallbacks:
            click.secho(f"    kept original at {f.span}: {f.reason}", fg="yellow")

        if report:
            report_path = out_dir / f"{Path(path).stem}.report.json"
            report_path.write_text(run_report.to_json(), encoding="utf-8")


def _run_with_progress(trip: RoundTrip, doc, bar):
    """Wrap RoundTrip.run so the progress bar advances per segment."""
    original = trip.translate_segment

    def wrapped(masked):
        result = original(masked)
        bar.update(1)
        return result

    trip.translate_segment = wrapped
    try:
        return trip.run(doc)
    finally:
        trip.translate_segment = original


def _dry_run(path: str) -> None:
    from .masking import Masker, rules_for

    source = Path(path).read_text(encoding="utf-8")
    fmt = formats.for_path(path)
    doc = fmt.parse(source, path=str(path))
    masker = Masker(rules_for(fmt.name))

    click.secho(f"\n{path}  [{fmt.name}]  {len(doc.segments)} segments", bold=True)
    for i, seg in enumerate(doc.segments):
        masked, mapping = masker.mask(seg.text)
        click.echo(f"\n  [{i}] {seg.span} {seg.meta.get('block', '')}")
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
