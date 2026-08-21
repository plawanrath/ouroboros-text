"""Multi-file LaTeX projects.

A paper is a root file plus the sections it includes, so being handed main.tex
means being asked to translate the project. These tests pin down the three
things that make that safe: reaching every section, refusing to leave the
project directory, and terminating on a cycle.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ouroboros import project
from ouroboros.formats import base as formats

formats.load_builtins()

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT = FIXTURES / "project"


# --------------------------------------------------------------- discovery


def test_every_included_section_is_found():
    files, _ = project.discover(PROJECT / "main.tex")
    found = {str(f.relative) for f in files}
    assert found == {
        "main.tex",
        "sections/introduction.tex",
        "sections/method.tex",
        "shared/notation.tex",     # reached at depth 2, via introduction.tex
    }


def test_the_root_is_standalone_and_the_sections_are_fragments():
    """The distinction decides whether a file's prose is seen at all."""
    files, _ = project.discover(PROJECT / "main.tex")
    by_name = {str(f.relative): f for f in files}
    assert by_name["main.tex"].fragment is False
    assert by_name["sections/introduction.tex"].fragment is True
    assert by_name["shared/notation.tex"].depth == 2


def test_a_commented_out_include_is_not_followed():
    files, _ = project.discover(PROJECT / "main.tex")
    assert not any("commented_out" in str(f.relative) for f in files)


def test_an_unresolved_include_is_reported_not_silently_dropped():
    """A missing section is the difference between translating a paper and
    translating its title page, so it has to be visible."""
    _, warnings = project.discover(PROJECT / "main.tex")
    assert any("missing_file" in w for w in warnings)


# ---------------------------------------------------------------- security


def test_an_include_cannot_escape_the_project(tmp_path):
    """\\input{../secret} must not be resolved, read, translated, or written."""
    (tmp_path / "secret.tex").write_text("SECRET\n", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.tex").write_text(
        "\\begin{document}\n\\input{../secret}\n\\input{/etc/passwd}\n\\end{document}\n",
        encoding="utf-8",
    )

    files, warnings = project.discover(proj / "main.tex")
    assert [str(f.relative) for f in files] == ["main.tex"]
    assert len(warnings) == 2
    for f in files:
        assert "secret" not in str(f.path)
        assert "passwd" not in str(f.path)


def test_a_cycle_terminates(tmp_path):
    (tmp_path / "main.tex").write_text(
        "\\begin{document}\n\\input{a}\n\\end{document}\n", encoding="utf-8")
    (tmp_path / "a.tex").write_text("Text in a.\n\\input{b}\n", encoding="utf-8")
    (tmp_path / "b.tex").write_text("Text in b.\n\\input{a}\n", encoding="utf-8")

    files, _ = project.discover(tmp_path / "main.tex")
    assert [str(f.relative) for f in files] == ["main.tex", "a.tex", "b.tex"]


def test_file_count_is_bounded(tmp_path):
    """A generated or pathological project must not run away."""
    (tmp_path / "main.tex").write_text(
        "\\begin{document}\n" + "".join(f"\\input{{s{i}}}\n" for i in range(20))
        + "\\end{document}\n", encoding="utf-8")
    for i in range(20):
        (tmp_path / f"s{i}.tex").write_text(f"Section {i} prose.\n", encoding="utf-8")

    files, warnings = project.discover(tmp_path / "main.tex", max_files=5)
    assert len(files) <= 5
    assert any("stopped after" in w for w in warnings)


# ---------------------------------------------------------------- fragments


def test_a_fragment_yields_no_prose_without_the_flag():
    """The safe default. A .tex with no document environment is usually a style
    file, and translating a preamble would be destructive."""
    source = (PROJECT / "sections" / "introduction.tex").read_text(encoding="utf-8")
    doc = formats.get("latex").parse(source)
    assert doc.segments == []


def test_a_fragment_yields_prose_with_the_flag():
    source = (PROJECT / "sections" / "introduction.tex").read_text(encoding="utf-8")
    doc = formats.get("latex").parse(source, fragment=True)
    text = " ".join(s.text for s in doc.segments)
    assert "This introduction lives in an included file" in text
    assert doc.unchanged() == source


def test_a_fragment_still_protects_what_it_should():
    source = (PROJECT / "sections" / "method.tex").read_text(encoding="utf-8")
    doc = formats.get("latex").parse(source, fragment=True)
    prose = " ".join(s.text for s in doc.segments)

    assert "whose caption must not be translated" not in prose   # \caption
    assert "\\mathcal{L}" not in prose                            # equation
    assert "\\includegraphics" not in prose
    assert doc.unchanged() == source


def test_is_standalone_ignores_a_commented_document():
    assert project.is_standalone("\\begin{document}\nx\n")
    assert not project.is_standalone("% \\begin{document}\n")
    assert not project.is_standalone("just prose\n")


# ------------------------------------------------------------------ assets


def test_assets_are_the_files_that_are_not_translated():
    files, _ = project.discover(PROJECT / "main.tex")
    translated = {f.path for f in files}
    assets = project.asset_files(PROJECT, translated)
    names = {a.name for a in assets}

    assert "refs.bib" in names          # needed by \bibliography
    assert "arch.pdf" in names          # needed by \includegraphics
    assert not names & {p.name for p in translated}


@pytest.mark.parametrize("junk", ["main.aux", "main.log", "main.fdb_latexmk"])
def test_build_products_are_not_copied(tmp_path, junk):
    (tmp_path / "main.tex").write_text("\\begin{document}\nx\n\\end{document}\n",
                                       encoding="utf-8")
    (tmp_path / junk).write_text("noise\n", encoding="utf-8")
    assets = project.asset_files(tmp_path, {(tmp_path / "main.tex").resolve()})
    assert junk not in {a.name for a in assets}


# ------------------------------------------------------------- planning


def _plan_for(*paths, **kw):
    from ouroboros.cli import _plan
    kw.setdefault("follow_inputs", True)
    kw.setdefault("fragment", False)
    kw.setdefault("copy_assets", False)
    return _plan(tuple(str(p) for p in paths), **kw)


def test_pointing_at_a_directory_still_finds_the_root_first(tmp_path):
    """Regression: alphabetical expansion could plan a section before its root.

    The section would then be planned as a standalone file and skipped by the
    root's traversal as already seen, so it would be parsed without the
    fragment flag and yield no prose at all.
    """
    proj = tmp_path / "paper"
    (proj / "sections").mkdir(parents=True)
    # "zmain" sorts after "sections/", so naive ordering visits the section first.
    (proj / "zmain.tex").write_text(
        "\\begin{document}\n\\input{sections/intro}\n\\end{document}\n", encoding="utf-8")
    (proj / "sections" / "intro.tex").write_text(
        "Introduction prose that must be reached.\n", encoding="utf-8")

    plan = _plan_for(proj)
    by_rel = {str(i.relative): i for i in plan.items}
    assert by_rel["sections/intro.tex"].fragment is True


def test_output_paths_mirror_the_project_tree():
    plan = _plan_for(PROJECT / "main.tex")
    rels = sorted(str(i.relative) for i in plan.items)
    assert rels == ["main.tex", "sections/introduction.tex",
                    "sections/method.tex", "shared/notation.tex"]


def test_two_files_with_the_same_name_do_not_collide():
    """A flat output directory would overwrite one method.tex with another."""
    plan = _plan_for(PROJECT / "main.tex")
    assert len({str(i.relative) for i in plan.items}) == len(plan.items)


def test_no_follow_inputs_translates_only_the_named_file():
    plan = _plan_for(PROJECT / "main.tex", follow_inputs=False)
    assert [str(i.relative) for i in plan.items] == ["main.tex"]


def test_a_fragment_opened_directly_warns_instead_of_doing_nothing():
    plan = _plan_for(PROJECT / "sections" / "introduction.tex")
    assert any("--fragment" in w for w in plan.warnings)
    assert plan.items[0].fragment is False


def test_the_fragment_flag_makes_a_directly_opened_section_work():
    plan = _plan_for(PROJECT / "sections" / "introduction.tex", fragment=True)
    assert plan.items[0].fragment is True
    assert not plan.warnings


def test_assets_are_planned_with_their_relative_paths():
    plan = _plan_for(PROJECT / "main.tex", copy_assets=True)
    rels = {str(rel) for _, rel in plan.assets}
    assert "refs.bib" in rels
    assert "figures/arch.pdf" in rels


# ------------------------------------------------------------ asset safety


def test_a_lone_tex_file_does_not_claim_its_directory_as_a_project():
    """Regression, and the worst bug this project has had.

    A single .tex was treated as a project root, its containing directory
    became the asset tree, and the asset copy then overwrote the tool's own
    translated output with the untouched original. Silent data loss.
    """
    plan = _plan_for(FIXTURES / "hazard.tex", copy_assets=True)

    assert [str(i.relative) for i in plan.items] == ["hazard.tex"]
    assert plan.assets == [], f"a lone file claimed assets: {plan.assets}"


def test_a_real_project_still_collects_its_assets():
    plan = _plan_for(PROJECT / "main.tex", copy_assets=True)
    assert {str(rel) for _, rel in plan.assets} == {"refs.bib", "figures/arch.pdf"}


def test_source_files_are_never_treated_as_assets():
    """An untranslated .tex copied in beside the translated ones looks like
    output but is not."""
    files, _ = project.discover(PROJECT / "main.tex")
    assets = project.asset_files(PROJECT, {f.path for f in files})
    assert not [a for a in assets if a.suffix in (".tex", ".md")]


def test_translating_two_files_side_by_side_does_not_clobber_either(tmp_path):
    """The exact shape that broke: a .md and a .tex in one directory."""
    (tmp_path / "doc.md").write_text("A paragraph of ordinary prose here.\n",
                                     encoding="utf-8")
    (tmp_path / "paper.tex").write_text(
        "\\begin{document}\nA paragraph of ordinary prose here.\n\\end{document}\n",
        encoding="utf-8")

    plan = _plan_for(tmp_path / "doc.md", tmp_path / "paper.tex", copy_assets=True)
    destinations = {str(i.relative) for i in plan.items}
    assert destinations == {"doc.md", "paper.tex"}
    assert not [rel for _, rel in plan.assets if str(rel) in destinations]
