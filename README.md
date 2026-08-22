# ouroboros-text

Round-trip LaTeX and Markdown through a pivot language with a local model, and
get back prose that means the same thing in your own voice.

The document goes out to French and comes back to English. References, tables,
figures, captions, equations, code, and citations never make the trip. They are
preserved byte for byte.

## Why the output is safe

The obvious way to build this is to parse the document, translate the prose
nodes, and render the tree back out. That approach corrupts documents, because
every construct the parser did not model gets normalised on the way out.

This tool never renders from a parse tree. It locates the character ranges that
hold prose, translates only those, and splices the results into the original
string. Everything else is not carefully preserved, it is never read into the
output path at all. A gap in the parser makes the tool translate less. It cannot
make it corrupt more.

The test suite asserts exactly that: every byte outside a prose span is
identical before and after a full round trip.

## Install

Python 3.11 or newer, and a machine that can hold a 30B model in memory.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

`llama-cpp-python` compiles from source and takes several minutes. Metal is
enabled by default on Apple Silicon.

Then fetch the model, once. It is about 19 GB and lands in `models/`, which is
gitignored.

```sh
.venv/bin/ouroboros download
```

## Set up a persona

A persona is a description of your writing style. It is injected into the return
leg of the round trip, which is the only leg whose output anyone reads.

```sh
cp personas/example.md personas/default.md
$EDITOR personas/default.md
```

Personas are gitignored. See `personas/README.md` for the format and for why the
style guide is private while the translation prompt is not.

The tool runs without one. It will say so and skip the style shaping.

## Use

```sh
# See what will and will not be translated, without loading a model.
ouroboros inspect paper.tex

# See exactly what the model will receive, placeholders and all.
ouroboros translate paper.tex --dry-run

# Round trip a file, a list of files, or a directory.
ouroboros translate paper.tex -o out/
ouroboros translate papers/ -o out/

# A multi-file paper: point at the root and the sections come too.
ouroboros translate paper/main.tex -o out/

# Review where the round trip stopped saying what the source said.
ouroboros report out/

# Show installed models, personas, formats, and backends.
ouroboros list
```

Inputs are never modified in place. Output goes to `out/`, alongside a
`<name>.report.json` recording every segment, its intermediate French, how many
attempts it took, and any segment that fell back.

### Options worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--model` | the only model in `models/` | Required when more than one is installed. |
| `--persona` | `personas/default.md` | `none` disables style shaping. |
| `--pivot` | `fr` | Accepts a chain: `--pivot fr,de` goes English, French, German, English. |
| `--effort` | `low` | Model reasoning strength. Translation rarely needs more. |
| `--attempts` | `2` | Tries per segment before keeping the original. |
| `--backend` | `llamacpp` | `ollama` is the alternative. |
| `--reflow` | off | Let paragraphs become single long lines instead of matching the source's wrapping. |
| `--glossary` | `glossary.txt` if present | Terms that must survive verbatim. |
| `--sample N` | off | Translate N paragraphs spread through each file, for a spot check. |
| `--limit N` | off | Translate only the first N paragraphs of each file. |
| `--no-cache` | off | Runs are cached and resumable by default. |

The cache is keyed on the model, the hop, the full system prompt, and the exact
input text, so editing a prompt or a persona correctly invalidates it.

## Freezing a region

Some prose is not yours to reword. A NeurIPS checklist carries questions whose
wording the template prescribes; a licence notice, a quoted passage, or an
author list is prose by every structural test this tool applies and must still
come back byte-identical.

Mark it with comments, which are invisible in the rendered document and cost
nothing to leave in permanently:

```latex
% ouroboros: off
Do the main claims made in the abstract accurately reflect the contributions?
% ouroboros: on
```

```markdown
<!-- ouroboros: off -->
Prescribed wording nobody may reword.
<!-- ouroboros: on -->
```

Segments inside a disabled region are simply not reported as prose, so they are
protected by exactly the same mechanism as a table or an equation rather than by
a special case anywhere downstream.

An unclosed `off` runs to the end of the file: forgetting to re-enable gives you
less translation than you meant, which is the harmless direction to be wrong in.
Markers do not nest, so a duplicated `off` cannot silently swallow the rest of
the document.

## Holding specific words steady

Three things drift that should not, and all three are fixed against the source
rather than against a dictionary.

**Spelling variants.** If the author wrote `emphasised` and the round trip
returns `emphasized`, the author's form is restored. This needs no word list and
carries no risk, because a word is only ever replaced by a form the source
document itself contains. It works in both directions and has no opinion about
which variant is correct.

**Heading capitalisation.** `Related Work` came back as `Related work`, and a
prompt instruction did not reliably stop it. Headings now have their
capitalisation restored word by word from the source. Restricted to headings on
purpose: applied to a paragraph it would capitalise a word mid-sentence merely
because some other sentence began with it.

**A glossary**, for terms that must survive verbatim:

```sh
cp glossary.example.txt glossary.txt     # picked up automatically
ouroboros translate paper.tex --glossary terms.txt -o out/
```

Glossary terms become placeholders, so their survival is a property of the
pipeline rather than a hope about the model. That guarantee is not free, and it
is worth knowing what it costs. Measured on the real model:

| Input to the model | French | Back to English |
|---|---|---|
| `The attention head learns ...` | `La tête d'attention apprend ...` ✓ | `The attention head learns ...` ✓ |
| `The [[0]] learns ...` | `Le [[0]] apprend ...` ✗ | nearby wording drifted |

With the noun hidden, the model has nothing to agree with, guesses the gender,
and the words around it drift too. **So the glossary is for names and
identifiers**, which carry no grammatical role and pay none of that cost: a
product name, a model identifier, a defined shorthand. Ordinary technical
vocabulary is better left visible, because the model already handles it.

`glossary.txt` is gitignored, since a glossary can name unpublished work.

## Runs that take hours

A paper is an hours-long job, so the tool says what it will cost before it
starts, and the estimate accounts for what is already cached:

```
$ ouroboros translate paper/main.tex -o out/
142 segments across 6 files, 96 already cached, 46 to translate, ~19 min (25s/segment)
  press Ctrl-C to stop; finished paragraphs are kept and cached
```

**Ctrl-C stops after the current paragraph**, not in the middle of one. What is
finished gets spliced and written, the rest stays in English, and the file is a
valid document either way. A second Ctrl-C aborts immediately. Re-running the
same command picks up where it left off, because the cache already holds every
completed hop:

```
$ ouroboros translate paper.md -o out/     # after stopping part way
4 segments, 3 already cached, 1 to translate, ~25s
```

**Spot-check before committing the afternoon.** `--sample N` translates N
paragraphs spread evenly through each file, which tells you far more than
`--limit N`, whose first N paragraphs are all introduction. Unselected
paragraphs are left in English; nothing is dropped.

```sh
ouroboros translate paper/main.tex --sample 5 -o preview/
```

The pace estimate starts from a measured default of 25s per segment and then
replaces it with whatever this machine actually does.

## Did the meaning survive?

Structure being preserved is not the same as claims being preserved, so the
round trip is checked against the source for the specific ways a technical
document goes wrong: a number changes, an identifier is swapped, a negation
appears or vanishes, a hedge is firmed up or softened, a clause is dropped.

```sh
ouroboros report out/
```

```
6 segments, 6 translated, 0 kept in English, 0 flagged
No segment changed a number, an identifier, a negation, or a hedge.
```

There are two tiers. **Numbers and identifiers block**: a changed digit is not
drift, it is a false claim, so it costs a retry and then a fallback and never
reaches the file. **Negation, hedging, and structure are reported**: they are
real but fuzzier, and a check that fails a three-hour run over the word
"generally" is a check people switch off.

### Why not embeddings

The obvious design is to embed both versions and compare them. It does not work,
and the failure is not subtle. Measured with `nomic-embed-text` on this
project's own output:

| Pair | Cosine similarity |
|---|---|
| Meaning negated ("does not halve") | 0.976 |
| Numbers changed (256 → 512, 16 → 64) | 0.996 |
| A correct round trip of a short bullet | 0.676 |
| An unrelated sentence | 0.253 |

Sentence embeddings encode what a passage is *about*, not what it *asserts*. A
falsified number is invisible to them and a negation nearly so, while a short
correct paraphrase scores below both. Ranking by similarity would surface the
good bullets and wave through the broken claim.

The exact checks catch every one of those corruptions, and on twenty genuine
round trips from real runs they produced zero false positives. They are also
locale-proof: French writes `1 000,5` where English writes `1,000.5`, and
comparison is on digit sequences, so that is not a change.

## Multi-file papers

Point at the root file and the whole project comes with it. `\input`,
`\include`, and `\subfile` are followed from the root, the output mirrors the
source directory tree, and the figures, `.bib`, and local `.sty` files are
copied alongside so the translated project still builds.

```sh
ouroboros inspect paper/main.tex      # what will be reached, and what will not
ouroboros translate paper/main.tex -o out/
```

```
out/
  main.tex
  sections/introduction.tex
  sections/method.tex
  refs.bib          <- copied, not translated
  figures/arch.pdf  <- copied, not translated
```

Three details are worth knowing.

An **included section has no `\begin{document}`**, and a `.tex` file without one
is normally treated as having no prose at all. That default is right when you
point the tool at a `.sty` by mistake and wrong for every section of a real
paper, so traversal marks what it finds as a fragment. To translate one section
on its own, pass `--fragment`; without it the tool tells you why it did nothing
rather than silently producing a copy.

An **unresolved include is reported, never skipped quietly.** A missing section
is the difference between translating a paper and translating its title page.

An **`\input` cannot leave the project directory.** `\input{../../etc/passwd}`
resolves to nothing and is reported. Cycles terminate, and both file count and
depth are bounded.

`--no-follow-inputs` translates only the file you named. `--no-copy-assets`
writes the `.tex` files alone.

## What gets translated

| Translated | Protected |
|---|---|
| Paragraphs | Tables, in full |
| Headings, without their `#` or `\section{}` markup | Figures, images, and their captions |
| List items, at any nesting depth | Code blocks, fences, `verbatim`, `lstlisting` |
| Ordered list items | Display and inline math |
| Blockquotes, including lists inside them | Citations, `\ref`, `\label`, and other identifiers |
| Footnote definitions | Footnote *markers* such as `[^1]` |
| Abstracts, `quote`, `description` bodies | LaTeX preamble, `%` comments, bibliography |
| Link and `\href` visible labels | URLs, front matter, link reference definitions |

Two of these were judgement calls. **Footnote definitions are prose**, so they
are translated, while the `[^1]:` label that names them is not. **Front matter
is left alone** even though a `title:` field is arguably prose, because it is
structured metadata and a translator that reflows a YAML value breaks the file.
Both are one line to change if you disagree.

Container structure is preserved exactly: the marker on the first line, and a
matching prefix on every continuation line. A bullet that grows from one line to
two during translation gets the indent it needs rather than escaping its list.

Two kinds of markup are invisible and easy to destroy, so both are handled
explicitly. A **task list checkbox** is not CommonMark, so it arrives as
ordinary prose and a translator will cheerfully move it to the end of the
sentence; it is masked. A **hard line break** is two trailing spaces, which
rewrapping would silently eat; the block is reported as two spans with the break
protected between them.

## How a paragraph is processed

1. The format module reports which character ranges are prose. Everything else
   is protected by omission. A range inside a container starts after the marker
   and records the prefix its continuation lines need.
2. The container prefix is stripped, so the model sees a paragraph rather than a
   bullet's indentation.
3. Inline fragments inside a prose range, such as `\cite{...}`, `$O(n)$`, and
   `` `code` ``, are replaced by placeholders like `[[0]]`.
4. The masked text is translated to the pivot language, then back.
5. The result is validated. Placeholders must return with the same identities
   and the same multiplicity, no control tokens may leak, the length must be
   within a sane ratio of the input, and every number must come back unchanged.
6. The persona's mechanical rules are applied in code, not by prompt.
7. Artifacts of the pivot language are stripped. French typography puts a narrow
   no-break space before `;` `:` `!` `?`, and that character rides back into the
   English as "bold ." It is removed on every run, whatever the persona says.
8. The paragraph is re-wrapped to the source's line width, and placeholders are
   restored and spliced into the original.

Steps 7 and 8 both run while the placeholders are still in place. Once restored,
a paragraph contains fragments like `$O(n \log n)$` that hold spaces but must
never be split across lines.

A segment that fails validation twice is left in English and recorded in the
report. One untranslated paragraph is visible and harmless. A paragraph that
silently lost a citation, or came back with 512 where the source said 256, is
neither.

## Extending it

Each of these is one file and no changes anywhere else.

- **A new input format** implements `parse(source) -> Document` and registers
  itself in `src/ouroboros/formats/`. It answers one question: which ranges are
  prose?
- **A new backend** implements `generate(system, user) -> str` and registers
  itself in `src/ouroboros/backends/`.
- **A new persona** is a Markdown file in `personas/`.
- **A longer cycle** is a config change. The hops are a list, not a hardcoded
  there-and-back.

## Tests

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

The suite runs against a mock backend and needs no model weights, because the
guarantee about protected content is a property of the segmentation and splicing
layers rather than of the model.

## Notes on the model

Muse-Glimmer-30B at Q5_K_M. On an M4 Max it decodes at roughly 13 to 15 tokens
per second, and a paragraph costs two calls at roughly 25 seconds each. A long
paper is therefore an hours-long job. The cache makes re-runs free, so start with
`inspect` and `--dry-run` and only commit to a full run once the segmentation
looks right.

Reasoning strength is set to `low`. The model still emits a reasoning block, and
that is deliberate: forcing it to skip one by prefilling the answer channel makes
it think in the visible output instead, which is both slower and wrong.
