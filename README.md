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

Python 3.10 or newer, and a machine that can hold a 30B model in memory.

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

# Show installed models, personas, formats, and backends.
ouroboros list
```

Inputs are never modified in place. Output goes to `out/`, alongside a
`report.json` recording every segment, its intermediate French, how many
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
| `--no-cache` | off | Runs are cached and resumable by default. |

The cache is keyed on the model, the hop, the full system prompt, and the exact
input text, so editing a prompt or a persona correctly invalidates it.

## How a paragraph is processed

1. The format module reports which character ranges are prose. Everything else
   is protected by omission.
2. Inline fragments inside a prose range, such as `\cite{...}`, `$O(n)$`, and
   `` `code` ``, are replaced by placeholders like `[[0]]`.
3. The masked text is translated to the pivot language, then back.
4. The result is validated. Placeholders must return with the same identities
   and the same multiplicity, no control tokens may leak, and the length must be
   within a sane ratio of the input.
5. The persona's mechanical rules are applied in code, not by prompt.
6. Artifacts of the pivot language are stripped. French typography puts a narrow
   no-break space before `;` `:` `!` `?`, and that character rides back into the
   English as "bold ." It is removed on every run, whatever the persona says.
7. The paragraph is re-wrapped to the source's line width, and placeholders are
   restored and spliced into the original.

Steps 6 and 7 both run while the placeholders are still in place. Once restored,
a paragraph contains fragments like `$O(n \log n)$` that hold spaces but must
never be split across lines.

A segment that fails validation twice is left in English and recorded in the
report. One untranslated paragraph is visible and harmless. A paragraph that
silently lost a citation is neither.

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
