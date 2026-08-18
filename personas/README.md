# Personas

A persona is a description of one author's writing style. It is injected into
the system prompt of the **return leg** of the round trip, where English is
reconstructed from the pivot language, because that is the only leg whose output
a reader ever sees.

Personas are private and are not checked in. `.gitignore` excludes everything in
this directory except `example.md` and this file.

## Setup

```sh
cp personas/example.md personas/default.md
$EDITOR personas/default.md
```

`ouroboros translate` uses `personas/default.md` when `--persona` is not given.
If no default persona exists, the run proceeds with no style shaping and says
so. Nothing breaks on a fresh clone.

Select a different one by filename stem:

```sh
ouroboros translate paper.tex --persona formal-academic
ouroboros translate paper.tex --persona none      # disable style shaping
```

## Why the style guide is private but the prompt is not

The translation mechanics live in `src/ouroboros/prompts.py` and are checked in:
placeholder handling, "do not summarise", "preserve every number". Those are
properties of the tool and are the same for everyone.

The persona is the part that describes a specific person's voice. That is the
only piece worth keeping out of a public repository, so it is the only piece
this directory holds.

## Format

Optional YAML-ish frontmatter carrying the mechanically enforced rules, then
free-form Markdown guidance for the model.

```markdown
---
name: default
description: One line describing the voice.
forbid:
  "—": ", "
banned_openers:
  - "Furthermore,"
---

Prose guidance goes here.
```

The split is deliberate. `forbid` and `banned_openers` are applied in code by a
deterministic post-processing pass, because a rule that can be checked by a
regex should never be left to a model that is free to forget it. Everything a
regex cannot express, such as rhythm and sentence construction, goes in the
prose body and is handled by the model.

## Two things to keep in mind

A persona file becomes part of a system prompt, so treat one you did not write
the way you would treat any other untrusted input. Loading is hardened against
the obvious failure modes: names are restricted to a safe character set so
`--persona ../../secrets` cannot escape this directory, files are size-capped,
and model control tokens are stripped so a persona cannot forge a conversation
turn. None of that makes a persona from a stranger a good idea.

A persona shapes expression only. It is layered under the translation rules and
is explicitly told that it may not add a claim, drop a claim, or change a hedge.
Meaning wins over style whenever the two conflict.
