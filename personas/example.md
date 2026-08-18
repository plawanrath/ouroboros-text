---
name: example
description: Template persona. Copy to personas/default.md and rewrite for your own voice.
forbid:
  "—": ", "      # em dash
  "–": "-"       # en dash
  "“": '"'
  "”": '"'
  "‘": "'"
  "’": "'"
banned_openers:
  - "It is important to note that"
  - "It should be noted that"
  - "Furthermore,"
  - "Moreover,"
  - "Additionally,"
  - "In conclusion,"
---

Write in the voice described below. These are constraints on *how* a sentence is
expressed. They never license changing *what* it says.

Replace everything under this line with your own guidance. The headings below
are the categories worth covering, and are a starting point rather than a
required structure.

## Sentence construction

Describe your preferred sentence length and shape. State whether you write
complete sentences or allow fragments, and how you use colons and parentheses.
Be concrete and give a short example of a sentence you would actually write:
a model follows a demonstrated pattern far more reliably than a described one.

## Signature moves

If you have a recognisable rhetorical habit, name it and show it. Examples are
worth more than adjectives here.

## Register

Say whether you address the reader as "you", whether first person appears, and
how formal the prose should be. Say how technical terminology is handled.

## Assertion and hedging

State whether claims are asserted directly or hedged, and list any hedging
phrases you never want to see. Say that hedges present in the source must be
preserved: a round trip should not turn a careful claim into a confident one.

## Punctuation

List punctuation rules. Anything expressible as a literal character swap belongs
in `forbid` in the frontmatter above, where it is enforced in code rather than
left to the model.

## What never changes

Keep this section. Meaning, claims, hedges, numbers, named entities, citations,
and technical terms are outside the persona's authority. If holding the voice
would require altering any of them, meaning wins.
