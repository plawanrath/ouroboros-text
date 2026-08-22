# Evals

A round trip either preserves a paper or it does not, and the only way to know
is to run one through and check. This directory holds the harness. It holds no
papers and no results: those go in `adhoc/`, which is gitignored, because the
inputs are other people's work and the outputs are large.

```sh
mkdir -p adhoc
cp -R /path/to/paper adhoc/original
ouroboros translate adhoc/original/main.tex -o adhoc/translated

python evals/evaluate.py adhoc/original adhoc/translated
python evals/judge.py adhoc/translated 25
```

## What each script answers

`evaluate.py` asks three questions separately, because they fail in different
ways and a single number would hide that.

**Structure** is a hard constraint. Every macro, environment, cross-reference
key, math span, citation, number and identifier must be byte-identical. One
difference is a failure no matter how well the prose reads: a paper that lost a
`\cite` is broken.

**Meaning** is what the round trip exists to preserve. The exact checks from
`ouroboros.fidelity` catch what can be checked exactly. Content-word overlap
ranks the rest, and is used only to decide what a human should read: it is a
coarse signal, and rewording is the point of the exercise, so a low overlap is a
hint rather than a verdict.

**Completion** is the share of segments that came back at all, rather than
falling back to English after failing validation.

`judge.py` asks the model directly whether two passages assert the same thing,
on the longest changed segments. Longest first, because a long paragraph makes
more claims and has more to lose, while a heading can barely change meaning.

Judging with the model that produced the translation is not independent, and the
number should be read with that in mind. It is still worth having: "do these
assert the same thing" is a different task from "translate this", and a failure
would have to survive both roles to stay hidden.

## Reading the scores

`structure preserved` below 100 means something is broken. There is no
acceptable non-zero level of macro loss.

`meaning preserved` counts severe fidelity issues, which are changed numbers and
changed identifiers. These also block during the run, so a low score here means
segments were kept in English rather than corrupted.

`mean content overlap` is deliberately not a target. A round trip that scored
1.00 would have changed nothing, which is a broken round trip rather than a
perfect one.
