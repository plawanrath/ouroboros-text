# Nested Containers

An ordinary paragraph before the list.

- A bullet whose prose runs long enough that the author hard wrapped it
  onto a second line, which carries a two space continuation indent.
- A short bullet with a citation [@smith2020].
- A bullet containing a nested list:
  - An inner bullet with its own prose.
  - Another inner bullet mentioning $x^2$.

1. An ordered item with prose in it.
2. A second ordered item that also runs long enough to wrap onto
   a continuation line with a three space indent.

> A blockquote paragraph with real prose that is long enough to be
> wrapped across two lines, each carrying the marker.

> - A bullet inside a blockquote.

Text after the containers, carrying a footnote reference[^1].

| Method | Score |
|--------|------:|
| Ours   |  94.7 |

- A bullet with a fenced block under it:

  ```python
  def f():
      return 1
  ```

- [ ] An unchecked task with prose in it.
- [x] A checked task with prose in it.

A paragraph with a hard break at the end of this line  
and a second line that follows the break.

Setext Heading
==============

[^1]: A footnote definition, which is prose and should be translated
    across its continuation line too.
