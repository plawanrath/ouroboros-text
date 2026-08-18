---
title: A Hazardous Document
author: Jane Doe
---

# Introduction

This is a plain paragraph of prose that should be translated. It mentions
a citation [@smith2020] and some inline math $O(n \log n)$ and a bare URL
<https://example.com/paper.pdf> plus `inline_code()` and a footnote[^1].

## Related Work

Prior work established the baseline. See [the survey](https://arxiv.org/abs/1234.5678)
for a broader treatment.

$$
\mathcal{L}(\theta) = \sum_{i=1}^{N} \log p(y_i \mid x_i; \theta)
$$

| Method | Accuracy | Latency |
|--------|---------:|--------:|
| Ours   |     94.7 |    12ms |
| Prior  |     91.2 |    31ms |

![Figure 1: Architecture of the proposed system.](figures/arch.png)

```python
def train(model, data):
    # this comment must not be translated
    return model.fit(data)
```

Another prose paragraph follows the code block and should be translated
normally, because it is ordinary running text.

[^1]: This is a footnote definition.

[survey]: https://example.com/survey
