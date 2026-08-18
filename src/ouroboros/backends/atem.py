"""The Muse Glimmer ATEM wire protocol.

The model does not emit a plain completion. It emits a sequence of channelled
blocks, each addressed to a recipient:

    <|start|>assistant to=self<|message|>...thinking...<|eom|>
    <|start|>assistant<|message|>...the actual answer...<|eot|>

Only the block addressed to ``user`` is the answer. Everything on ``to=self`` is
reasoning and must be discarded, or a translation comes back with the model's
deliberations glued to the front of it.

We render the prompt here rather than delegating to the GGUF's embedded Jinja
template. The template defaults reasoning strength to ``high``, which is a large
throughput cost on a job that issues two calls per paragraph, and the documented
way to override it is a Jinja keyword argument that llama.cpp does not forward.
Writing the handful of control tokens ourselves makes the effort level settable
and the prompt exactly inspectable.
"""
from __future__ import annotations

import re

from ..persona import strip_control_tokens

BOS = "<|begin_of_text|>"
EOT = "<|eot|>"
EOM = "<|eom|>"

#: Valid values for the reasoning directive the system block carries.
EFFORTS = ("low", "medium", "high", "xhigh")

_CHANNEL = re.compile(
    r"(?:\s*to=(?P<to>[^<\s]+))?\s*<\|message\|>(?P<body>.*?)(?:<\|eom\|>|<\|eot\|>|\Z)",
    re.DOTALL,
)


def render_prompt(system: str, user: str, effort: str = "low", include_bos: bool = True) -> str:
    """Build a single-turn ATEM prompt ending in the generation cue.

    The reasoning directive is written into the system block itself. The model's
    own template skips adding its default when the system text already contains
    the phrase, so this is the one override path that does not depend on the
    runtime forwarding template keyword arguments.

    Both arguments are stripped of control tokens first. Rendering the prompt by
    hand means a document containing the literal text ``<|eot|>``, which is
    entirely possible in a paper about language models, would otherwise close
    the block it sits in and forge a conversation turn. Sanitising here covers
    every caller, so no other layer has to remember to.
    """
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")

    system = strip_control_tokens(system)
    user = strip_control_tokens(user)

    return (
        f"{BOS if include_bos else ''}"
        f"<|start|>system<|message|>{system}\n\n"
        f"Reasoning strength: {effort}.\n\n"
        f'# Valid recipients: "self", "user".{EOT}'
        f"<|start|>user<|message|>{user}{EOT}"
        f"<|start|>assistant"
    )


def parse_response(raw: str) -> str:
    """Extract the user-channel content, discarding reasoning blocks.

    Generation begins immediately after ``<|start|>assistant``, so the first
    block arrives without its opening tag. Prepending one normalises the split.
    """
    blocks = ("<|start|>assistant" + raw).split("<|start|>assistant")
    out: list[str] = []
    for block in blocks:
        if not block:
            continue
        m = _CHANNEL.match(block)
        if not m:
            continue
        recipient = (m.group("to") or "user").strip()
        if recipient == "user":
            out.append(m.group("body"))
    text = "".join(out).strip()

    if not text:
        # No channel markers at all. Some quantisations and sampler settings can
        # emit a bare completion; falling back to the raw text with control
        # tokens stripped is better than returning nothing.
        text = re.sub(r"<\|[^|]*\|>", "", raw).strip()
    return text


#: Stop strings for the sampler. <|eot|> ends the turn; <|eom|> only ends a
#: block, so it must NOT be a stop or reasoning would truncate the answer.
STOP = [EOT]
