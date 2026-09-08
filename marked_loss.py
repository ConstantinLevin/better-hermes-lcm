"""fork: betterlcm — every place the fork still drops or cuts text leaves a marker.

Rule (docs/fork-design.md): a summary is acceptable only while its provenance is intact AND
the visible text still hints at what was cut. These helpers build those hints so the
upstream call sites stay one-liners. Every marker is prefixed ``[LCM`` so ``lcm_doctor`` and
a reader can find them, and the message-body marker keeps upstream's literal
``...[truncated]...`` because tests and log greps look for it.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# Upstream's head/tail split for a 3000-char cap was 2000 + 800; keep those ratios so the
# 256k anchor reproduces upstream's serialisation byte for byte apart from the marker.
_HEAD_RATIO = 2000 / 3000
_TAIL_RATIO = 800 / 3000
_ARGS_KEEP_RATIO = 400 / 500

TRUNCATED_LITERAL = "...[truncated]..."
BYPASS_TRIM_SUFFIX = "…[LCM bypass trim: text cut to fit the cap; full text in the host transcript]"
BYPASS_FINAL_TRIM_SUFFIX = "…[LCM cut]"
ROTATE_MARKER_PREFIX = "[LCM rotate marker]"

_WS_RE = re.compile(r"\s+")


def elide_text(text: str, cap: int) -> str:
    """Keep head + tail of ``text`` under ``cap`` chars with a marked, sized elision."""
    if cap <= 0 or len(text) <= cap:
        return text
    head = max(1, round(cap * _HEAD_RATIO))
    tail = max(0, round(cap * _TAIL_RATIO))
    removed = len(text) - head - tail
    marker = (
        f"\n{TRUNCATED_LITERAL} [LCM elided {removed} of {len(text)} chars before summarising; "
        "the full message is in the raw store — lcm_expand this node]\n"
    )
    return text[:head] + marker + (text[-tail:] if tail else "")


def elide_args(args: str, cap: int) -> str:
    """Tool-call arguments: keep the head, say how much is missing."""
    if cap <= 0 or len(args) <= cap:
        return args
    keep = max(1, round(cap * _ARGS_KEEP_RATIO))
    return args[:keep] + f"...[LCM elided {len(args) - keep} of {len(args)} chars of arguments]"


def content_head(text: str, limit: int = 240) -> str:
    """A single-line, bracket-free preview of ``text`` for placeholders and stubs."""
    flat = _WS_RE.sub(" ", str(text or "")).strip()
    flat = flat.replace("[", "(").replace("]", ")").replace(";", ",")
    if len(flat) <= limit:
        return flat
    return flat[: max(1, limit - 1)].rstrip() + "…"


def externalized_head_note(text: str) -> str:
    """Appended after an externalized placeholder in summariser input and active stubs."""
    head = content_head(text)
    if not head:
        return ""
    return f"\n[LCM head of externalized output: {head}]"


def unmatched_tool_call_note() -> str:
    return "[no tool result in this chunk]"


def rotate_marker_summary(
    *,
    session_id: str,
    store_ids: Sequence[int],
    message_count: int,
    token_count: int,
    roles: Iterable[str],
    first_head: str,
    last_head: str,
) -> str:
    """Summary text of the d0 node written when ``rotate`` skips un-summarised raw rows."""
    lo = min(store_ids) if store_ids else 0
    hi = max(store_ids) if store_ids else 0
    role_counts: dict[str, int] = {}
    for role in roles:
        role_counts[role] = role_counts.get(role, 0) + 1
    roles_text = ", ".join(f"{n}×{r}" for r, n in sorted(role_counts.items())) or "?"
    return (
        f"{ROTATE_MARKER_PREFIX} {message_count} raw messages ({roles_text}; {token_count} tokens; "
        f"store ids {lo}..{hi}; session {session_id}) were rotated out of the active context "
        "WITHOUT being summarised. Nothing here is summarised: expand this node to read them.\n"
        f"First: {first_head}\nLast: {last_head}\n"
        f"Expand for details about: raw messages {lo}..{hi} rotated without a summary"
    )


def assembly_omission_marker(
    *,
    omitted_node_ids: List[int],
    depth_cap_hits: List[int],
    omitted_tail_messages: int,
) -> str:
    """One prefix part naming what the assembly budget/caps left out of this turn's context."""
    lines = ["[LCM assembly omissions — nothing below is deleted; it is just not rendered this turn]"]
    if omitted_node_ids:
        shown = ", ".join(str(n) for n in omitted_node_ids[:40])
        more = f" (+{len(omitted_node_ids) - 40} more)" if len(omitted_node_ids) > 40 else ""
        lines.append(
            f"- {len(omitted_node_ids)} summary node(s) did not fit the assembly budget: "
            f"lcm_expand(node_id=…) for {shown}{more}"
        )
    for depth in depth_cap_hits:
        lines.append(
            f"- more d{depth} summaries exist than assembly_max_nodes_per_depth renders; "
            "lcm_status / lcm_inspect to list them"
        )
    if omitted_tail_messages:
        lines.append(
            f"- {omitted_tail_messages} large fresh-tail message(s) were skipped by the assembly cap; "
            "they remain in the raw store (lcm_recent / lcm_expand)"
        )
    return "\n".join(lines)
