"""fork: betterlcm — every place the fork still drops or cuts text leaves a marker.

Rule (docs/fork-design.md): a summary is acceptable only while its provenance is intact AND
the visible text still hints at what was cut. These helpers build those hints so the
upstream call sites stay one-liners. Every marker is prefixed ``[LCM`` so ``lcm_doctor`` and
a reader can find them, and the message-body marker keeps upstream's literal
``...[truncated]...`` because tests and log greps look for it.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

# Upstream's head/tail split for a 3000-char cap was 2000 + 800; keep those ratios so the
# 256k anchor reproduces upstream's serialisation byte for byte apart from the marker.
_HEAD_RATIO = 2000 / 3000
_TAIL_RATIO = 800 / 3000
_ARGS_KEEP_RATIO = 400 / 500

TRUNCATED_LITERAL = "...[truncated]..."
BYPASS_TRIM_SUFFIX = "…[LCM bypass trim: text cut to fit the cap; full text in the host transcript]"
BYPASS_FINAL_TRIM_SUFFIX = "…[LCM cut]"
ROTATE_MARKER_PREFIX = "[LCM rotate marker]"
# fork: betterlcm — the receipt for a bypassed session's dropped messages. It is identified by
# this prefix so the cap-trimming loop can refuse to remove or shorten the one message that
# says something was removed (audit p05 BY01).
BYPASS_OMISSION_PREFIX = "[Context omitted:"


def bypass_omission_marker(dropped_messages: int, dropped_chars: int) -> str:
    """Name what the deterministic bypass trim dropped from a session LCM does not store."""
    return (
        f"{BYPASS_OMISSION_PREFIX} this session is ignored/stateless for LCM, and Hermes native "
        f"compression was unavailable. {dropped_messages} older message(s) (~{dropped_chars} "
        "chars) were dropped here to keep the request inside the model context window; they "
        "are not stored by LCM and remain only in the host transcript.]"
    )


_BYPASS_OMISSION_COUNTS_RE = re.compile(r"(\d+) older message\(s\) \(~(\d+) chars\)")
_BYPASS_COMPACT_COUNTS_RE = re.compile(r"(\d+) msg / (\d+) chars dropped")


def bypass_omission_counts(text: str) -> tuple[int, int]:
    """The (messages, chars) a receipt records, in either of its two forms.

    fork: betterlcm — reading both forms makes compaction IDEMPOTENT: compacting an already
    compact receipt used to turn the counted sentence into an uncounted one (verify-4 #17).
    """
    value = str(text or "")
    for pattern in (_BYPASS_OMISSION_COUNTS_RE, _BYPASS_COMPACT_COUNTS_RE):
        match = pattern.search(value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 0, 0


def compact_bypass_omission_marker(text: str) -> str:
    """The shortest honest form of the receipt, for a cap nothing else can satisfy.

    fork: betterlcm — the receipt is never removed, but when the budget cannot hold it AND the
    live request, the counts are what must survive, not the sentence around them.
    """
    messages, chars = bypass_omission_counts(text)
    if not messages and not chars:
        return f"{BYPASS_OMISSION_PREFIX} older messages dropped by the LCM bypass trim]"
    return (
        f"{BYPASS_OMISSION_PREFIX} {messages} msg / {chars} chars dropped, "
        "host transcript only]"
    )


def is_bypass_omission_marker(message: Any) -> bool:
    """fork: betterlcm — is this the receipt above?"""
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    return isinstance(content, str) and content.lstrip().startswith(BYPASS_OMISSION_PREFIX)

_WS_RE = re.compile(r"\s+")


def elide_text(text: str, cap: int, *, original_chars: int | None = None) -> str:
    """Keep head + tail of ``text`` under ``cap`` chars with a marked, sized elision.

    fork: betterlcm — an elision must not swallow an EARLIER receipt. Sanitisation runs first
    and leaves its own "[LCM: N chars of injected context removed]" line in the middle of the
    text; the elision then cut that line out and reported only the characters IT removed, so a
    20,031-character message became a 6,065-character one whose accounting said 5,785
    (round-2 verify-4 #13). Marker lines from the elided middle are carried through, and
    ``original_chars`` (the size before sanitisation) is reported when it is known.
    """
    if cap <= 0 or len(text) <= cap:
        return text
    head = max(1, round(cap * _HEAD_RATIO))
    tail = max(0, round(cap * _TAIL_RATIO))
    removed = len(text) - head - tail
    kept_head = text[:head]
    kept_tail = text[-tail:] if tail else ""
    before_sanitising = ""
    if original_chars is not None and int(original_chars) > len(text):
        before_sanitising = (
            f"; this message held {int(original_chars)} chars before the removals recorded below"
        )
    marker = (
        f"\n{TRUNCATED_LITERAL} [LCM elided {removed} of {len(text)} chars before summarising"
        f"{before_sanitising}; the full message is in the raw store — lcm_expand this node]\n"
    )
    carried: List[str] = []
    for fragment in marker_fragments(text[head:len(text) - tail]):
        if fragment in carried or fragment in kept_head or fragment in kept_tail:
            continue
        carried.append(fragment)
    if carried:
        shown = carried[:20]
        more = f"\n[LCM: +{len(carried) - 20} further receipt(s) in the elided span]" if len(carried) > 20 else ""
        marker += "\n".join(shown) + more + "\n"
    return kept_head + marker + kept_tail


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


COMPACT_ASSEMBLY_OMISSION_PREFIX = "[LCM assembly omissions — not rendered this turn: "

ASSEMBLY_OMISSION_MARKER_HEADER = (
    "[LCM assembly omissions — nothing below is deleted; it is just not rendered this turn]"
)


def missing_tool_result_stub(tool_call_id: str, *, store_ids: Sequence[int] | None = None,
                             archived: bool | None = None,
                             elsewhere_in_window: bool = False) -> str:
    """Stand in for a tool result the replay window does not contain.

    fork: betterlcm — upstream's stub said the result was "in the context summary above", which
    was a claim about a summary nobody had verified: with an empty DAG the reader was sent to
    something that did not exist (verify-4 #10). Naming the raw store unconditionally was the
    same mistake one level down: a call that NEVER received a result was described as archived
    and findable (round-2 verify-4 #19). ``archived`` distinguishes "stored, here is where"
    from "never received"; ``None`` means the store could not be consulted.
    """
    call = tool_call_id or "?"
    if elsewhere_in_window and not store_ids:
        return (
            "[LCM: this call's result cannot be replayed at this position; the result for it "
            f"appears elsewhere in this window and is quoted in the receipt below "
            f"(tool_call_id={call})]"
        )
    if archived is False:
        return (
            "[LCM: this call's result is not in the replayed window, and no result for it is "
            f"in the raw store either — none was ever received (tool_call_id={call})]"
        )
    if store_ids:
        ids = ", ".join(str(store_id) for store_id in list(store_ids)[:10])
        return (
            "[LCM: this call's result is not in the replayed window; it is archived — "
            f"lcm_expand(store_id={ids}) (tool_call_id={call})]"
        )
    return (
        "[LCM: this call's result is not in the replayed window, and whether it is archived "
        f"could not be checked here — try lcm_grep or lcm_expand for tool_call_id={call}]"
    )


def orphan_tool_results_marker(results: List[dict]) -> str:
    """Name tool results that answered no call in the replayed window.

    They cannot be replayed as ``tool`` messages without their call — the provider contract
    forbids it — but they are real content, and dropping them silently is exactly the loss
    this fork removes (verify-4 #10).
    """
    parts = []
    for result in results[:10]:
        call_id = str(result.get("tool_call_id") or "?")
        head = content_head(str(result.get("content") or ""), limit=120)
        parts.append(f"tool_call_id={call_id}: {head}")
    more = f" (+{len(results) - 10} more)" if len(results) > 10 else ""
    return (
        f"[LCM: {len(results)} tool result(s) answered no call in this replay window and are "
        f"not replayed here; they are stored — lcm_grep / lcm_expand. {'; '.join(parts)}{more}]"
    )


def injected_context_marker(removed_chars: int) -> str:
    """Name a span of host-injected context removed from the summariser's input.

    The block is dropped so recalled/injected text cannot steer the summariser, but the raw
    message is stored unchanged: this marker says the removal happened and how much it was,
    so the summary can never present a shortened message as the whole one.
    """
    # NB: neither angle brackets nor the tag name — the marker is re-scanned by the stripper
    # it comes from, and naming the tag would put the injected envelope's own vocabulary back
    # into the summariser's input.
    return f"[LCM: {removed_chars} chars of injected context removed before summarising]"


def unmappable_rows_marker(count: int) -> str:
    """Name consumed messages that have no durable store row of their own.

    fork: betterlcm — a host truncation marker whose file could not be copied has no row the
    archive can point at (the host owns the file and deletes it). The leaf still summarises the
    span, so the summary says plainly that those messages cannot be expanded, rather than
    implying the whole span is recoverable (round-2 verify-2 #2).
    """
    return (
        f"[LCM: {count} message(s) in this span are host truncation markers with no durable "
        "copy — they are summarised above but cannot be expanded; the host's own file has "
        "expired or was never copied]"
    )


def excluded_reply_marker(store_ids: List[int]) -> str:
    """Name rows a leaf consumed but deliberately kept out of the summariser input.

    Replies to host-injected placeholders are noise for a summary and content for the
    archive, so they are published as sources of the node and named here — never dropped
    silently, never mistaken for something the summary covers.
    """
    shown = ", ".join(str(store_id) for store_id in store_ids[:20])
    more = f" (+{len(store_ids) - 20} more)" if len(store_ids) > 20 else ""
    return (
        f"[LCM: {len(store_ids)} repl(y/ies) to ignored host-injected message(s) are sources of "
        f"this node but are NOT summarised above; read them with lcm_expand — store ids {shown}{more}]"
    )


RECEIPT_LINE_PREFIX = "[LCM:"


# fork: betterlcm — EVERY marker spelling this module writes, not only "[LCM:". Inheritance
# recognised the colon form alone, so a child carrying a rotate marker ("[LCM rotate marker] …
# nothing here is summarised") condensed into a parent with no warning at all: the reader was
# told the parent covered material that was never summarised (round-2 verify-4 #18).
_MARKER_LINE_PREFIXES = (
    RECEIPT_LINE_PREFIX,
    ROTATE_MARKER_PREFIX,
    COMPACT_ASSEMBLY_OMISSION_PREFIX,
    ASSEMBLY_OMISSION_MARKER_HEADER,
    BYPASS_OMISSION_PREFIX,
)


# A marker is often INLINE, not on a line of its own: sanitisation replaces an injected block
# in the middle of a sentence. Preserving whole lines alone therefore missed exactly the
# receipts that mattered (round-2 verify-4 #13).
# The receipt is the BRACKETED marker, not the rest of the line: a marker followed by 200,000
# ordinary characters restored the whole elided body when the fragment ran to the end of the
# line (round-4 verify-2 #7). Markers never contain a closing bracket of their own.
_MARKER_FRAGMENT_RE = re.compile(r"\[(?:LCM|Context omitted:)[^\]\n]*\]?")
# A receipt this module writes is far shorter than this; the bound only stops a pathological
# unterminated marker from carrying a whole message with it.
_MARKER_FRAGMENT_MAX_CHARS = 2_000


def marker_fragments(text: str) -> List[str]:
    """Every marker this module wrote that occurs in ``text``, in order, de-duplicated.

    fork: betterlcm — the fragment is not shortened below its closing bracket. A 300-character
    cut removed the recovery clause from a long receipt (round-3 verify-4 #10); running to the
    end of the line restored the surrounding body (round-4 verify-2 #7). The marker itself is
    what travels.
    """
    found: List[str] = []
    for match in _MARKER_FRAGMENT_RE.finditer(str(text or "")):
        fragment = match.group(0).strip()[:_MARKER_FRAGMENT_MAX_CHARS]
        if fragment and fragment not in found:
            found.append(fragment)
    return found


def is_marker_line(line: str) -> bool:
    """True for a line this module wrote to record a removal or an unsummarised span."""
    stripped = str(line or "").strip()
    return any(stripped.startswith(prefix) for prefix in _MARKER_LINE_PREFIXES)


def inherited_receipts(summaries: Iterable[str]) -> List[str]:
    """Every marker line found in these summaries, de-duplicated in order.

    fork: betterlcm — a condensed parent is written by the summariser, which has no obligation
    to reproduce a receipt its sources carried. Merging them into the parent keeps the record
    of what was excluded attached to the node that now stands for it (verify-4 #8), and every
    spelling counts, not only the ``[LCM:`` one (round-2 verify-4 #18).
    """
    seen: List[str] = []
    for summary in summaries:
        for line in str(summary or "").splitlines():
            stripped = line.strip()
            if is_marker_line(stripped):
                if stripped not in seen:
                    seen.append(stripped)
                continue
            # fork: betterlcm — a receipt can sit AFTER visible text on the same line (a
            # sanitiser replaces an injected block mid-sentence). Whole-line matching ignored
            # those, so the loss they record ended at the condensation boundary after all
            # (round-3 verify-4 #10).
            for fragment in marker_fragments(stripped):
                if fragment not in seen:
                    seen.append(fragment)
    return seen


MINIMAL_ASSEMBLY_OMISSION_MARKER = "[LCM: content omitted from this prefix — lcm_status]"


def minimal_assembly_omission_marker() -> str:
    """The smallest receipt that still says something is missing.

    fork: betterlcm — when neither the full nor the one-line receipt fits the summary budget,
    the receipt used to be dropped and recorded only in status: the prefix then omitted content
    silently, which is precisely the failure the receipt exists to prevent (round-2 verify-4
    #17). This form is ~12 tokens and is emitted even when it puts the prefix marginally over
    its budget; the counts and ids stay in ``lcm_status``.
    """
    return MINIMAL_ASSEMBLY_OMISSION_MARKER


LEADING_TURNS_DROPPED_PREFIX = "[LCM: leading turn(s) not replayable at the start of a request"


def leading_turns_dropped_marker(dropped: int, roles: List[str]) -> str:
    """Name assistant/tool turns dropped because a request cannot START with them.

    fork: betterlcm — a conversation replayed to a provider may not open with an assistant or
    tool message, so the leading ones were dropped silently; an assistant turn holding a
    decision simply vanished from the agent's own view of its history (round-3 verify-4 #7).
    The rows are untouched in the store.
    """
    role_text = ", ".join(roles[:8]) or "?"
    return (
        f"{LEADING_TURNS_DROPPED_PREFIX}: {dropped} turn(s) ({role_text}) were removed from "
        "the start of this replay because a provider request cannot begin with them; the "
        "stored rows are unchanged — lcm_recent / lcm_expand]"
    )


# Envelope fields that CHANGE what a turn means and are cheap to render inline.
_INLINE_ENVELOPE_FIELDS = ("name", "is_error", "status", "error_code", "finish_reason")


def envelope_summary_suffix(envelope: dict, *, max_listed: int = 10) -> str:
    """fork: betterlcm — render the small outcome fields, inventory the rest.

    The summariser was given ``[ASSISTANT]: Visible`` for a turn whose envelope also carried
    ``reasoning_content="DECISION cancel"`` and ``is_error=True``: a failed step read exactly
    like a successful one and a decision reached the summariser nowhere (round-3 verify-4 #8).
    Small fields are rendered; larger ones are named with their size and left in the store.
    """
    if not isinstance(envelope, dict) or not envelope:
        return ""
    inline: List[str] = []
    listed: List[str] = []
    for key, value in envelope.items():
        if not isinstance(key, str) or key.startswith("lcm_") or value in (None, "", [], {}):
            continue
        if key in _INLINE_ENVELOPE_FIELDS and not isinstance(value, (dict, list)):
            inline.append(f"{key}={value}")
            continue
        try:
            size = len(value) if isinstance(value, (str, list, dict)) else len(str(value))
        except Exception:  # pragma: no cover - defensive
            size = 0
        listed.append(f"{key} ({size} chars)")
    parts = []
    if inline:
        parts.append(" [" + ", ".join(inline[:max_listed]) + "]")
    if listed:
        shown = ", ".join(sorted(listed)[:max_listed])
        more = f" (+{len(listed) - max_listed} more)" if len(listed) > max_listed else ""
        parts.append(
            f"\n{RECEIPT_LINE_PREFIX} envelope field(s) not summarised here: {shown}{more}; "
            "the stored message holds them — lcm_expand]"
        )
    return "".join(parts)


def revision_rows_marker(store_ids: List[int]) -> str:
    """Name archived corrections that supersede rows this node covers.

    fork: betterlcm — a correction the host made to an already-stored message is archived as
    its own row. The leaf covering the original covers the correction too, so the newer text is
    never reachable from no summary, and this line says the newer version exists
    (round-3 verify-2 #8).
    """
    ids = ", ".join(str(store_id) for store_id in store_ids[:40])
    more = f" (+{len(store_ids) - 40} more)" if len(store_ids) > 40 else ""
    return (
        f"{RECEIPT_LINE_PREFIX} {len(store_ids)} of the message(s) summarised here were later "
        f"CORRECTED by the host; the newer version(s) are sources of this node and are not in "
        f"the text above — lcm_expand(store_id={ids}{more})]"
    )


def aggregated_inherited_receipt_marker(receipts: List[str], node_ids: List[int]) -> str:
    """One line standing for receipts whose verbatim copies would defeat condensation.

    fork: betterlcm — a condensed parent inherits its children's receipts verbatim, and four
    children carrying distinct receipts made the "condensed" parent LARGER than its sources
    (round-2 verify-2 #9), which raises the very pressure condensation exists to reduce. The
    children are still stored, still reachable from this node's ``source_ids``, and still carry
    the full receipts, so the aggregate names how many there are and where to read them. It is
    itself a receipt line, so a further condensation inherits it in turn.
    """
    ids = ", ".join(str(node_id) for node_id in node_ids)
    return (
        f"{RECEIPT_LINE_PREFIX} {len(receipts)} loss receipt(s) from the source summaries are "
        f"kept verbatim on node(s) {ids} — lcm_expand(node_id=...) to read them]"
    )


def recovered_body_rows_marker(store_ids: List[int]) -> str:
    """Name the archive rows holding bytes a host truncation marker stands for.

    fork: betterlcm — when the durable copy of a recovered host output cannot be written, the
    bytes are stored as an extra archive row next to the marker row. Those rows belonged to no
    node, so the leaf covering the marker left the real content outside the graph and read as
    if expansion were impossible (round-2 verify-4 #1). They are sources of the leaf now, and
    this line says where the bytes are.
    """
    ids = ", ".join(str(store_id) for store_id in store_ids[:40])
    more = f" (+{len(store_ids) - 40} more)" if len(store_ids) > 40 else ""
    return (
        f"{RECEIPT_LINE_PREFIX} {len(store_ids)} recovered host-output archive row(s) are "
        f"sources of this node and are NOT summarised above — the complete bytes are in the "
        f"store: lcm_expand(store_id={ids}{more})]"
    )


def compact_assembly_omission_marker(
    *,
    omitted_node_ids: List[int],
    depth_cap_hits: List[int],
    omitted_tail_messages: int,
    dropped_internal_turns: int = 0,
    redacted_internal_turns: int = 0,
) -> str:
    """The one-line form, for a summary budget that cannot hold the full receipt.

    fork: betterlcm — the receipt is what tells the reader something is missing, so it must
    survive a budget that the full sentence does not fit into (verify-4 #9). The counts are
    what matter; ``lcm_status`` and the log carry the ids.
    """
    counts = []
    if omitted_node_ids:
        counts.append(f"{len(omitted_node_ids)} summary node(s)")
    if depth_cap_hits:
        counts.append(f"{len(depth_cap_hits)} depth cap(s)")
    if omitted_tail_messages:
        counts.append(f"{omitted_tail_messages} tail message(s)")
    if dropped_internal_turns:
        counts.append(f"{dropped_internal_turns} internal-only turn(s)")
    if redacted_internal_turns:
        counts.append(f"{redacted_internal_turns} turn(s) with internal content removed")
    if not counts:
        return ""
    return (
        COMPACT_ASSEMBLY_OMISSION_PREFIX
        + ", ".join(counts)
        + "; nothing is deleted — lcm_status / lcm_expand]"
    )


def assembly_omission_marker(
    *,
    omitted_node_ids: List[int],
    depth_cap_hits: List[int],
    omitted_tail_messages: int,
    dropped_internal_turns: int = 0,
    redacted_internal_turns: int = 0,
) -> str:
    """One prefix part naming what the assembly budget/caps left out of this turn's context."""
    lines = [ASSEMBLY_OMISSION_MARKER_HEADER]
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
    if dropped_internal_turns:
        # fork: betterlcm — active-context cleanup removes assistant turns whose only content
        # was internal/reasoning material. Upstream logged that for the operator and left the
        # agent's own view of its history quietly one turn shorter (audit p05 SA01).
        lines.append(
            f"- {dropped_internal_turns} assistant turn(s) held only internal/reasoning content "
            "and are not replayed; the stored rows are unchanged (lcm_recent / lcm_expand)"
        )
    if redacted_internal_turns:
        # fork: betterlcm — a turn can be PARTLY internal: the visible text is replayed and the
        # reasoning block is not. Only whole dropped turns were counted, so a turn that lost a
        # decision written inside <think> left no trace at all (round-2 verify-4 #16).
        lines.append(
            f"- {redacted_internal_turns} replayed assistant turn(s) had internal/reasoning "
            "content removed from the replay; the stored rows are unchanged "
            "(lcm_recent / lcm_expand)"
        )
    return "\n".join(lines)
