#!/usr/bin/env python3
"""Can a reader get from the frontier the engine ACTUALLY delivered back to the right originals?

`scripts/e2e_no_loss.py` proves the structural half: nothing becomes unreachable. It cannot
prove this half, because it replaces the model with a fact-id copier — so nothing in this
repository has ever asked a reader to *choose* a node out of the prefix it was handed and then
checked whether the right original lines came back. That gap is issue #2. Issue #8 is the other
half of the same run: the summaries the reader navigates by must still say what the sources
said, and "was started" must not have become "succeeded".

Two scorers, two results, never one rate (benchmarking/nav_fidelity.py). A reader can navigate
perfectly to a summary that lies; a faithful sentence can be unfindable. Merging them hides
both.

    scripts/e2e_index_navigation.py 262144 400
    scripts/e2e_index_navigation.py 1000000 3000 --summariser real --reader model

Like its sibling it runs THIS CHECKOUT with nothing installed (LCM_E2E_PLUGIN_DIR points it
elsewhere), and it reuses that script's plugin bootstrap and DAG walk rather than forking them.

What this can and cannot establish
----------------------------------
`--summariser stub` / `--reader lexical` exercise the whole chain deterministically and prove
the plumbing, the depth and the scorers' wiring. They are NOT a measurement of any model: the
output says so on every line that reports a score. Only `--summariser real --reader model`
measures a model, it measures the route this environment is configured for and no other, and a
single labelled corpus is a probe, not an error rate. `--summariser real` never falls back to
a stub: if no route is configured the run stops and says so, because a failed operation that
reads as success is the defect this fork exists to remove.

Defects that belong to other issues are labelled as such rather than booked as model failure:
a source reachable from no delivered node, a recovery that came back truncated or paged out, a
statement whose distinguishing text never reached the summariser at all. Those are evidence
defects; they are counted, named, and kept out of the denominators.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Reuse the sibling's plugin bootstrap (it loads THIS checkout as `hermes_lcm` and puts the
# host's agent modules on the path) instead of copying it. Its driver is guarded by
# __main__, so importing it runs the bootstrap and nothing else — e2e_no_loss.py is the
# release gate and is not modified by this file.
_spec = importlib.util.spec_from_file_location(
    "lcm_e2e_no_loss", os.path.join(_HERE, "e2e_no_loss.py")
)
e2e_no_loss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e_no_loss)

from hermes_lcm import coverage_doctor, escalation, tokens as lcm_tokens  # noqa: E402
from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402

# APPENDED, never inserted. The plugin root holds a top-level ``tools.py`` and the host holds a
# top-level ``tools`` package; putting the plugin root first makes the host's own
# ``from tools.hook_output_spill import ...`` resolve to the plugin's module, which then dies on
# its relative import. That silently cost this script the host's normalize_tool_schema, and it
# would shadow the host for any other module that imports ``tools`` mid-run.
sys.path.append(_ROOT)
from benchmarking.nav_fidelity import (  # noqa: E402
    BOUNDED_OBSERVATIONS,
    CorpusPatternError,
    attribute_recovery_defects,
    recovery_causes,
    NavigationCase,
    ReaderTrace,
    ScoredText,
    StateClaim,
    score_fidelity,
    score_navigation,
    validate_patterns,
)

# The real route, captured before anything can replace it. `--summariser real` calls THIS and
# nothing else; no branch of this file can substitute a stub for it.
_REAL_SUMMARY_ROUTE = escalation._call_llm_for_summary

DEFAULT_CORPUS = os.path.join(_ROOT, "tests", "fixtures", "nav_fidelity_corpus.json")

# The frontier as the model sees it: engine.py renders every delivered node with its id.
_FRONTIER_BLOCK_RE = re.compile(
    r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d(\d+), node (\d+)\)\]"
)
# Every marker this fork leaves where something was cut is prefixed "[LCM"; host spills are
# "[Externalized". Either one inside a recovery means the reader did not get the whole row.
_LOSS_MARKER_RE = re.compile(r"\[LCM\b|\[Externalized\b")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}|\d{2,}")
_STOPWORDS = frozenset("""
about after all and any are our was were what which who whom why the that this these those
for from had has have how into its was were will with your you can did does done not but
""".split())


class RouteUnavailable(RuntimeError):
    """No model route is configured here. Never downgraded to a stub."""


# ── environment pinning ─────────────────────────────────────────────────────────────────────

def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", "-C", _ROOT, *args), capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def pin_tokenizer() -> dict[str, Any]:
    """Settle the token counter BEFORE the run and report what settled.

    tokens.py loads tiktoken on a daemon thread and adopts it mid-run when it arrives
    (tokens.py:46/:67). A 13-nodes-versus-26-nodes discrepancy in this repository was exactly
    that: the first 1,353 rows counted with the estimator and the rest with cl100k. A run whose
    tokenizer state is unlogged is not evidence, so this waits for the loader to finish and
    records which counter the whole run then used.
    """
    lcm_tokens._get_encoder()
    thread = getattr(lcm_tokens, "_encoder_thread", None)
    if thread is not None:
        thread.join(timeout=180)
    encoder = getattr(lcm_tokens, "_encoder", None)
    version = ""
    if encoder is not None:
        try:
            import tiktoken
            version = getattr(tiktoken, "__version__", "") or ""
        except Exception:
            version = ""
    return {
        "backend": "tiktoken/cl100k_base" if encoder is not None else "char-estimate-fallback",
        "tiktoken_version": version,
        # reports that the load ATTEMPT settled, not that an encoder exists — tokens.py sets
        # it either way, and this fork polices exactly that wording (minor 12)
        "loader_settled": bool(getattr(lcm_tokens, "_encoder_ready", False)),
        "encoder_present": encoder is not None,
        "generation": int(getattr(lcm_tokens, "_encoder_generation", 0)),
        "pinned_before_run": True,
    }


def environment_block(args, corpus_digest: str, input_digest: str,
                      tokenizer: dict[str, Any], engine: LCMEngine,
                      messages_total: int, tool_surface: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
    except Exception as exc:  # pragma: no cover - diagnostic only
        status = {"error": str(exc)}
    effective = {
        "context_length": engine.context_length,
        "threshold_tokens": engine.threshold_tokens,
        "condense_budget_tokens": engine.effective_condense_budget_tokens,
        "incremental_max_depth": engine.effective_incremental_max_depth,
        "condensation_fanin": engine._config.condensation_fanin,
        "leaf_chunk_tokens": engine.effective_leaf_chunk_tokens,
        "fresh_tail_count": engine.effective_fresh_tail_count,
        "summary_concurrency": engine.effective_summary_concurrency,
        "assembly_max_nodes_per_depth": engine._config.assembly_max_nodes_per_depth,
        "max_assembly_tokens": engine._config.max_assembly_tokens,
    }
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "window": args.window,
        "turns": args.turns,
        "summariser": args.summariser,
        "reader": args.reader,
        "python": sys.version.split()[0],
        "plugin_dir": os.environ.get("LCM_E2E_PLUGIN_DIR") or _ROOT,
        "plugin_commit": _git("rev-parse", "--short", "HEAD"),
        "plugin_tree": "dirty" if _git("status", "--porcelain") else "clean",
        "host_agent_path": os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes"))
                           + "/hermes-agent",
        "tokenizer": tokenizer,
        "corpus": {"path": args.corpus, "sha256": corpus_digest},
        "input_digest_sha256": input_digest,
        "input_messages": messages_total,
        "effective_config": effective,
        "window_scaling": status.get("window_scaling"),
        "tool_surface": tool_surface,
        "live_turn_ingest_on_tool_call": bool(args.dispatch_messages),
    }


# ── summariser routes ───────────────────────────────────────────────────────────────────────

class RouteRecorder:
    """Delegates to the real route and records every call. Never substitutes text."""

    def __init__(self, target: Callable[..., Optional[str]]):
        self._target = target
        self.calls = 0
        self.failures = 0
        self.errors: list[str] = []
        # _invoke_summary_llm inspects this signature to decide which optional kwargs to pass.
        self.__wrapped__ = target

    def __call__(self, prompt, max_tokens, **kwargs):
        self.calls += 1
        result = self._target(prompt, max_tokens, **kwargs)
        if not isinstance(result, str) or not result.strip():
            self.failures += 1
            error = getattr(getattr(escalation, "_LAST_ROUTE_ERROR", None), "error", None)
            self.errors.append(str(error) if error else "route returned no text")
        return result


def _source_text_from_prompt(prompt: Any) -> str:
    """The untrusted-data envelope's source content — what the acceptance rule counts."""
    if not isinstance(prompt, list):
        return str(prompt)
    for message in prompt:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or message.get("role") != "user":
            continue
        try:
            envelope = json.loads(content)
            parts = [source.get("content") for source in envelope.get("sources") or []
                     if isinstance(source, dict)]
            parts = [part for part in parts if isinstance(part, str)]
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
        return content
    return ""


def _extractive_index(source: str, max_tokens: int) -> str:
    """An honest stand-in: keep the most index-dense sentences, plus an index line.

    This is extractive, not generative, and it is strictly shorter than its source — it is
    NOT the stub padded out to cross a gate (CLAUDE.md: padding the stub only tests the stub;
    this run crosses the condensation gate through configuration instead). It preserves the
    wording of the densest sentences, so statement fidelity is testable end to end in a
    deterministic run without pretending a model produced it.
    """
    entities = coverage_doctor.extract_index_entities(source, limit=80)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", source) if part.strip()]
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: -sum(1 for entity in entities if entity.lower() in sentences[i].lower()),
    )
    budget = max(1, min(int(max_tokens or 0) // 2, max(1, len(source) // 8)))
    kept: list[int] = []
    used = 0
    for index in ranked:
        cost = len(sentences[index])
        if used + cost > budget * 3 and kept:
            break
        kept.append(index)
        used += cost
    body = " ".join(sentences[i] for i in sorted(kept))
    head = f"Index over {len(sentences)} source sentence(s): {', '.join(entities[:20])}."
    return f"{head}\n{body}\nExpand for details about: {', '.join(entities[:3]) or 'this span'}"


_UPGRADE_REWRITES = (
    (re.compile(r"was started at [^.]*\.", re.IGNORECASE), "completed successfully."),
    (re.compile(r"whether it completed is unknown", re.IGNORECASE), "it completed"),
    (re.compile(r"stopped at batch \d+ of \d+", re.IGNORECASE), "finished all batches"),
    (re.compile(r"is explicitly ruled out", re.IGNORECASE), "is enabled"),
    (re.compile(r"\bfailed with exit code \d+\b", re.IGNORECASE), "succeeded"),
    (re.compile(r"We plan to add", re.IGNORECASE), "We added"),
)


def _unfaithful_index(source: str, max_tokens: int) -> str:
    """Negative control: the extractive stub with every hedge upgraded to a result.

    Nothing here is a claim about any model. It exists so a run can demonstrate that the
    fidelity scorer catches the headline failure through the unmodified production acceptance
    path — `_is_index_shaped_summary` plus "smaller than the source" accepts this text, which
    is the whole point of #8.
    """
    text = _extractive_index(source, max_tokens)
    for pattern, replacement in _UPGRADE_REWRITES:
        text = pattern.sub(replacement, text)
    return text


def install_summariser(mode: str) -> RouteRecorder:
    if mode == "real":
        recorder = RouteRecorder(_REAL_SUMMARY_ROUTE)
        escalation._call_llm_for_summary = recorder
        return recorder

    if mode == "stub":
        generator = _extractive_index
    elif mode == "unfaithful-stub":
        generator = _unfaithful_index
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown summariser mode: {mode}")

    def _stub(prompt, max_tokens, model="", timeout=None, **_ignored):
        return generator(_source_text_from_prompt(prompt), max_tokens)

    recorder = RouteRecorder(_stub)
    escalation._call_llm_for_summary = recorder
    return recorder


def preflight_summary_route() -> str:
    """Prove the configured route answers, before a run claims to have used it."""
    probe = _REAL_SUMMARY_ROUTE(
        "Session note: the deploy was started and its outcome is not yet known.", 96
    )
    if not isinstance(probe, str) or not probe.strip():
        error = getattr(getattr(escalation, "_LAST_ROUTE_ERROR", None), "error", None)
        raise RouteUnavailable(
            "--summariser real was requested but the plugin's configured summariser route "
            f"returned nothing: {error or 'no error recorded'}. This run is stopping instead "
            "of falling back to the stub: a stubbed run reported as a real one is exactly the "
            "'a failed operation must not read as success' defect this fork exists to remove."
        )
    return probe.strip()[:120]


# ── the reader's model route (provider-neutral; the operator's configuration decides) ───────

def call_reader_model(messages: list[dict], tools: list[dict], args) -> tuple[Any, dict]:
    from agent.auxiliary_client import call_llm  # host-resolved route

    from hermes_lcm.model_routing import apply_lcm_model_route

    route_info: dict[str, str] = {}
    kwargs: dict[str, Any] = {
        "task": args.reader_task,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": args.reader_max_tokens,
        "route_info": route_info,
    }
    if tools:
        kwargs["tools"] = tools
    apply_lcm_model_route(kwargs, args.reader_model)
    return call_llm(**kwargs), route_info


def preflight_reader_route(args) -> dict:
    try:
        response, route_info = call_reader_model(
            [{"role": "user", "content": "Reply with the single word: ready."}], [], args
        )
        content = response.choices[0].message.content
    except Exception as exc:
        raise RouteUnavailable(
            f"--reader model was requested but no model route answered here: {exc}. "
            "Stopping rather than reporting a scripted reader's numbers as a model's."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise RouteUnavailable(
            "--reader model was requested but the configured route returned no content. "
            "Stopping rather than reporting a scripted reader's numbers as a model's."
        )
    return route_info


# ── the conversation ────────────────────────────────────────────────────────────────────────

_FILLER_TOPICS = (
    ("invoice-pdf", "the PDF renderer", "INV"),
    ("billing-webhooks", "the webhook dispatcher", "WBH"),
    ("search-indexer", "the search indexer", "SRCH"),
    ("session-store", "the session store", "SESS"),
    ("audit-trail", "the audit trail writer", "AUD"),
    ("rate-shaper", "the rate shaper", "RSH"),
)


def filler_turn(index: int, rng: random.Random) -> list[dict]:
    """Plausible engineering chatter that carries no corpus answer.

    It uses the same shapes as the anchors — ticket ids, file paths, timings, decisions — so
    lexical proximity alone does not point at an anchor, and every filler ticket id comes from
    the same namespace as the anchors'.
    """
    service, subject, prefix = _FILLER_TOPICS[index % len(_FILLER_TOPICS)]
    ticket = f"{prefix}-{4000 + (index * 7) % 5000}"
    if index % 5 == 3:
        call_id = f"call_nav_{index}"
        return [
            {"role": "user", "content": f"{ticket}: pull the last deploy log for {service}."},
            {"role": "assistant", "content": f"Reading the {service} deploy log for {ticket}.",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": json.dumps(
                                              {"path": f"/var/log/{service}/deploy-{index}.log"})}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": (f"{ticket} deploy log for {service}\n"
                         + "\n".join(f"  step {n}: ok ({rng.randint(10, 900)} ms)"
                                     for n in range(40)))},
            {"role": "assistant",
             "content": (f"{ticket}: the {service} deploy log is clean - {subject} rolled out "
                         f"without a retry. Nothing here touches the ledger path.")},
        ]
    return [
        {"role": "user",
         "content": f"{ticket}: is {subject} still queueing behind the {index % 7} shard?"},
        {"role": "assistant",
         "content": (f"{ticket}: {subject} drained in {rng.randint(2, 90)} s and the queue is "
                     f"flat. We keep the current setting for {service}; the alternative was "
                     f"measured and made no difference. " + ("Routine detail. " * 60))},
    ]


def build_conversation(corpus: dict, turns: int, seed: int) -> tuple[list[list[dict]], dict[int, str]]:
    """Return (turn list, turn index -> anchor id)."""
    rng = random.Random(seed)
    placement: dict[int, str] = {}
    for anchor in corpus["anchors"]:
        slot = max(1, min(turns - 1, int(round(float(anchor["position"]) * turns))))
        while slot in placement:
            slot += 1
        placement[slot] = anchor["anchor_id"]
    by_id = {anchor["anchor_id"]: anchor for anchor in corpus["anchors"]}
    built: list[list[dict]] = []
    for index in range(turns):
        anchor_id = placement.get(index)
        if anchor_id:
            built.append([dict(message) for message in by_id[anchor_id]["messages"]])
        else:
            built.append(filler_turn(index, rng))
    return built, placement


# ── the DAG side of the delivered frontier ──────────────────────────────────────────────────

def delivered_frontier(context: list[dict]) -> list[tuple[int, int]]:
    """(depth, node_id) for every node rendered into the context the host would send."""
    text = "\n".join(
        value if isinstance(value := message.get("content"), str) else json.dumps(value, default=str)
        for message in context
    )
    return [(int(depth), int(node_id)) for depth, node_id in _FRONTIER_BLOCK_RE.findall(text)]


def descend(engine: LCMEngine, node_id: int) -> tuple[set[int], dict[int, set[int]]]:
    """Walk down from one node. Returns (visited node ids, store_id -> holding leaf ids).

    The holders matter, not the ancestors: once condensation has collapsed the frontier to a
    single node, "did the reader choose the right node from the frontier" has exactly one
    answer and measures nothing. The real task is the descent to the leaf that holds the line.
    """
    nodes = {node.node_id: node for node in engine._dag.get_session_nodes(engine._session_id)}
    visited: set[int] = set()
    holders: dict[int, set[int]] = {}
    stack = [int(node_id)]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        node = nodes.get(current)
        if node is None:
            continue
        if node.source_type == "messages":
            for value in node.source_ids:
                holders.setdefault(int(value), set()).add(current)
        else:
            stack.extend(int(value) for value in node.source_ids)
    return visited, holders


def store_ids_under(engine: LCMEngine, node_id: int) -> set[int]:
    """Every raw store_id reachable from one node, walking condensation edges down."""
    return set(descend(engine, node_id)[1])


# ── tool execution and evidence labelling ───────────────────────────────────────────────────

def run_tool(engine: LCMEngine, name: str, arguments: dict, messages: Optional[list],
             args) -> dict[str, Any]:
    """Execute through the engine's real dispatch, exactly as the host does."""
    kwargs = {}
    if args.dispatch_messages and messages is not None:
        kwargs["messages"] = messages
    try:
        raw = engine.handle_tool_call(name, arguments, **kwargs)
    except Exception as exc:
        return {"__tool_error__": f"{type(exc).__name__}: {exc}"}
    try:
        return json.loads(raw)
    except Exception:
        return {"__unparsed__": raw}


def _collect_text(value: Any, sink: list[str], store_ids: set[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("store_id", "source_id") and isinstance(item, int):
                store_ids.add(item)
            elif key in ("content", "summary", "text", "envelope", "envelope_raw",
                         "tool_calls", "expanded_content") and isinstance(item, str):
                sink.append(item)
            else:
                _collect_text(item, sink, store_ids)
    elif isinstance(value, list):
        for item in value:
            _collect_text(item, sink, store_ids)
    elif isinstance(value, str):
        sink.append(value)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _node_ids_in(value: Any, found: Optional[set[int]] = None) -> set[int]:
    """Every node id a tool result showed the reader — what it may navigate to next."""
    if found is None:
        found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("node_id", "parent_node_id") and _as_int(item) is not None:
                found.add(_as_int(item))
            else:
                _node_ids_in(item, found)
    elif isinstance(value, list):
        for item in value:
            _node_ids_in(item, found)
    return found


def _tool_message_content(result: dict[str, Any], args,
                          truncated_views: list[dict[str, Any]]) -> str:
    """The tool result as the reader sees it. Uncut by default.

    This used to be ``json.dumps(result)[:200_000]``: the reader's only view of the evidence
    silently shortened, with no marker and no counter, leaving broken JSON in its transcript —
    while scoring used the untruncated text, so a line cut out of the reader's view was
    charged to the reader as a miss. Cutting our own tooling's evidence without a marker is
    the defect this fork exists to remove, so the default now cuts nothing. An operator who
    sets a cap gets a valid JSON object that says what was withheld, by whom, and how to get
    it back — and the event is recorded in the run's output.
    """
    rendered = json.dumps(result)
    cap = int(getattr(args, "max_tool_result_chars", 0) or 0)
    if cap <= 0 or len(rendered) <= cap:
        return rendered
    note = (
        f"[HARNESS ELISION: this tool result was {len(rendered)} characters; the reader was "
        f"shown the first {cap} and {len(rendered) - cap} were withheld by the harness, not by "
        f"the tool. Re-run with --max-tool-result-chars 0 (the default) to see all of it.]"
    )
    truncated_views.append({"original_chars": len(rendered), "shown_chars": cap,
                            "withheld_chars": len(rendered) - cap})
    return json.dumps({"__harness_elision__": note, "result_head": rendered[:cap]})


def label_evidence(result: dict[str, Any], recovered_text: str) -> tuple[list[str], list[str]]:
    """Separate 'the evidence was defective' from 'the reader chose badly'.

    A reader can navigate correctly and still be handed truncated, paged-out or falsely
    complete evidence (#50/#51/#52), an unfinished generation (#32), or a frontier the assembly
    cap shaped (#5). None of those is a model failure and the harness must not score them as
    one.

    Returns ``(causes, observations)``. A cause carries the rows it is known to be about —
    ``None`` when the tool reported a failure without identifying which rows it cost, such as a
    corrupt payload or a child node no longer in the DAG. An observation only BOUNDS what came
    back — one page plus a cursor is lcm_expand's normal contract — and it withdraws a line only
    where it explains that line's absence.
    """
    observations: list[str] = []
    tool_failed = bool(result.get("__tool_error__") or result.get("error"))
    identified: list[int] = []
    unidentified = False
    declared_incomplete = False
    pagination = result.get("pagination")
    for holder in (result, pagination if isinstance(pagination, dict) else {}):
        # Rows the tool NAMED: the narrowest attribution available, and the one that keeps a
        # failure from withdrawing lines it never touched.
        for key in ("missing_source_store_ids", "unreadable_source_ids", "missing_source_ids",
                    "unresolved_source_ids"):
            for value in holder.get(key) or ():
                parsed = _as_int(value)
                if parsed is not None:
                    identified.append(parsed)
        # Failures the tool reported WITHOUT naming the rows they cost. The node is then the
        # only honest scope; dropping them would charge the reader for what they removed.
        for key in ("missing_source_node_ids", "corrupt_payloads"):
            if holder.get(key):
                unidentified = True
    if isinstance(pagination, dict):
        if pagination.get("complete") is False:
            declared_incomplete = True
        if pagination.get("has_more") or pagination.get("remaining_sources"):
            observations.append("evidence:paged_result")
        if pagination.get("complete") is True and pagination.get("has_more"):
            # NOT a defect of this recovery: `complete` in tools.py:1746-1759 means "no source
            # was unreadable", while `has_more` is the paging cursor. Recorded because two
            # different questions answered in one dict under a name that reads as "this is the
            # whole node" is a reporting shape the no-loss doctrine warns about — a finding
            # about tools.py, which this harness does not own, not a reader result.
            observations.append("observed:complete_flag_beside_has_more")
    if result.get("has_more"):
        observations.append("evidence:paged_result")
    if _LOSS_MARKER_RE.search(recovered_text):
        observations.append("evidence:truncated_or_marked")
    causes = recovery_causes(tool_failed=tool_failed, identified_missing_rows=identified,
                             unidentified_failure=unidentified,
                             declared_incomplete=declared_incomplete)
    return causes, sorted(dict.fromkeys(observations))


# ── readers ─────────────────────────────────────────────────────────────────────────────────

def _terms(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text or "")
            if word.lower() not in _STOPWORDS]


class LexicalReader:
    """A deterministic weak reader: term overlap against the rendered frontier, nothing else.

    It never sees the corpus labels, the anchor ids or the store ids — only the delivered
    context and the question, which is the same input the model reader gets. It is a FLOOR and
    a plumbing check, never a model measurement, and every line of output that carries its
    numbers says so.
    """

    name = "lexical-baseline"

    def __init__(self, max_hops: int, max_pages: int):
        self.max_hops = max_hops
        self.max_pages = max_pages

    def read(self, engine, context, question, tool_schemas, args):
        blocks = self._blocks(context)
        chosen: list[int] = []
        results: list[dict] = []
        texts: list[str] = []
        store_ids: set[int] = set()
        defects: list[str] = []
        observations: list[str] = []
        bounded_nodes: dict[int, set[str]] = {}
        bounded_direct: dict[int, set[str]] = {}
        pages_followed = 0
        # One greedy descent, not a sweep of the DAG: take the best-matching rendered node,
        # then at each condensation take its best-matching child. Expanding everything would
        # buy a perfect recall for free, which is why node_precision travels beside it.
        queue = [node_id for _depth, node_id, _text
                 in sorted(blocks, key=lambda b: -self._score(question, b[2]))[:2]]
        hops = 0
        reached_a_leaf = False
        while queue and hops < self.max_hops:
            node_id = queue.pop(0)
            hops += 1
            chosen.append(node_id)
            descended = False
            # Follow the cursor the tool hands back. lcm_expand returns one page plus
            # next_source_offset; stopping at page one and calling the remainder lost would
            # score the tool's own contract as a navigation failure.
            pages = [run_tool(engine, "lcm_expand", {"node_id": node_id}, None, args)]
            for _page in range(self.max_pages):
                pagination = pages[-1].get("pagination")
                if not isinstance(pagination, dict) or not pagination.get("has_more"):
                    break
                offset = pagination.get("next_source_offset")
                if offset is None:
                    break
                pages.append(run_tool(engine, "lcm_expand",
                                      {"node_id": node_id, "source_offset": int(offset)},
                                      None, args))
                pages_followed += 1
            for page in pages:
                sink: list[str] = []
                _collect_text(page, sink, store_ids)
                piece = "\n".join(sink)
                texts.append(piece)
                results.append(page)
                found_causes, found_observations = label_evidence(page, piece)
                defects.extend(cause.label for cause in found_causes)
                # Each cause is attributed to the recovery it actually broke: the exact rows
                # the tool named, or failing that the rows this node holds. A trace-level
                # defect cannot say which line it broke, so it may not excuse any of them.
                per_row, node_wide = attribute_recovery_defects(found_causes)
                for store_id, labels in per_row.items():
                    bounded_direct.setdefault(store_id, set()).update(labels)
                for name in node_wide:
                    bounded_nodes.setdefault(node_id, set()).add(name)
                # Bounds are attributed to THIS NODE, not to the whole trace. A marker in one
                # page of one node used to bound every line of every node the reader touched.
                for name in found_observations:
                    if name == "evidence:paged_result":
                        continue  # re-derived from the last page below
                    observations.append(name)
                    if name in BOUNDED_OBSERVATIONS:
                        bounded_nodes.setdefault(node_id, set()).add(name)
            last = pages[-1].get("pagination")
            if isinstance(last, dict) and last.get("has_more"):
                # Still short after the cursor was followed to the hop limit: THIS is a bound
                # the reader was actually left with, on THIS node. An intermediate page saying
                # has_more is just the cursor doing its job.
                observations.append("evidence:paged_result")
                bounded_nodes.setdefault(node_id, set()).add("evidence:paged_result")
            if pages[0].get("source_type") == "nodes":
                children = [child for page in pages for child in (page.get("expanded") or [])
                            if isinstance(child, dict) and child.get("node_id") is not None]
                children.sort(key=lambda child: -self._score(question, str(child.get("summary") or "")))
                # Depth FIRST. Breadth-first interleaving between two frontier branches ran
                # out of hops before either descent reached a leaf on a depth-3 DAG, which
                # reads as "the reader found nothing" when it never got to look.
                queue[:0] = [int(child["node_id"]) for child in children[:1]]
                descended = True
            if not descended:
                reached_a_leaf = True
        recovered = "\n".join(texts)
        answer = self._answer(question, recovered)
        status = "ok" if recovered.strip() else "empty"
        return {
            "status": status,
            "detail": "" if status == "ok" else "no tool result carried any text",
            "chosen_node_ids": tuple(chosen),
            "recovered_store_ids": tuple(sorted(store_ids)),
            "recovered_text": recovered,
            "answer": answer,
            "evidence_defects": tuple(sorted(dict.fromkeys(defects))),
            "observations": tuple(sorted(dict.fromkeys(observations))),
            "bounded_nodes": {node: sorted(labels) for node, labels in bounded_nodes.items()},
            "bounded_store_ids_direct": {s: sorted(v) for s, v in bounded_direct.items()},
            "unsourced_node_ids": (),  # it only ever opens ids it read from the frontier
            "tool_calls": len(results),
            "pages_followed": pages_followed,
            # A descent that ran out of hops before it reached a leaf did not look and find
            # nothing; it never got to look. Reporting that as a navigation failure would be
            # the same lie as an empty result reading as "there is nothing".
            "budget_exhausted": bool(hops >= self.max_hops and not reached_a_leaf),
        }

    @staticmethod
    def _blocks(context: list[dict]) -> list[tuple[int, int, str]]:
        text = "\n".join(message.get("content") for message in context
                         if isinstance(message.get("content"), str))
        out: list[tuple[int, int, str]] = []
        matches = list(_FRONTIER_BLOCK_RE.finditer(text))
        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            out.append((int(match.group(1)), int(match.group(2)), text[match.start():end]))
        return out

    @staticmethod
    def _score(question: str, body: str) -> float:
        wanted = set(_terms(question))
        if not wanted:
            return 0.0
        have = set(_terms(body))
        return len(wanted & have) / len(wanted)

    @staticmethod
    def _answer(question: str, recovered: str) -> str:
        wanted = set(_terms(question))
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", recovered)
                     if part.strip()]
        ranked = sorted(sentences, key=lambda s: -len(wanted & set(_terms(s))))
        picked = [s for s in ranked[:6] if wanted & set(_terms(s))]
        return " ".join(picked)


class ModelReader:
    """The real thing: the delivered context plus the question, and the real tool surface."""

    name = "model"

    def __init__(self, args):
        self.args = args

    def read(self, engine, context, question, tool_schemas, args):
        # The reader gets ONLY the context compress() returned plus the question. No extra
        # system prompt, no hints, no node ids: whatever guidance exists is already inside
        # that context because the engine put it there.
        messages = [dict(message) for message in context]
        messages.append({"role": "user", "content": question})
        chosen: list[int] = []
        texts: list[str] = []
        store_ids: set[int] = set()
        defects: list[str] = []
        observations: list[str] = []
        bounded_nodes: dict[int, set[str]] = {}
        bounded_direct: dict[int, set[str]] = {}
        truncated_views: list[dict[str, Any]] = []
        unsourced: list[int] = []
        # Provenance for every node id the reader names. It starts as the ids rendered into the
        # frontier it was handed and grows with every id a tool result shows it. An id from
        # neither is a guess, not navigation — at the low anchor the holder leaves are literally
        # 1, 2, 4, 5, so probing small integers would otherwise score.
        visible_nodes = {node_id for _depth, node_id in delivered_frontier(context)}
        answer = ""
        status = "empty"
        detail = "the reader never produced a final message"

        def _snapshot(**overrides) -> dict[str, Any]:
            payload = {
                "chosen_node_ids": tuple(chosen),
                "recovered_store_ids": tuple(sorted(store_ids)),
                "recovered_text": "\n".join(texts),
                "evidence_defects": tuple(sorted(dict.fromkeys(defects))),
                "observations": tuple(sorted(dict.fromkeys(observations))),
                "bounded_nodes": {n: sorted(v) for n, v in bounded_nodes.items()},
                "bounded_store_ids_direct": {s: sorted(v) for s, v in bounded_direct.items()},
                "unsourced_node_ids": tuple(dict.fromkeys(unsourced)),
                "tool_result_views_truncated": truncated_views,
                "tool_calls": len(chosen) + len(unsourced),
            }
            payload.update(overrides)
            return payload

        for _step in range(args.max_tool_calls + 1):
            try:
                response, _route = call_reader_model(messages, tool_schemas, args)
            except Exception as exc:
                return _snapshot(status="error", detail=f"{type(exc).__name__}: {exc}", answer="")
            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            content = message.content if isinstance(message.content, str) else ""
            if not calls:
                answer = content
                status = "ok" if content.strip() else "empty"
                detail = "" if status == "ok" else "the reader's final message had no content"
                break
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [{"id": call.id, "type": "function",
                                "function": {"name": call.function.name,
                                             "arguments": call.function.arguments}}
                               for call in calls],
            })
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except Exception:
                    arguments = {}
                node_id = _as_int(arguments.get("node_id"))
                store_id_arg = _as_int(arguments.get("store_id"))
                sourced = node_id is None or node_id in visible_nodes
                if node_id is not None:
                    (chosen if sourced else unsourced).append(node_id)

                result = run_tool(engine, call.function.name, arguments, messages, args)
                sink: list[str] = []
                seen_ids: set[int] = set()
                _collect_text(result, sink, seen_ids)
                piece = "\n".join(sink)
                # Every node id this result showed the reader becomes navigable from here on.
                visible_nodes.update(_node_ids_in(result))
                found_causes, found_observations = label_evidence(result, piece)
                if sourced:
                    store_ids.update(seen_ids)
                    texts.append(piece)
                    defects.extend(cause.label for cause in found_causes)
                    observations.extend(found_observations)
                    # Attributed to the node (or the row) this call was about, and re-derived
                    # per call: a later page of the same node that ends with has_more false
                    # clears the bound, and a marker in one node's page never bounds another's.
                    # CAUSES are attributed the same way — a trace-level label cannot say
                    # which line it broke, so it may not excuse any of them.
                    per_row, node_wide = attribute_recovery_defects(found_causes)
                    for missing_id, labels in per_row.items():
                        bounded_direct.setdefault(missing_id, set()).update(labels)
                    bounds = {name for name in found_observations
                              if name in BOUNDED_OBSERVATIONS}
                    bounds |= set(node_wide)
                    paged = "evidence:paged_result"
                    if node_id is not None:
                        current = bounded_nodes.setdefault(node_id, set())
                        current.discard(paged)
                        current |= bounds
                    elif store_id_arg is not None:
                        current = bounded_direct.setdefault(store_id_arg, set())
                        current.discard(paged)
                        current |= bounds
                else:
                    # The text still went to the model — that is what the host would do — but
                    # it credits no recall, or probing small integers would score navigation.
                    observations.append("reader:unsourced_node_id")

                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": _tool_message_content(result, args,
                                                                  truncated_views)})
        return _snapshot(status=status, detail=detail, answer=answer)


# ── the run ─────────────────────────────────────────────────────────────────────────────────

def _corpus_patterns(corpus: dict) -> list[tuple[str, str]]:
    """Every regex the corpus carries, each with a name a maintainer can find it by."""
    entries: list[tuple[str, str]] = []
    for question in corpus.get("questions", []):
        for index, pattern in enumerate(question.get("must_not_claim", [])):
            entries.append((f"questions[{question.get('question_id')}]"
                            f".must_not_claim[{index}]", pattern))
    for claim in corpus.get("state_claims", []):
        for field_name in ("truth_patterns", "forbidden_patterns", "hedge_patterns"):
            for index, pattern in enumerate(claim.get(field_name, [])):
                entries.append((f"state_claims[{claim.get('claim_id')}]"
                                f".{field_name}[{index}]", pattern))
    return entries


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _print_block(title: str, payload: dict[str, Any]) -> None:
    print(f"=== {title} ===")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("window", nargs="?", type=int, default=262144)
    parser.add_argument("turns", nargs="?", type=int, default=400)
    parser.add_argument("--summariser", choices=("stub", "unfaithful-stub", "real"),
                        default="stub",
                        help="'real' resolves the route through the plugin's own configuration "
                             "and NEVER falls back to a stub")
    parser.add_argument("--reader", choices=("lexical", "model", "auto"), default="auto",
                        help="'auto' = model when --summariser real, lexical otherwise")
    parser.add_argument("--reader-task", default="compression",
                        help="host auxiliary task whose configured route the reader uses")
    parser.add_argument("--reader-model", default="",
                        help="provider-neutral route override, parsed by the plugin's own "
                             "model_routing; empty means the task default")
    parser.add_argument("--reader-max-tokens", type=int, default=1200)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--max-tool-result-chars", type=int, default=0,
                        help="0 (the default) shows the reader every tool result whole. A "
                             "positive value cuts, and the cut is marked in the reader's "
                             "transcript and counted in the output — never silent")
    parser.add_argument("--max-pages", type=int, default=6,
                        help="how far the scripted reader follows lcm_expand's own cursor; "
                             "what is still bounded after that is reported as a bound")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--condense-budget-fraction", type=float, default=0.001,
                        help="same fraction at both anchors — it moves the condensation GATE "
                             "so depth >= 1 is reached without padding the stub")
    parser.add_argument("--condensation-fanin", type=int, default=2)
    parser.add_argument("--require-exact-tokenizer", action="store_true",
                        help="fail unless the real encoder loaded, instead of the estimator")
    parser.add_argument("--dispatch-messages", action="store_true",
                        help="pass the reader's live messages to handle_tool_call the way the "
                             "host's native dispatcher does; off by default so the evaluation "
                             "cannot mutate the frontier it is measuring")
    parser.add_argument("--json-out", default="",
                        help="write the full result record to this path as JSON")
    args = parser.parse_args(argv)

    if args.reader == "auto":
        args.reader = "model" if args.summariser == "real" else "lexical"

    with open(args.corpus, "r", encoding="utf-8") as handle:
        corpus_raw = handle.read()
    corpus = json.loads(corpus_raw)
    corpus_digest = hashlib.sha256(corpus_raw.encode()).hexdigest()

    # Compile every corpus pattern BEFORE anything runs. The scorer treats an uncompilable
    # pattern as unmatched so one typo cannot crash a run mid-flight, which means a detector
    # can be silently switched off and the run still reports false_claims: 0. This is the
    # signal that makes that impossible.
    try:
        validate_patterns(_corpus_patterns(corpus))
    except CorpusPatternError as exc:
        print("=== e2e_index_navigation: NOT RUN ===")
        print(str(exc))
        return 5

    tokenizer = pin_tokenizer()
    if args.require_exact_tokenizer and not tokenizer["backend"].startswith("tiktoken"):
        print("FAILED: --require-exact-tokenizer was set and the real encoder did not load; "
              "the run would have counted with the char estimator.")
        return 2

    route_probe = ""
    reader_route: dict[str, Any] = {}
    try:
        if args.summariser == "real":
            route_probe = preflight_summary_route()
        if args.reader == "model":
            reader_route = preflight_reader_route(args)
    except RouteUnavailable as exc:
        print("=== e2e_index_navigation: NOT RUN ===")
        print(str(exc))
        return 3

    recorder = install_summariser(args.summariser)

    base = os.environ.get("CLAUDE_JOB_DIR") or os.environ.get("TMPDIR") or "/tmp"
    home = os.path.join(base, "tmp" if os.environ.get("CLAUDE_JOB_DIR") else "",
                        f"lcm-nav-{args.window}-{args.summariser}")
    os.makedirs(home, mode=0o700, exist_ok=True)
    db = os.path.join(home, "lcm.db")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except OSError:
            pass

    config = LCMConfig(
        database_path=db,
        # The condensation gate, not the stub: engine.py returns early while the frontier is
        # under effective_condense_budget_tokens, which a small frontier never crosses. Moving
        # the gate is a configuration change; padding the stub would only test the stub.
        summary_budget_fraction=args.condense_budget_fraction,
        condensation_fanin=args.condensation_fanin,
    )
    engine = LCMEngine(config=config, hermes_home=home)
    session = f"nav-{args.window}-{args.summariser}"
    engine.on_session_start(session, platform="cli", context_length=args.window)

    turns, placement = build_conversation(corpus, args.turns, args.seed)
    input_digest = _digest(turns)
    messages_total = sum(len(turn) for turn in turns)

    # Resolved before the run and printed with the environment: a navigation score is only
    # about the tool surface the agent actually has, so where that surface came from is part
    # of the pinned environment, not an implementation detail.
    try:
        tool_schemas, tool_surface = resolve_tool_surface(engine)
    except ToolSurfaceDrift as exc:
        print("=== e2e_index_navigation: NOT RUN ===")
        print(str(exc))
        engine.shutdown()
        return 4

    env = environment_block(args, corpus_digest, input_digest, tokenizer, engine,
                            messages_total, tool_surface)
    env["summariser_route_probe"] = route_probe or "(no real route used)"
    env["reader_route"] = reader_route or "(no model route used)"
    _print_block("environment (pinned before the run)", env)

    started = time.monotonic()
    context: list[dict] = [{"role": "system", "content": "You are a careful engineer."}]
    compactions = 0
    for turn in turns:
        context = context + [dict(message) for message in turn]
        if engine.should_compress_preflight(context):
            context = engine.compress(context)
            compactions += 1
        else:
            engine.ingest(context)
    elapsed = time.monotonic() - started

    nodes = engine._dag.get_session_nodes(session)
    depths = sorted({node.depth for node in nodes})
    frontier = delivered_frontier(context)
    rows = engine._store.get_session_messages(session)
    stored = {row["store_id"]: str(row.get("content") or "") for row in rows}
    active_text = "\n".join(message.get("content") for message in context
                            if isinstance(message.get("content"), str))

    run_block = {
        "compactions": compactions,
        "elapsed_seconds": round(elapsed, 1),
        "stored_rows": len(stored),
        "summary_nodes": len(nodes),
        "depths": depths,
        "max_depth": max(depths) if depths else -1,
        "delivered_frontier": [{"depth": depth, "node_id": node_id}
                               for depth, node_id in frontier],
        "active_messages": len(context),
        "summariser_route_calls": recorder.calls,
        "summariser_route_failures": recorder.failures,
        "summariser_route_errors": recorder.errors[:5],
    }
    _print_block("run", run_block)

    # ── hard preconditions. A run that did not do the thing is not evidence of the thing. ──
    problems: list[str] = []
    if run_block["max_depth"] < 1:
        problems.append(
            f"condensation never ran: depths {depths}. Depth >= 1 is a PRECONDITION of this "
            f"gate, not an observation — a frontier of leaves says nothing about whether a "
            f"reader can navigate a condensed index."
        )
    if not frontier:
        problems.append("no summary node was rendered into the delivered context; there was "
                        "nothing for a reader to navigate")
    if recorder.calls == 0:
        problems.append("the summariser route was never called; no summary in this run was "
                        "produced by the mode this run claims")
    if args.summariser == "real" and recorder.failures >= recorder.calls:
        problems.append("every real summariser call failed; this is not a real-summariser run")
    leaf_count = sum(1 for node in nodes if node.source_type == "messages")
    if leaf_count < 2:
        problems.append(
            f"only {leaf_count} leaf node(s) exist, so every labelled source has the same "
            f"answer and there is nothing to choose between"
        )

    # ── resolve the labels against what was actually stored ──────────────────────────────────
    anchors = {anchor["anchor_id"]: anchor for anchor in corpus["anchors"]}
    anchor_store_ids: dict[str, list[int]] = {}
    for anchor_id, anchor in anchors.items():
        needle = anchor["evidence"]
        anchor_store_ids[anchor_id] = [store_id for store_id, content in stored.items()
                                       if needle in content]
    missing_anchors = [anchor_id for anchor_id, ids in anchor_store_ids.items() if not ids]
    if missing_anchors:
        problems.append(
            f"{len(missing_anchors)} labelled anchor(s) are in no stored row at all "
            f"({', '.join(missing_anchors[:5])}); the corpus never reached the store, so "
            f"nothing downstream can be scored as a reader or model result"
        )

    frontier_ids = [node_id for _depth, node_id in frontier]
    reachable_nodes: set[int] = set()
    holders: dict[int, set[int]] = {}
    for node_id in frontier_ids:
        visited, node_holders = descend(engine, node_id)
        reachable_nodes.update(visited)
        for store_id, owners in node_holders.items():
            holders.setdefault(store_id, set()).update(owners)
    covering = {store_id: tuple(sorted(owners)) for store_id, owners in holders.items()}
    # Everything between the delivered frontier and a holder: opening one of these is how a
    # reader gets down to a leaf, so it is navigation, not waste.
    path_nodes = tuple(sorted(reachable_nodes
                              - {node for owners in holders.values() for node in owners}))

    labelled_ids = {store_id for ids in anchor_store_ids.values() for store_id in ids}
    holder_leaves = sorted({node for store_id in labelled_ids
                            for node in holders.get(store_id, ())})
    run_block["labelled_sources"] = len(labelled_ids)
    run_block["holder_leaves_for_labelled_sources"] = holder_leaves
    run_block["path_nodes_between_frontier_and_holders"] = list(path_nodes)
    if len(holder_leaves) < 2:
        # With one holder there is nothing to choose between, and a perfect node recall would
        # mean only that the reader opened the one node that exists.
        problems.append(
            f"every labelled source sits under the same {len(holder_leaves)} leaf node(s) "
            f"{holder_leaves}; node recall over that is not a navigation measurement"
        )

    raw_in_context = {store_id for anchor_id, ids in anchor_store_ids.items() for store_id in ids
                      if anchors[anchor_id]["evidence"] in active_text}

    cases: list[NavigationCase] = []
    for question in corpus["questions"]:
        expected: list[int] = []
        snippets: dict[int, str] = {}
        for anchor_id in question["anchor_ids"]:
            for store_id in anchor_store_ids.get(anchor_id, []):
                expected.append(store_id)
                snippets[store_id] = anchors[anchor_id]["evidence"]
        cases.append(NavigationCase(
            question_id=question["question_id"],
            question=question["question"],
            expected_store_ids=tuple(expected),
            covering_node_ids={sid: covering.get(sid, ()) for sid in expected},
            path_node_ids=path_nodes,
            raw_in_context_store_ids=tuple(sid for sid in expected if sid in raw_in_context),
            evidence_snippets=snippets,
            must_mention=tuple(question.get("must_mention", ())),
            must_not_claim=tuple(question.get("must_not_claim", ())),
            category=question.get("category", ""),
        ))

    # ── the reader stage ────────────────────────────────────────────────────────────────────
    reader = (ModelReader(args) if args.reader == "model"
              else LexicalReader(args.max_tool_calls, args.max_pages))
    if args.reader != "model":
        tool_schemas = []  # the scripted reader dispatches directly; nothing is offered to it
    traces: list[ReaderTrace] = []
    reader_detail: list[dict[str, Any]] = []
    # One walk per node, reused for every question, so bound attribution is cheap.
    store_ids_by_node = {node.node_id: store_ids_under(engine, node.node_id) for node in nodes}
    for case in cases:
        outcome = reader.read(engine, context, case.question, tool_schemas, args)
        # A bound applies to the rows the bounded node holds, and to nothing else. Attributing
        # it run-wide let one marker in one page bound every line the reader was looking for.
        bounded_store_ids: dict[int, set[str]] = {
            int(store_id): set(labels)
            for store_id, labels in (outcome.get("bounded_store_ids_direct") or {}).items()
        }
        for node_id, labels in (outcome.get("bounded_nodes") or {}).items():
            if not labels:
                continue  # the node was read whole; it bounds nothing
            for store_id in store_ids_by_node.get(int(node_id), ()):
                bounded_store_ids.setdefault(store_id, set()).update(labels)
        traces.append(ReaderTrace(
            question_id=case.question_id,
            status=outcome["status"],
            detail=outcome["detail"],
            chosen_node_ids=outcome["chosen_node_ids"],
            recovered_store_ids=outcome["recovered_store_ids"],
            recovered_text=outcome["recovered_text"],
            answer=outcome["answer"],
            evidence_defects=outcome["evidence_defects"],
            observations=outcome["observations"],
            bounded_store_ids={sid: tuple(sorted(labels))
                               for sid, labels in bounded_store_ids.items()},
            unsourced_node_ids=tuple(outcome.get("unsourced_node_ids") or ()),
        ))
        reader_detail.append({
            "question_id": case.question_id,
            "tool_calls": outcome["tool_calls"],
            "pages_followed": outcome.get("pages_followed", 0),
            "budget_exhausted": bool(outcome.get("budget_exhausted")),
            "chosen_node_ids": list(outcome["chosen_node_ids"]),
            "unsourced_node_ids": list(outcome.get("unsourced_node_ids") or ()),
            "bounded_nodes": outcome.get("bounded_nodes") or {},
            "tool_result_views_truncated": outcome.get("tool_result_views_truncated") or [],
            "answer_excerpt": (outcome["answer"] or "")[:240],
        })

    navigation = score_navigation(cases, traces)
    navigation["reader"] = reader.name
    navigation["measures_a_model"] = args.reader == "model"
    navigation["per_reader_call"] = reader_detail
    exhausted = sum(1 for item in reader_detail if item.get("budget_exhausted"))
    navigation["reader_budget_exhausted"] = exhausted
    if exhausted:
        navigation["complete"] = False
        navigation["incomplete_reasons"] = list(navigation["incomplete_reasons"]) + [
            f"{exhausted} question(s) ran out of the --max-tool-calls budget before the "
            f"descent reached a leaf; those misses say nothing about the index"
        ]

    # ── fidelity over the SAME run ──────────────────────────────────────────────────────────
    claims = [StateClaim(
        claim_id=item["claim_id"],
        entity=item["entity"],
        true_state=item["true_state"],
        source_store_ids=tuple(store_id for anchor_id in item["anchor_ids"]
                               for store_id in anchor_store_ids.get(anchor_id, [])),
        truth_patterns=tuple(item.get("truth_patterns", ())),
        forbidden_patterns=tuple(item.get("forbidden_patterns", ())),
        hedge_patterns=tuple(item.get("hedge_patterns", ())),
    ) for item in corpus["state_claims"]]

    texts = _fidelity_texts(engine, nodes, claims, traces, cases)
    fidelity = score_fidelity(claims, texts)
    fidelity["summariser"] = args.summariser
    fidelity["measures_a_model"] = args.summariser == "real"

    # A route that failed for SOME chunks leaves those chunks unsummarised — escalation raises
    # rather than publishing a fallback — so the index this run scored is partly absent. That
    # is not a clean measurement of anything, whichever way the numbers came out.
    if recorder.failures:
        reason = (
            f"{recorder.failures} of {recorder.calls} summariser call(s) failed, so part of "
            f"this run's index was never written: "
            f"{'; '.join(recorder.errors[:3]) or 'no error recorded'}"
        )
        for block in (fidelity, navigation):
            block["complete"] = False
            block["incomplete_reasons"] = list(block["incomplete_reasons"]) + [reason]
        fidelity["summariser_route_failures"] = recorder.failures

    note = (
        "A stub summariser or a lexical reader proves the chain and the scorers, and measures "
        "no model. Only --summariser real --reader model measures one, and then only the route "
        "this environment is configured for, on one labelled corpus."
    )
    # Mirrored into BOTH score blocks: either one gets copied into an issue on its own, and a
    # block that does not carry its run's validity reads as valid even when condensation never
    # fired (minor 7).
    for block in (navigation, fidelity):
        block["run_valid"] = not problems
        block["preconditions_failed"] = problems
        block["window"] = args.window
        block["turns"] = args.turns
        block["note"] = note
    navigation["observations_note"] = (
        "observations and recalls are tallied over SCORED cases only "
        f"({navigation['scored']} of {navigation['cases_total']}); cases withdrawn for a "
        "reader or evidence reason contribute to neither"
    )
    navigation["frontier_shape"] = {
        "delivered_nodes": len(frontier),
        "holder_leaves": len(run_block["holder_leaves_for_labelled_sources"]),
        "note": (
            "This run moved the condensation gate with --condense-budget-fraction "
            f"{args.condense_budget_fraction}, which is not production's default. With a "
            "single delivered node the run measures DESCENT to the holding leaf rather than "
            "choice among delivered nodes; the >=2-holder-leaf precondition keeps it from "
            "being vacuous, but the index shape under test is not the shape production has "
            "today."
        ),
    }

    _print_block("navigation (#2) — source findability from the delivered frontier", navigation)
    _print_block("fidelity (#8) — statement truth in the same run's summaries and answers",
                 fidelity)

    verdict = {
        "run_valid": not problems,
        "preconditions_failed": problems,
        "navigation_complete": navigation["complete"],
        "fidelity_complete": fidelity["complete"],
        "measures_a_model": {"navigation": navigation["measures_a_model"],
                             "fidelity": fidelity["measures_a_model"]},
        "note": note,
    }
    _print_block("verdict", verdict)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"environment": env, "run": run_block, "navigation": navigation,
                       "fidelity": fidelity, "verdict": verdict}, handle, indent=2, default=str)

    engine.shutdown()
    return 0 if not problems else 1


def _mirror_normalize_tool_schema(schema: Any) -> Optional[dict]:
    """The two rules ``agent.memory_manager.normalize_tool_schema`` applies, and no others.

    Used only when the host module cannot be imported at all. When it CAN be imported, the
    host's own function is what builds the surface and this is run beside it purely to prove
    the two still agree — see :func:`resolve_tool_surface`.
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    name = schema.get("name", "")
    # a nameless tool makes strict providers 400, so the host drops it and so does this
    return schema if name and isinstance(name, str) else None


class ToolSurfaceDrift(RuntimeError):
    """The mirror no longer matches the host's normaliser. Never scored through."""


def resolve_tool_surface(engine: LCMEngine) -> tuple[list[dict], dict[str, Any]]:
    """The tool surface Hermes really injects, plus where it came from.

    ``agent_init.py::_inject_context_engine_tools`` appends exactly
    ``engine.get_tool_schemas()`` — gated on the ``context_engine`` toolset — after passing
    each through ``agent.memory_manager.normalize_tool_schema``. Scoring navigation against a
    surface the agent does not have would measure the wrong thing, so the host's function is
    used whenever it is importable, and the local mirror is only a fallback for a host-less
    checkout.

    When both are available they are compared, and a disagreement raises rather than being
    resolved in either direction. A mirror that has silently drifted from the host is exactly
    the after-an-upstream-merge failure this fork's standing merge task exists to catch, and a
    harness that scores through it would report a confident number about a surface nobody runs.
    """
    raw = list(engine.get_tool_schemas())
    mirrored = [_mirror_normalize_tool_schema(item) for item in raw]
    try:
        from agent.memory_manager import normalize_tool_schema
    except Exception as exc:
        provenance = {
            "source": "local mirror of agent.memory_manager.normalize_tool_schema",
            "host_importable": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "mirror_checked_against_host": False,
        }
        normalized = mirrored
    else:
        normalized = [normalize_tool_schema(item) for item in raw]
        drift = [
            {"index": index,
             "name": (item or {}).get("name") if isinstance(item, dict) else None,
             "host": host, "mirror": mirror}
            for index, (item, host, mirror) in enumerate(zip(raw, normalized, mirrored))
            if host != mirror
        ]
        if drift:
            raise ToolSurfaceDrift(
                "the local mirror of agent.memory_manager.normalize_tool_schema no longer "
                f"agrees with the host on {len(drift)} schema(s): "
                f"{json.dumps(drift[:3], default=str)[:600]}. Refusing to score a navigation "
                "run against a tool surface that may not be the one Hermes offers."
            )
        provenance = {
            "source": "agent.memory_manager.normalize_tool_schema (host)",
            "host_importable": True,
            "mirror_checked_against_host": True,
            "mirror_agrees": True,
        }
    surface = [{"type": "function", "function": schema} for schema in normalized if schema]
    provenance["tools"] = [entry["function"]["name"] for entry in surface]
    return surface, provenance


def _fidelity_texts(engine, nodes, claims, traces, cases) -> list[ScoredText]:
    """Every published summary plus every reader answer, each with the claims it must carry."""
    by_claim = {claim.claim_id: claim for claim in claims}
    texts: list[ScoredText] = []
    for node in nodes:
        under = store_ids_under(engine, node.node_id)
        claim_ids = tuple(claim.claim_id for claim in claims
                          if under.intersection(claim.source_store_ids))
        if not claim_ids:
            continue
        source_text, unreadable, truncated = coverage_doctor._source_text_for_node(engine, node)
        carries = {}
        for claim_id in claim_ids:
            claim = by_claim[claim_id]
            carries[claim_id] = bool(
                claim.entity.lower() in source_text.lower()
                or any(re.search(pattern, source_text, re.IGNORECASE)
                       for pattern in claim.truth_patterns)
            )
        index_block = ""
        meta_store = getattr(engine._dag, "node_meta", None)
        if meta_store is not None:
            try:
                meta = meta_store.read(node.node_id) or {}
                index_block = str(meta.get("index_block") or "")
            except Exception:
                index_block = ""
        texts.append(ScoredText(
            text_id=f"node:{node.node_id}",
            kind="leaf" if node.depth == 0 else "condensation",
            depth=node.depth,
            text=f"{node.summary}\n{index_block}",
            claim_ids=claim_ids,
            source_carries_claim=carries,
            entity_coverage=coverage_doctor.coverage_of(
                node.summary, index_block, source_text, source_truncated=truncated),
        ))
    cases_by_id = {case.question_id: case for case in cases}
    for trace in traces:
        case = cases_by_id.get(trace.question_id)
        if case is None or trace.status != "ok":
            continue
        relevant = tuple(claim.claim_id for claim in claims
                         if set(claim.source_store_ids).intersection(case.expected_store_ids))
        if not relevant:
            continue
        carries = {}
        for claim_id in relevant:
            claim = by_claim[claim_id]
            carries[claim_id] = bool(
                claim.entity.lower() in trace.recovered_text.lower()
                or any(re.search(pattern, trace.recovered_text, re.IGNORECASE)
                       for pattern in claim.truth_patterns)
            )
        texts.append(ScoredText(
            text_id=f"answer:{trace.question_id}",
            kind="answer",
            depth=None,
            text=trace.answer,
            claim_ids=relevant,
            source_carries_claim=carries,
            entity_coverage=None,
        ))
    return texts


if __name__ == "__main__":
    raise SystemExit(main())
