"""End-to-end no-loss exercise against the DEPLOYED better-hermeslcm plugin.

Not a unit test: it drives the engine the way the host does — feeding compress()'s own return
value back as the next turn's context — over a long, tool-heavy conversation, then checks the
property the fork exists for:

    every stored raw message is either still in the active context, or reachable from a
    published summary node (directly, or through condensation), and every distinctive fact is
    mentioned by whatever covers it.

It runs against whatever is INSTALLED at ~/.hermes/plugins/hermes-lcm, on the host's own
interpreter — which is how it caught a crash in the parallel leaf pipeline (a CPython 3.14
private-API change) that 3,200 unit tests could not see.

Usage: scripts/e2e_no_loss.py <window_tokens> <turns>
       scripts/e2e_no_loss.py 262144 400      # low anchor, two compactions
       scripts/e2e_no_loss.py 1000000 3000    # high anchor, parallel lookahead
"""
import importlib.util
import os
import re
import sys
import json

PLUGIN = "/home/agent/.hermes/plugins/hermes-lcm"
sys.path.insert(0, "/home/agent/.hermes/hermes-agent")
spec = importlib.util.spec_from_file_location(
    "hermes_lcm", f"{PLUGIN}/__init__.py", submodule_search_locations=[PLUGIN]
)
hermes_lcm = importlib.util.module_from_spec(spec)
sys.modules["hermes_lcm"] = hermes_lcm
spec.loader.exec_module(hermes_lcm)

from hermes_lcm import escalation  # noqa: E402
from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402

FACT_RE = re.compile(r"FACT-\d{5}")


SEEN_BY_SUMMARISER: set = set()


def fake_summary(prompt, max_tokens, model="", timeout=None):
    """A summariser that behaves like an INDEX: it names every fact id it was given."""
    if isinstance(prompt, list):
        source = "\n".join(str(m.get("content") or "") for m in prompt)
    else:
        source = str(prompt)
    facts = sorted(set(FACT_RE.findall(source)))
    SEEN_BY_SUMMARISER.update(facts)
    if not facts:
        return "Session housekeeping with no recorded facts.\nExpand for details about: housekeeping"
    return (
        f"Covered {len(facts)} recorded fact(s): {', '.join(facts)}.\n"
        f"Expand for details about: {facts[0]}..{facts[-1]}"
    )


def build_turn(index: int) -> list:
    fact = f"FACT-{index:05d}"
    if index % 4 == 3:
        call_id = f"call_{index}"
        return [
            {"role": "user", "content": f"{fact}: please read the deployment log."},
            {"role": "assistant", "content": f"Reading it now for {fact}.",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": json.dumps({"path": f"/logs/{index}.log"})}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": f"{fact} log body\n" + ("x" * 4000)},
            {"role": "assistant", "content": f"The log for {fact} shows the rollout finished."},
        ]
    return [
        {"role": "user", "content": f"{fact}: decide whether to keep the {index % 7} config."},
        {"role": "assistant", "content":
            f"Decision for {fact}: keep it, because the alternative regressed. " + ("detail " * 200)},
    ]


def reachable_store_ids(engine, session_id: str) -> set:
    """Every raw store_id reachable from any node of this session, walking through nodes."""
    nodes = engine._dag.get_session_nodes(session_id)
    by_id = {node.node_id: node for node in nodes}
    seen_nodes, store_ids, stack = set(), set(), [node.node_id for node in nodes]
    while stack:
        node_id = stack.pop()
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        node = by_id.get(node_id)
        if node is None:
            continue
        if node.source_type == "messages":
            store_ids.update(int(value) for value in node.source_ids)
        else:
            stack.extend(int(value) for value in node.source_ids)
    return store_ids


def main() -> int:
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    base = os.environ.get("CLAUDE_JOB_DIR") or os.environ.get("TMPDIR") or "/tmp"
    home = os.path.join(base, "tmp" if os.environ.get("CLAUDE_JOB_DIR") else "", f"lcm-e2e-{window}")
    os.makedirs(home, mode=0o700, exist_ok=True)
    db = os.path.join(home, "lcm.db")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except OSError:
            pass

    escalation._call_llm_for_summary = fake_summary
    config = LCMConfig(database_path=db)
    engine = LCMEngine(config=config, hermes_home=home)
    session = f"e2e-{window}"
    engine.on_session_start(session, platform="cli", context_length=window)

    context: list = [{"role": "system", "content": "You are a careful engineer."}]
    compactions = 0
    for index in range(turns):
        context = context + build_turn(index)
        if engine.should_compress_preflight(context):
            before = len(context)
            context = engine.compress(context)
            compactions += 1
            if len(context) != before:
                pass
        else:
            engine.ingest(context)

    rows = engine._store.get_session_messages(session)
    stored = {row["store_id"]: str(row.get("content") or "") for row in rows}
    covered = reachable_store_ids(engine, session)
    active_text = "\n".join(str(m.get("content") or "") for m in context)
    nodes = engine._dag.get_session_nodes(session)
    node_text = "\n".join(node.summary for node in nodes)

    orphan_rows = []
    for store_id, content in stored.items():
        if store_id in covered:
            continue
        fact = FACT_RE.search(content)
        if fact and fact.group(0) in active_text:
            continue          # still raw in the active context
        if not fact:
            continue          # no distinctive marker to trace (scaffold/marker rows)
        orphan_rows.append((store_id, content[:60]))

    facts_sent = {f"FACT-{index:05d}" for index in range(turns)}
    facts_visible = set(FACT_RE.findall(active_text)) | set(FACT_RE.findall(node_text))
    missing_facts = sorted(facts_sent - facts_visible)
    # the sharper question: did anything vanish BEFORE the summariser could index it?
    never_offered = sorted(facts_sent - SEEN_BY_SUMMARISER - set(FACT_RE.findall(active_text)))

    print(f"window={window} turns={turns} compactions={compactions}")
    print(f"  stored rows: {len(stored)}   summary nodes: {len(nodes)} "
          f"(depths {sorted({n.depth for n in nodes})})")
    print(f"  rows reachable from a node: {len(covered)}")
    print(f"  active context: {len(context)} messages, "
          f"{engine.last_prompt_tokens or 'n/a'} host tokens, threshold {engine.threshold_tokens}")
    print(f"  UNREACHABLE rows (neither raw in context nor under a node): {len(orphan_rows)}")
    for row in orphan_rows[:10]:
        print("     ", row)
    print(f"  facts sent: {len(facts_sent)}   facts findable in context+index: {len(facts_visible)}")
    print(f"  MISSING facts (not in context and not in any index): {len(missing_facts)} "
          f"{missing_facts[:10]}")
    print(f"  facts never offered to the summariser and not raw in context: "
          f"{len(never_offered)} {never_offered[:10]}")
    status = json.loads(engine.handle_tool_call("lcm_status", {}))
    print("  status:", {k: status.get(k) for k in
                        ("compression_count", "last_compression_status", "threshold_tokens")
                        if k in status})
    engine.shutdown()
    return 0 if not orphan_rows and not missing_facts and not never_offered else 1


if __name__ == "__main__":
    raise SystemExit(main())
