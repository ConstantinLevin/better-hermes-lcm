"""#50/#51 — the read side may not report complete over evidence it did not read.

Six shapes reported completeness they had not earned (a page stopped at the token budget, a
truncated last child summary, a requested hydration whose archive file was gone, an error-only
block dropped from the recursive walk, the raw-search projection's silent omissions, and an
unreadable index sidecar), and the accountant that decides how much still fits measured a
fraction of what is actually serialised — so the walk's stop limits fired against the wrong
number, and when they fired nothing said which evidence was never loaded.

An explicitly requested, exactly resumable page is NOT loss: the cursors here stay exact, and
the tests below reconstruct the originals through them byte for byte. What changes is only what
the response CLAIMS about itself.
"""
import json
import time

import pytest

from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_tokens


def _engine(tmp_path, name="completeness.db", **kw):
    cfg = LCMConfig(**kw)
    cfg.database_path = str(tmp_path / name)
    return LCMEngine(config=cfg, hermes_home=str(tmp_path))


def _leaf(engine, session, store_ids, summary="a leaf\nExpand for details about: it", depth=0):
    return engine._dag.add_node_with_meta(
        SummaryNode(session_id=session, depth=depth, summary=summary, token_count=5,
                    source_token_count=500, source_ids=list(store_ids),
                    source_type="messages", created_at=time.time()),
        level=1,
    )


def _parent(engine, session, child_ids, summary="a parent\nExpand for details about: it", depth=1):
    return engine._dag.add_node_with_meta(
        SummaryNode(session_id=session, depth=depth, summary=summary, token_count=5,
                    source_token_count=500, source_ids=list(child_ids),
                    source_type="nodes", created_at=time.time()),
        level=1,
    )


def _capture_blocks(monkeypatch):
    captured = {}

    def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
        captured["blocks"] = context_blocks
        return "a synthesised answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
    return captured


def _serialized(payload):
    return count_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


# -- 50a: a page that stopped at the budget -------------------------------------------------

def test_a_page_that_stopped_at_the_token_budget_is_not_complete(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("pg", platform="cli", context_length=200_000)
        bodies = [f"body {index}: " + ("detail " * 200) for index in range(3)]
        store_ids = [e._store.append("pg", {"role": "user", "content": body}, source="cli")
                     for body in bodies]
        e._store.commit()
        node_id = _leaf(e, "pg", store_ids)

        args = {"node_id": node_id, "max_tokens": 40}
        recovered = ""
        pages = 0
        while True:
            page = json.loads(lcm_tools.lcm_expand(args, engine=e))
            pagination = page["pagination"]
            recovered += "".join(m["content"] for m in page["expanded"])
            pages += 1
            if not pagination["has_more"]:
                break
            # a page with a continuation has NOT returned everything the node holds
            assert pagination["complete"] is False, pagination
            assert pagination["incomplete_reason"], pagination
            assert pages < 200
            args = {
                "node_id": node_id,
                "max_tokens": 40,
                "source_offset": pagination["next_source_offset"],
                "content_offset": pagination["next_content_offset"],
            }
        # the LAST page — nothing left — is complete, and the pages reassemble the originals
        assert page["pagination"]["complete"] is True, page["pagination"]
        assert recovered == "".join(bodies)
    finally:
        e.shutdown()


# -- 50b: the truncated LAST child summary --------------------------------------------------

def test_a_truncated_last_child_summary_is_not_a_complete_expansion(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("cs", platform="cli", context_length=200_000)
        long_summary = "decision " * 400 + "\nExpand for details about: decisions"
        child = _leaf(e, "cs", [1], summary=long_summary)
        parent = _parent(e, "cs", [child])

        payload = json.loads(lcm_tools.lcm_expand({"node_id": parent, "max_tokens": 40}, engine=e))
        pagination = payload["pagination"]
        assert payload["expanded"][0]["summary_truncated"] is True
        # no child follows, so the CHILD LIST is exhausted — but the content is not
        assert pagination["has_more"] is False
        assert pagination["complete"] is False, pagination
        assert "summary" in pagination["incomplete_reason"], pagination
        assert pagination["truncated_child_summaries"][0]["node_id"] == child
        assert pagination["truncated_child_summaries"][0]["continue_with"]["tool"] == "lcm_describe"
    finally:
        e.shutdown()


# -- 50c: a REQUESTED hydration whose archive file is gone ----------------------------------

def test_a_requested_hydration_that_cannot_read_its_archive_fails_closed(tmp_path):
    from hermes_lcm.externalize import get_large_output_storage_dir

    e = _engine(
        tmp_path,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=100,
    )
    try:
        e.on_session_start("hy", platform="cli", context_length=200_000)
        e.threshold_tokens = 1
        big = "BIGOUTPUT " + "z" * 4000
        e.compress([
            {"role": "assistant", "content": "run", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big},
            {"role": "user", "content": "tail"},
        ])
        node_id = e._dag.get_session_nodes("hy")[0].node_id
        hydrated = json.loads(lcm_tools.lcm_expand(
            {"node_id": node_id, "hydrate": True}, engine=e))
        assert big in json.dumps(hydrated["expanded"]), "fixture: the payload should hydrate"

        storage = get_large_output_storage_dir(e._config, hermes_home=str(tmp_path), create=False)
        removed = sorted(storage.glob("*.json"))
        assert removed, "fixture: expected an archived payload file"
        for path in removed:
            path.unlink()

        # hydration was REQUESTED and could not be delivered: fail closed
        failed = json.loads(lcm_tools.lcm_expand(
            {"node_id": node_id, "hydrate": True}, engine=e))
        assert failed["pagination"]["complete"] is False, failed["pagination"]
        assert failed["pagination"]["unhydrated_externalized_refs"], failed["pagination"]
        tool_row = next(m for m in failed["expanded"] if m.get("role") == "tool")
        assert tool_row["hydration_failed"] is True
        assert tool_row["hydration_failed_reason"]

        # ... and the ordinary ref view, which never asked for the payload, is a different case
        plain = json.loads(lcm_tools.lcm_expand({"node_id": node_id}, engine=e))
        assert plain["pagination"]["complete"] is True, plain["pagination"]
    finally:
        e.shutdown()


# -- 50d: error-only blocks dropped by the RECURSIVE walk -----------------------------------

def test_a_missing_source_under_a_descendant_is_not_dropped_from_the_walk(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("dw", platform="cli", context_length=200_000)
        leaf = _leaf(e, "dw", [987654321])          # the row is not there
        middle = _parent(e, "dw", [leaf], depth=1)
        root = _parent(e, "dw", [middle], depth=2)

        captured = _capture_blocks(monkeypatch)
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query", {"prompt": "what happened?", "node_ids": [root]}))

        assert any(
            block.get("pagination", {}).get("missing_source_store_ids") == [987654321]
            for block in captured["blocks"]
        ), [block.get("type") for block in captured["blocks"]]
        assert payload["complete"] is False, payload
        assert payload["missing_source_store_ids"] == [987654321]
    finally:
        e.shutdown()


def test_a_missing_child_node_under_a_descendant_is_not_dropped_from_the_walk(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("dc", platform="cli", context_length=200_000)
        middle = _parent(e, "dc", [4242424], depth=1)   # the child node is not there
        root = _parent(e, "dc", [middle], depth=2)

        captured = _capture_blocks(monkeypatch)
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query", {"prompt": "what happened?", "node_ids": [root]}))

        assert any(
            block.get("pagination", {}).get("missing_source_node_ids") == [4242424]
            for block in captured["blocks"]
        ), [block.get("type") for block in captured["blocks"]]
        assert payload["complete"] is False, payload
    finally:
        e.shutdown()


# -- the raw-search projection --------------------------------------------------------------

def test_a_raw_search_hit_says_what_its_projection_left_out(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("rs", platform="cli", context_length=200_000)
        prefix = "NEVER deploy to production. " * 40
        e._store.append("rs", {
            "role": "assistant",
            "content": prefix + "auditneedle describes the cancelled deployment.",
            "tool_calls": [{"id": "c9", "type": "function", "function": {
                "name": "deploy", "arguments": json.dumps({"target": "prod", "confirm": False})}}],
            "reasoning_content": "weighing the rollback " * 50,
        }, source="cli")
        e._store.commit()

        captured = _capture_blocks(monkeypatch)
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "what does auditneedle say?", "query": "auditneedle",
             "context_max_tokens": 60}))

        raw = next(b for b in captured["blocks"] if b["type"] == "raw_messages")
        hit = raw["messages"][0]
        # the window starts at the match, so everything before it was left out — say so
        assert hit["content_prefix_omitted_chars"] == len(prefix)
        assert hit["content_prefix_continue_with"]["content_offset"] == 0
        # the calls and the envelope are either projected or named, never silently dropped
        assert hit.get("tool_calls") or hit.get("tool_calls_omitted")
        assert hit.get("envelope") or hit.get("envelope_omitted")
        assert raw["pagination"]["complete"] is False, raw["pagination"]
        assert raw["pagination"]["incomplete_reason"]
        assert payload["complete"] is False, payload
    finally:
        e.shutdown()


# -- the summary block's own read status ----------------------------------------------------

def test_an_unreadable_index_sidecar_makes_the_answer_incomplete(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ix", platform="cli", context_length=200_000)
        store_id = e._store.append("ix", {"role": "user", "content": "a decision"}, source="cli")
        e._store.commit()
        node_id = _leaf(e, "ix", [store_id])

        def explode(*args, **kwargs):
            raise RuntimeError("sidecar table is unreadable")

        monkeypatch.setattr(e._dag.node_meta, "read", explode)
        _capture_blocks(monkeypatch)
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query", {"prompt": "what was decided?", "node_ids": [node_id]}))

        assert payload["complete"] is False, payload
        assert payload["index_block_read_failures"], payload
    finally:
        e.shutdown()


# -- 51a: the accountant measures what is actually serialised -------------------------------

def test_the_context_budget_counts_what_is_actually_serialised(tmp_path, monkeypatch):
    """The old counter charged summary text, the source path, message content and child
    summaries — not tool calls, not the envelope, not the index block, not the JSON itself. A
    DAG of call-only rows emitted ten times its budget and called the result complete."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("bg", platform="cli", context_length=200_000)
        children = []
        for index in range(6):
            store_id = e._store.append("bg", {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{index}", "type": "function", "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": f"/tmp/{index}", "body": "x" * 3000})}}],
            }, source="cli")
            children.append(_leaf(e, "bg", [store_id]))
        root = _parent(e, "bg", children, depth=1)

        captured = _capture_blocks(monkeypatch)
        budget = 2_000
        json.loads(e.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "what was written?", "node_ids": [root], "context_max_tokens": budget}))

        blocks = captured["blocks"]
        emitted = _serialized(blocks)
        # the counter that decides what still fits and the thing that is sent are one number
        assert lcm_tools._context_content_token_count(blocks) == emitted
        assert emitted <= budget * 2, (emitted, budget)
    finally:
        e.shutdown()


def test_the_counter_and_the_declared_overhead_equal_what_synthesis_actually_sends(tmp_path):
    """`_expand_message_sources` charged body + calls + envelope as raw text while the query
    walk charged a different, smaller thing. Two accountants in one file, disagreeing.

    The check is against `build_untrusted_data_messages` — the thing that is really sent — and
    not against a copy of the counter's own arithmetic, which would prove only that the
    implementation equals itself."""
    from hermes_lcm.prompt_boundary import build_untrusted_data_messages
    from hermes_lcm.tokens import count_messages_tokens

    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ac", platform="cli", context_length=200_000)
        store_id = e._store.append("ac", {
            "role": "assistant",
            "content": "deploying now",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "deploy", "arguments": json.dumps({"target": "prod"})}}],
            "reasoning_content": "thinking about the rollout " * 200,
        }, source="cli")
        e._store.commit()
        leaf = _leaf(e, "ac", [store_id])
        e._dag.node_meta.write(leaf, level=1, summary="a leaf",
                               index_block="- topic one\n- topic two\n" * 120)
        root = _parent(e, "ac", [leaf])

        blocks = lcm_tools._collect_context_blocks_for_node(
            e, e._dag.get_node(root), max_tokens=100_000, hydrate_externalized_content=True)
        rendered = json.dumps(blocks)
        # the calls, the envelope and the index block are all really in there
        assert "index_block" in rendered and "tool_calls" in rendered and "envelope" in rendered

        prompt = "what was deployed?"
        emitted = count_messages_tokens(build_untrusted_data_messages(
            operation="lcm_expand_query",
            system_instructions=lcm_tools._EXPANSION_SYSTEM_PROMPT,
            request={"question": prompt},
            sources=[{"provenance": {"source_type": "expanded_lcm_context",
                                     "block_count": len(blocks)},
                      "content": blocks}],
        ))
        counted = lcm_tools._context_content_token_count(blocks)
        overhead = lcm_tools._synthesis_request_overhead_tokens(prompt)
        assert abs(emitted - (counted + overhead)) <= 4, (emitted, counted, overhead)
    finally:
        e.shutdown()


# -- 51c / 51d: work that was never done must say so ----------------------------------------

def test_a_budget_exhausted_before_the_walk_says_the_evidence_was_never_read(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("nb", platform="cli", context_length=200_000)
        store_id = e._store.append(
            "nb", {"role": "user", "content": "DECISION: cancel the launch"}, source="cli")
        e._store.commit()
        leaf = _leaf(e, "nb", [store_id])
        root = _parent(e, "nb", [leaf])

        captured = _capture_blocks(monkeypatch)
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "what was decided?", "node_ids": [root], "context_max_tokens": 1}))

        unread = [b for b in captured["blocks"] if b.get("type") == "unread_evidence"]
        assert unread, [b.get("type") for b in captured["blocks"]]
        assert unread[0]["pagination"]["complete"] is False
        assert "budget" in unread[0]["pagination"]["incomplete_reason"]
        # the receipt names the node that HOLDS the unread evidence, not the root whose
        # traversal already failed: following it must return what was missed, not repeat it
        assert leaf in unread[0]["pagination"]["unread_node_ids"]
        assert unread[0]["pagination"]["continue_with"]["node_id"] == leaf
        assert payload["complete"] is False, payload
    finally:
        e.shutdown()


def test_a_walk_stopped_at_the_node_visit_cap_names_the_work_it_did_not_do(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("nv", platform="cli", context_length=200_000)
        store_ids = [e._store.append("nv", {"role": "user", "content": f"row {i}"}, source="cli")
                     for i in range(4)]
        e._store.commit()
        leaves = [_leaf(e, "nv", [store_id]) for store_id in store_ids]
        root = _parent(e, "nv", leaves, depth=1)

        node = e._dag.get_node(root)
        blocks = lcm_tools._collect_descendant_evidence_blocks(
            e, node, max_tokens=100_000, remaining_node_visits=[1])
        unread = [b for b in blocks if b.get("type") == "unread_evidence"]
        assert unread, [b.get("type") for b in blocks]
        assert "visit" in unread[0]["pagination"]["incomplete_reason"]
    finally:
        e.shutdown()


def test_a_walk_that_finished_everything_claims_no_omission(tmp_path):
    """A receipt is a claim that something was left unread. The fix must not emit one for a
    parent whose sources were all processed — an exhausted frame is still on the stack when
    the loop ends."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("done", platform="cli", context_length=200_000)
        store_ids = [e._store.append("done", {"role": "user", "content": f"row {i}"}, source="cli")
                     for i in range(3)]
        e._store.commit()
        leaves = [_leaf(e, "done", [store_id]) for store_id in store_ids]
        root = _parent(e, "done", leaves, depth=1)

        node = e._dag.get_node(root)
        blocks = lcm_tools._collect_descendant_evidence_blocks(e, node, max_tokens=100_000)
        assert [b for b in blocks if b["type"] == "child_messages"], blocks
        assert not [b for b in blocks if b.get("type") == "unread_evidence"], blocks
        assert all(b["pagination"]["complete"] is True for b in blocks), blocks
    finally:
        e.shutdown()


def _roots_emission(engine, monkeypatch, roots, budget):
    captured = _capture_blocks(monkeypatch)
    payload = json.loads(engine.handle_tool_call("lcm_expand_query", {
        "prompt": "what was decided?", "node_ids": roots,
        "max_results": len(roots), "context_max_tokens": budget,
    }))
    return payload, captured["blocks"], _serialized(captured["blocks"])


def test_roots_the_budget_never_reached_cost_only_their_names(tmp_path, monkeypatch):
    """The receipt for unread work may not be trimmed away — so it must not have to compete for
    the budget in the first place. Emitting a per-root block plus a per-root receipt made the
    context grow with the caller's own node list, far past the budget it was given, and an
    oversized request is a request a host may truncate mid-JSON."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("mr", platform="cli", context_length=200_000)
        roots = []
        for index in range(40):
            store_id = e._store.append(
                "mr", {"role": "user", "content": f"decision {index}: " + "detail " * 200},
                source="cli")
            roots.append(_leaf(e, "mr", [store_id], summary=f"leaf {index} " * 40))
        e._store.commit()

        payload_few, _, emitted_few = _roots_emission(e, monkeypatch, roots[:4], 1)
        payload_many, blocks_many, emitted_many = _roots_emission(e, monkeypatch, roots, 1)

        # 36 further roots the budget cannot reach cost their ids in one receipt, not a block
        # and a receipt each
        assert emitted_many - emitted_few < 300, (emitted_few, emitted_many)
        named = {
            node_id
            for block in blocks_many if block.get("type") == "unread_evidence"
            for node_id in block["pagination"]["unread_node_ids"]
        }
        assert len(named) >= 30, sorted(named)
        assert payload_many["complete"] is False
        assert payload_few["complete"] is False
    finally:
        e.shutdown()


def test_a_receipt_is_not_emitted_for_a_node_that_holds_nothing(tmp_path):
    """A receipt is a claim that something was not read. A node with no children below it has
    nothing unread, and `unread_node_ids: []` would be a false claim of the same family."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("empty", platform="cli", context_length=200_000)
        childless = _parent(e, "empty", [], summary="a parent with no recorded children")
        node = e._dag.get_node(childless)
        assert lcm_tools._collect_descendant_evidence_blocks(e, node, max_tokens=0) == []
        assert lcm_tools._collect_descendant_evidence_blocks(
            e, node, max_tokens=100, remaining_node_visits=[0]) == []
    finally:
        e.shutdown()


def test_a_requested_window_and_a_budget_cut_do_not_give_the_same_reason(tmp_path):
    """Both leave the page short of what the node holds, and both are honest — but one is the
    caller's own instruction and the other is a limit they did not choose."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("rw", platform="cli", context_length=200_000)
        store_ids = [e._store.append("rw", {"role": "user", "content": f"body {i}: " + "x" * 400},
                                     source="cli") for i in range(3)]
        e._store.commit()
        node_id = _leaf(e, "rw", store_ids)

        asked = json.loads(lcm_tools.lcm_expand(
            {"node_id": node_id, "source_limit": 1, "max_tokens": 100_000}, engine=e))
        cut = json.loads(lcm_tools.lcm_expand(
            {"node_id": node_id, "max_tokens": 20}, engine=e))

        assert asked["pagination"]["complete"] is False
        assert cut["pagination"]["complete"] is False
        assert "requested" in asked["pagination"]["incomplete_reason"]
        assert "budget" not in asked["pagination"]["incomplete_reason"]
        assert "budget" in cut["pagination"]["incomplete_reason"]
    finally:
        e.shutdown()


def test_several_explicit_roots_report_the_ones_the_budget_never_reached(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("mr", platform="cli", context_length=200_000)
        roots = []
        for index in range(3):
            store_id = e._store.append(
                "mr", {"role": "user", "content": f"decision {index}: " + "detail " * 300},
                source="cli")
            roots.append(_leaf(e, "mr", [store_id], summary=f"leaf {index} " * 50))
        e._store.commit()

        captured = _capture_blocks(monkeypatch)
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "what was decided?", "node_ids": roots, "context_max_tokens": 30}))

        # one receipt names every root the budget could not reach, rather than one empty
        # block each: the count of receipts is not the contract, naming the nodes is
        unread_nodes = {
            node_id
            for b in captured["blocks"] if b.get("type") == "unread_evidence"
            for node_id in b["pagination"]["unread_node_ids"]
        }
        assert roots[-1] in unread_nodes, [
            (b.get("type"), b.get("node_id")) for b in captured["blocks"]]
        assert payload["complete"] is False, payload
        # ... and that reason is NOT the max_results one, which cannot buy a token back
        assert "requested_nodes_not_processed" not in payload, payload
    finally:
        e.shutdown()
