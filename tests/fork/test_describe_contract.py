"""#52 — a plain lcm_describe must return a usable summary page, and every continuation
the handlers advertise must be a declared parameter of the tool it names.

`summary_max = _parse_positive_int(args.get("summary_max_chars"), 0) or 4_000` clamped to 1
before the fallback could fire, so a describe with no arguments returned ONE character of the
summary with a cursor at offset 1. Nothing was destroyed — the bytes stay in the node and the
response says it is partial — but the default is unusable, and the continuation fields the
handlers emit were not declared in the published schemas.
"""
import json
import time

from hermes_lcm import schemas, tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, **kw):
    cfg = LCMConfig(**kw)
    cfg.database_path = str(tmp_path / "describe.db")
    return LCMEngine(config=cfg, hermes_home=str(tmp_path))


def _node_with_summary(engine, summary, session="ds"):
    return engine._dag.add_node_with_meta(
        SummaryNode(session_id=session, depth=0, summary=summary, token_count=500,
                    source_token_count=2000, source_ids=[1], source_type="messages",
                    created_at=time.time()),
        level=1,
    )


LONG_SUMMARY = "decision " * 1129 + "\nExpand for details about: decisions"  # 9,036 chars


def test_a_plain_describe_returns_a_usable_summary_page(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ds", platform="cli", context_length=200_000)
        node_id = _node_with_summary(e, LONG_SUMMARY)
        payload = json.loads(lcm_tools.lcm_describe({"node_id": node_id}, engine=e))

        assert payload["summary"] == LONG_SUMMARY[:4_000], len(payload["summary"])
        assert payload["summary_chars"] == len(LONG_SUMMARY)
        assert payload["summary_offset"] == 0
        assert payload["summary_complete"] is False
        assert payload["summary_next_offset"] == 4_000
        assert payload["summary_continue_with"]["summary_offset"] == 4_000
    finally:
        e.shutdown()


def test_an_unusable_summary_length_falls_back_to_the_default(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ds", platform="cli", context_length=200_000)
        node_id = _node_with_summary(e, LONG_SUMMARY)
        for unusable in (0, -5, "not-a-number", None):
            payload = json.loads(lcm_tools.lcm_describe(
                {"node_id": node_id, "summary_max_chars": unusable}, engine=e))
            assert payload["summary"] == LONG_SUMMARY[:4_000], (unusable, len(payload["summary"]))
    finally:
        e.shutdown()


def test_an_explicit_summary_length_is_honoured(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ds", platform="cli", context_length=200_000)
        node_id = _node_with_summary(e, LONG_SUMMARY)
        payload = json.loads(lcm_tools.lcm_describe(
            {"node_id": node_id, "summary_max_chars": 120}, engine=e))
        assert payload["summary"] == LONG_SUMMARY[:120]
        assert payload["summary_next_offset"] == 120

        whole = json.loads(lcm_tools.lcm_describe(
            {"node_id": node_id, "summary_max_chars": len(LONG_SUMMARY)}, engine=e))
        assert whole["summary"] == LONG_SUMMARY
        assert whole["summary_complete"] is True
        assert "summary_next_offset" not in whole
    finally:
        e.shutdown()


def test_an_offset_at_or_past_the_end_is_empty_and_says_the_real_length(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ds", platform="cli", context_length=200_000)
        node_id = _node_with_summary(e, LONG_SUMMARY)
        for offset in (len(LONG_SUMMARY), len(LONG_SUMMARY) + 500):
            payload = json.loads(lcm_tools.lcm_describe(
                {"node_id": node_id, "summary_offset": offset}, engine=e))
            assert payload["summary"] == ""
            # an empty page must not read as "this node has no summary"
            assert payload["summary_chars"] == len(LONG_SUMMARY)
            assert payload["summary_complete"] is True
    finally:
        e.shutdown()


def test_the_whole_summary_is_reachable_by_following_the_default_pages(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ds", platform="cli", context_length=200_000)
        node_id = _node_with_summary(e, LONG_SUMMARY)
        page = json.loads(lcm_tools.lcm_describe({"node_id": node_id}, engine=e))
        collected = page["summary"]
        pages = 1
        while not page["summary_complete"]:
            assert pages < 10, "a plain describe should not need ten pages for 9k characters"
            page = json.loads(lcm_tools.lcm_describe(
                {k: v for k, v in page["summary_continue_with"].items() if k != "tool"}, engine=e))
            collected += page["summary"]
            pages += 1
        assert collected == LONG_SUMMARY
    finally:
        e.shutdown()


def test_every_continuation_the_handlers_emit_is_a_declared_tool_parameter(tmp_path):
    """A schema-driven model only ever sees the declared parameters. The handlers advertise
    summary_offset/summary_max_chars for lcm_describe and envelope_offset for lcm_expand, and
    none of the three was published."""
    declared = {
        "lcm_describe": set(schemas.LCM_DESCRIBE["parameters"]["properties"]),
        "lcm_expand": set(schemas.LCM_EXPAND["parameters"]["properties"]),
    }
    assert {"summary_offset", "summary_max_chars"} <= declared["lcm_describe"]
    assert "envelope_offset" in declared["lcm_expand"]
    # ... and in the RIGHT schema: envelope paging belongs to expansion, not to describe
    assert "envelope_offset" not in declared["lcm_describe"]
    assert "summary_max_chars" not in declared["lcm_expand"]

    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ds", platform="cli", context_length=200_000)
        child = _node_with_summary(e, LONG_SUMMARY)
        parent = e._dag.add_node_with_meta(
            SummaryNode(session_id="ds", depth=1, summary="the parent", token_count=5,
                        source_token_count=500, source_ids=[child], source_type="nodes",
                        created_at=time.time()),
            level=1,
        )
        store_id = e._store.append(
            "ds", {"role": "user", "content": "short", "reasoning_content": "z" * 20_000},
            source="cli")
        e._store.commit()
        raw_node = e._dag.add_node_with_meta(
            SummaryNode(session_id="ds", depth=0, summary="s\nExpand for details about: x",
                        token_count=5, source_token_count=50, source_ids=[store_id],
                        source_type="messages", created_at=time.time()),
            level=1,
        )

        # a truncated child summary names lcm_describe; every argument it sends must be declared
        expanded = json.loads(e.handle_tool_call("lcm_expand", {"node_id": parent, "max_tokens": 40}))
        continuation = expanded["expanded"][0]["summary_continue_with"]
        assert set(continuation) - {"tool"} <= declared[continuation["tool"]], continuation
        rest = json.loads(e.handle_tool_call(
            continuation["tool"], {k: v for k, v in continuation.items() if k != "tool"}))
        assert len(rest["summary"]) > 1

        # the same for a paged envelope, which names lcm_expand
        page = json.loads(e.handle_tool_call("lcm_expand", {"node_id": raw_node, "max_tokens": 500}))
        envelope_continuation = page["expanded"][0]["envelope_continue_with"]
        assert set(envelope_continuation) - {"tool"} <= declared[envelope_continuation["tool"]], \
            envelope_continuation
        following = json.loads(e.handle_tool_call(
            envelope_continuation["tool"],
            {k: v for k, v in envelope_continuation.items() if k != "tool"}))
        assert following["expanded"][0]["envelope_offset"] == envelope_continuation["envelope_offset"]
    finally:
        e.shutdown()
