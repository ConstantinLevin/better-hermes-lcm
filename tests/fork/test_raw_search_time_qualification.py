"""#37, third reader — a raw search hit must not let an ingest time read as an event time.

`_collect_raw_match_context_block` put the messages table's `timestamp` column into the
synthesis context under the bare name `timestamp`. That column is written as `ingested_at`
(store.py: both are the same `time.time()` at append), so a model reasoning over the retrieved
context read "when LCM first saw this row" as "when this happened". The host's own message time
lives in `observed_at`, and it is NULL when the host recorded none — which must stay visible as
unknown rather than be filled in from the ingest clock.

Same four key names the store_id branch of lcm_expand uses, deliberately: one vocabulary for
one distinction.
"""
import json
import time

from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, **kw):
    cfg = LCMConfig(**kw)
    cfg.database_path = str(tmp_path / "rawtime.db")
    return LCMEngine(config=cfg, hermes_home=str(tmp_path))


def _raw_hit(engine, monkeypatch, query="auditneedle"):
    captured = {}

    def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
        captured["blocks"] = context_blocks
        return "an answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
    json.loads(engine.handle_tool_call(
        "lcm_expand_query", {"prompt": "when did it happen?", "query": query}))
    raw = next(b for b in captured["blocks"] if b["type"] == "raw_messages")
    return raw["messages"][0]


def test_a_raw_search_hit_separates_when_it_happened_from_when_it_was_stored(tmp_path, monkeypatch):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("rt", platform="cli", context_length=200_000)
        happened_at = time.time() - 86_400 * 30  # the host says: a month ago
        e._store.append("rt", {
            "role": "user",
            "content": "auditneedle: the deployment was cancelled",
            "timestamp": happened_at,
        }, source="cli")
        e._store.commit()

        item = _raw_hit(e, monkeypatch)
        assert item["timestamp_kind"] == "lcm_ingest_time"
        assert item["observed_at"] == happened_at
        assert item["observed_at_source"] == "host_message_timestamp"
        # the ingest clock is a different fact, and it is today
        assert item["ingested_at"] > happened_at + 86_400
        assert item["timestamp"] == item["ingested_at"]
    finally:
        e.shutdown()


def test_a_raw_search_hit_leaves_an_unrecorded_event_time_unknown(tmp_path, monkeypatch):
    """Unknown stays unknown: the key is present and null, never filled in from the ingest
    clock, because a substituted time is indistinguishable from a recorded one."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        e.on_session_start("ru", platform="cli", context_length=200_000)
        e._store.append("ru", {
            "role": "user",
            "content": "auditneedle: the deployment was cancelled",
        }, source="cli")
        e._store.commit()

        item = _raw_hit(e, monkeypatch)
        assert item["timestamp_kind"] == "lcm_ingest_time"
        assert "observed_at" in item, "an absent key reads as 'not applicable', not 'unknown'"
        assert item["observed_at"] is None
        assert item["observed_at_source"] is None
        assert item["ingested_at"] > 0
    finally:
        e.shutdown()
