"""Step 9a — hot paths give identical results; SQLite/LRU sizes follow the curve."""
import time

from hermes_lcm import tokens as tokens_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, window):
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / f"hot-{window}.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start(f"h{window}", platform="cli", context_length=window)
    return e


def test_frontier_tokens_projection_equals_full_decode(tmp_path):
    e = _engine(tmp_path, 262_144)
    try:
        base = time.time()
        ids = [e._dag.add_node(SummaryNode(
            session_id=e._session_id, depth=0, summary=f"l{i}", token_count=7 + i,
            source_token_count=1, source_ids=[], source_type="messages", created_at=base + i,
        )) for i in range(6)]
        e._dag.add_node(SummaryNode(
            session_id=e._session_id, depth=1, summary="c", token_count=100,
            source_token_count=1, source_ids=ids[:4], source_type="nodes", created_at=base + 10,
        ))
        # another session must not leak in
        e._dag.add_node(SummaryNode(
            session_id="other", depth=0, summary="x", token_count=999,
            source_token_count=1, source_ids=[], source_type="messages", created_at=base,
        ))
        full = sum(n.token_count for n in e._summary_frontier_nodes())
        assert full == (7 + 4) + (7 + 5) + 100
        assert e._dag.get_frontier_token_total(e._session_id) == full
        assert e._summary_frontier_tokens() == full
    finally:
        e.shutdown()


def test_sqlite_cache_and_token_cache_follow_the_curve(tmp_path):
    e = _engine(tmp_path, 262_144)
    try:
        assert int(e.effective_sqlite_cache_kib) == 2048
        assert e._store.connection.execute("PRAGMA cache_size").fetchone()[0] == -2048
        assert tokens_mod._count_tokens_cached.cache_info().maxsize == 2048
        e._set_context_length(1_000_000, source="test")
        assert int(e.effective_sqlite_cache_kib) == 65_536
        assert e._store.connection.execute("PRAGMA cache_size").fetchone()[0] == -65_536
        assert e._dag.connection.execute("PRAGMA cache_size").fetchone()[0] == -65_536
        assert tokens_mod._count_tokens_cached.cache_info().maxsize == 8192
        assert tokens_mod.count_tokens("hello world") == tokens_mod.count_tokens("hello world")
        e._set_context_length(262_144, source="test")
        assert tokens_mod._count_tokens_cached.cache_info().maxsize == 2048
    finally:
        e.shutdown()


def test_non_string_values_are_counted_as_serialized(tmp_path):
    """Audit A W4: `len(value)//4` on a dict counts keys, not content.

    Hosts may hand tool-call arguments through as a dict. Counting them by key count made a
    50,000-character call cost ~1 token, so every pressure/tail/chunk/assembly decision derived
    from it was wrong by three orders of magnitude.
    """
    import json as _json
    from hermes_lcm.tokens import count_message_tokens, count_tokens
    args = {"command": "x" * 50_000}
    assert count_tokens(args) == count_tokens(_json.dumps(args, ensure_ascii=False, sort_keys=True))
    assert count_tokens(args) > 10_000
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "terminal", "arguments": args}}]}
    assert count_message_tokens(msg) > 10_000
    # a string argument is unchanged, and both shapes now agree
    string_msg = {"role": "assistant", "content": "",
                  "tool_calls": [{"id": "c1", "type": "function",
                                  "function": {"name": "terminal",
                                               "arguments": _json.dumps(args, ensure_ascii=False, sort_keys=True)}}]}
    assert count_message_tokens(msg) == count_message_tokens(string_msg)
    # values JSON cannot represent still get a real estimate rather than a key count
    assert count_tokens({"o": object()}) > 0


def test_like_fallback_orders_before_limiting(tmp_path):
    """Audit p04/B10: the LIKE path (used for CJK/emoji queries by design, not only a broken
    FTS index) paged an unordered candidate set and sorted the page, so `sort="recency",
    limit=1` returned the newest of an arbitrary page rather than the newest match."""
    import time as _time
    from hermes_lcm.dag import SummaryDAG, SummaryNode
    dag = SummaryDAG(tmp_path / "like.db")
    try:
        base = _time.time()
        newest = None
        for i in range(300):
            newest = dag.add_node(SummaryNode(
                session_id="s", depth=0, summary=f"部署 note {i}", token_count=5,
                source_token_count=9, source_ids=[], source_type="messages", created_at=base + i))
        top = dag.search("部署", session_id="s", limit=1, sort="recency")
        assert [n.node_id for n in top] == [newest]
        top3 = dag.search("部署", session_id="s", limit=3, sort="recency")
        assert [n.node_id for n in top3] == [newest, newest - 1, newest - 2]
    finally:
        dag.close()
