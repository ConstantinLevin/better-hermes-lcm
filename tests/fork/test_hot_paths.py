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


def test_node_decoding_does_not_depend_on_physical_column_order(tmp_path):
    """Audit p04 DG1: reads used `SELECT *` and decoded positionally, so a database whose
    columns were added in a different order — an older build, a restored backup, a future
    migration — silently decoded expand_hint as summary, with no error anywhere."""
    import sqlite3
    import time as _time
    from hermes_lcm.dag import SummaryDAG, SummaryNode
    path = tmp_path / "reordered.db"
    dag = SummaryDAG(path)
    node_id = dag.add_node(SummaryNode(session_id="s", depth=1, summary="THE SUMMARY",
                                       token_count=7, source_token_count=9, source_ids=[3],
                                       source_type="messages", created_at=_time.time(),
                                       expand_hint="THE HINT"))
    dag.close()

    # append a column, exactly as a later migration would: physical order now differs
    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE summary_nodes ADD COLUMN a_future_column TEXT DEFAULT 'x'")
    conn.commit()
    conn.close()

    dag = SummaryDAG(path)
    try:
        node = dag.get_node(node_id)
        assert node.summary == "THE SUMMARY"
        assert node.expand_hint == "THE HINT"
        assert node.source_ids == [3] and node.depth == 1 and node.token_count == 7
        assert dag.get_session_nodes("s")[0].summary == "THE SUMMARY"
    finally:
        dag.close()


def test_a_second_engine_cannot_shrink_the_shared_token_cache():
    """verify-3 O8: the memo is process-global while engines are not, so a 256k clone shrank
    (and emptied) the cache a 1M engine had just grown."""
    from hermes_lcm import tokens as tokens_module

    original = tokens_module._count_tokens_cached
    requests = dict(tokens_module._token_cache_requests)
    try:
        tokens_module._token_cache_requests.clear()
        class _Owner:
            pass

        big, small = _Owner(), _Owner()
        tokens_module.set_token_cache_size(8192, owner=big)
        assert tokens_module._count_tokens_cached.cache_info().maxsize == 8192
        tokens_module.set_token_cache_size(2048, owner=small)
        assert tokens_module._count_tokens_cached.cache_info().maxsize == 8192, \
            "the smaller request must not shrink the shared cache"
        tokens_module.set_token_cache_size(2048, owner=big)  # the big engine rebound smaller
        assert tokens_module._count_tokens_cached.cache_info().maxsize == 2048
    finally:
        tokens_module._token_cache_requests.clear()
        tokens_module._token_cache_requests.update(requests)
        tokens_module._count_tokens_cached = original


def test_an_unchanged_prefix_costs_no_revision_reads(tmp_path):
    """round-3 verify-2 #10: every ingest re-read all already-ingested host ids — full content,
    calls and envelope — and the prefix is unchanged on almost every turn."""
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    cfg = LCMConfig(database_path=str(tmp_path / "prefixperf.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("pp", platform="cli", context_length=200_000)
        messages = [
            {"role": "user", "content": f"turn {index} " + "word " * 20, "message_id": f"m{index}"}
            for index in range(50)
        ]
        e._ingest_messages(messages)
        e._store.commit()

        reads = {"n": 0}
        real = e._store.latest_rows_by_host_message_id

        def counted(*args, **kwargs):
            reads["n"] += 1
            return real(*args, **kwargs)

        e._store.latest_rows_by_host_message_id = counted
        e._ingest_messages(messages)          # the first snapshot after ingest: one lookup
        assert reads["n"] == 1
        for _ in range(5):
            e._ingest_messages(list(messages))  # unchanged: no database work at all
        assert reads["n"] == 1, f"{reads['n']} revision queries on an unchanged prefix"

        edited = [dict(message) for message in messages]
        edited[3] = dict(edited[3], content="CORRECTED")
        e._ingest_messages(edited)
        assert reads["n"] == 2, "a changed prefix is looked up once more"
    finally:
        e.shutdown()
