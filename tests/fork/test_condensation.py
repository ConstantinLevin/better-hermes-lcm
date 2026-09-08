"""Step 9 — interpolated condensation trigger, oldest-first selection, ratios from config."""
import time

import pytest

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

W256 = 262_144
W1M = 1_000_000


def _engine(tmp_path, window=None, **kw):
    cfg = LCMConfig(**{"condensation_fanin": 4, "incremental_max_depth": 3, **kw})
    cfg.database_path = str(tmp_path / f"cond-{window}.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    if window:
        e.on_session_start(f"c{window}", platform="cli", context_length=window)
    else:
        e._session_id = "c-nowindow"
    return e


def _leaf(e, tokens, *, earliest, created=None, summary=None):
    return e._dag.add_node(SummaryNode(
        session_id=e._session_id, depth=0,
        summary=summary or f"leaf {earliest}", token_count=tokens, source_token_count=tokens * 5,
        source_ids=[], source_type="messages",
        created_at=created if created is not None else time.time(),
        earliest_at=earliest, latest_at=earliest + 1,
    ))


@pytest.fixture
def mock_summariser(monkeypatch):
    calls = []

    def fake(prompt, max_tokens, model="", timeout=None):
        calls.append(max_tokens)
        return "condensed\nExpand for details about: mock"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", fake)
    return calls


@pytest.mark.parametrize("window", [None, W256])
def test_condense_at_256k_uses_count_rule(tmp_path, mock_summariser, window):
    e = _engine(tmp_path, window)
    try:
        assert int(e.effective_condense_budget_tokens or 0) == 0
        base = time.time()
        for i in range(4):
            _leaf(e, 10, earliest=base + i, created=base + i)
        e._maybe_condense()
        nodes = e._dag.get_session_nodes(e._session_id)
        assert sorted(n.depth for n in nodes) == [0, 0, 0, 0, 1]
    finally:
        e.shutdown()


def test_condense_at_1m_requires_budget(tmp_path, mock_summariser):
    e = _engine(tmp_path, W1M)
    try:
        assert int(e.effective_condense_budget_tokens) == 200_000
        base = time.time()
        for i in range(8):  # two full fanin groups, tiny pile
            _leaf(e, 1000, earliest=base + i, created=base + i)
        e._maybe_condense()
        assert all(n.depth == 0 for n in e._dag.get_session_nodes(e._session_id))
        assert e._last_condensation_suppressed_reason == "frontier_within_budget"
        assert mock_summariser == []
    finally:
        e.shutdown()


def test_condense_at_1m_over_budget_condenses_oldest_first_until_under(tmp_path, mock_summariser):
    e = _engine(tmp_path, W1M)
    try:
        base = time.time()
        # 6 leaves of 60k = 360k > 200k. created_at is deliberately the REVERSE of earliest_at
        # so the test proves selection follows earliest_at (content age), not insertion order.
        ids = []
        for i in range(6):
            ids.append(_leaf(e, 60_000, earliest=base + i, created=base + (10 - i)))
        e._maybe_condense()
        nodes = e._dag.get_session_nodes(e._session_id)
        d1 = [n for n in nodes if n.depth == 1]
        assert len(d1) == 1  # one group brought the pile to 120k + summary <= 200k
        assert sorted(d1[0].source_ids) == sorted(ids[:4])  # the four OLDEST by earliest_at
        assert e._dag.node_meta.read(d1[0].node_id)["level"] == 1
        assert e._summary_frontier_tokens() <= 200_000
    finally:
        e.shutdown()


def test_condense_over_budget_uses_upstreams_loop_not_a_drain(tmp_path, mock_summariser):
    """The budget is a GATE in front of upstream's loop, not a replacement for it.

    An earlier version drained the frontier until it was under budget; with the tiny budget the
    curve yields just above 256k that condensed everything on every compaction. Upstream's loop
    does one group per depth per call, and that is what must happen once the gate opens.
    """
    e = _engine(tmp_path, W1M)
    try:
        base = time.time()
        for i in range(12):
            _leaf(e, 60_000, earliest=base + i, created=base + i)
        e._maybe_condense()
        d1 = [n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 1]
        assert len(d1) == 1
        assert sorted(d1[0].source_ids) == sorted(n.node_id for n in
                                                  sorted(e._dag.get_session_nodes(e._session_id),
                                                         key=lambda n: n.node_id)[:4])
    finally:
        e.shutdown()


def test_condense_budget_regime_skips_depths_at_the_cap(tmp_path, mock_summariser):
    e = _engine(tmp_path, W1M, incremental_max_depth=1)
    try:
        base = time.time()
        for i in range(4):
            e._dag.add_node(SummaryNode(
                session_id=e._session_id, depth=1, summary=f"d1 {i}", token_count=80_000,
                source_token_count=1, source_ids=[], source_type="nodes",
                created_at=base + i, earliest_at=base + i, latest_at=base + i,
            ))
        e._maybe_condense()
        assert all(n.depth == 1 for n in e._dag.get_session_nodes(e._session_id))
    finally:
        e.shutdown()


def test_summary_size_rules_come_from_config(tmp_path, mock_summariser):
    e = _engine(
        tmp_path, W256,
        leaf_summary_min_tokens=50, leaf_summary_max_tokens=80, leaf_summary_ratio=0.5,
        condensation_min_tokens=30, condensation_ratio=0.1,
        fresh_tail_count=1, leaf_chunk_tokens=1,
    )
    try:
        e.threshold_tokens = 1
        e.compress([{"role": "user", "content": "word " * 300}, {"role": "user", "content": "tail"}])
        # leaf: clamp(0.5*~300, 50, 80) = 80 -> summariser max_tokens = 2*80
        assert mock_summariser[0] == 160
        base = time.time()
        for i in range(4):
            _leaf(e, 100, earliest=base + i, created=base + i)
        mock_summariser.clear()
        e._maybe_condense()
        # condensation: max(30, 0.1*source) -> max_tokens = 2*budget; source is the summaries' text
        assert mock_summariser and mock_summariser[0] >= 60
    finally:
        e.shutdown()
