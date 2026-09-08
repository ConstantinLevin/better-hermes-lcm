"""Step 12 — lcm_doctor coverage and the suggester's window-scaled branch."""
import json
import time

from hermes_lcm import coverage_doctor, presets
from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def test_extract_index_entities_finds_the_things_a_reader_would_search_for():
    text = (
        "We decided to use sqlite instead of postgres. See dag.py and /home/agent/lcm.db; "
        "call compile_message_patterns, port 3456, url https://example.com/x?y=1, "
        "the flag 'no-loss' and BuildTurnContext failed with error 502."
    )
    ents = coverage_doctor.extract_index_entities(text)
    for expected in ("dag.py", "/home/agent/lcm.db", "compile_message_patterns", "3456",
                     "https://example.com/x?y=1", "no-loss", "BuildTurnContext", "decided", "instead of", "error"):
        assert expected in ents, expected


def test_coverage_of_scores_presence_in_summary_plus_index_block():
    sources = "decided on sqlite; edited dag.py and node_meta.py; port 3456"
    full = coverage_doctor.coverage_of("decided sqlite; dag.py, node_meta.py; port 3456", "", sources)
    assert full["fraction"] == 1.0
    reworded = coverage_doctor.coverage_of("sqlite chosen; dag.py, node_meta.py; port 3456", "", sources)
    assert reworded["fraction"] == 0.8 and reworded["missing_sample"] == ["decided"]
    thin = coverage_doctor.coverage_of("some database work", "", sources)
    assert thin["fraction"] < 0.5 and "dag.py" in thin["missing_sample"]
    via_index = coverage_doctor.coverage_of("db work", "Expand: dag.py node_meta.py 3456 decided sqlite", sources)
    assert via_index["fraction"] == 1.0


def _engine(tmp_path):
    cfg = LCMConfig(incremental_max_depth=0)
    cfg.database_path = str(tmp_path / "cov.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start("cov", platform="cli", context_length=200_000)
    return e


def test_doctor_coverage_reports_nodes_under_the_floor(tmp_path):
    e = _engine(tmp_path)
    try:
        sid1 = e._store.append("cov", {"role": "user", "content": "edit dag.py and node_meta.py; port 3456; decided sqlite"}, source="cli")
        sid2 = e._store.append("cov", {"role": "user", "content": "run compile_message_patterns on config.yaml at 09:41; error 502 fixed"}, source="cli")
        e._store._conn.commit()
        good = e._dag.add_node(SummaryNode(
            session_id="cov", depth=0, summary="dag.py + node_meta.py edited; port 3456; decided sqlite",
            token_count=12, source_token_count=20, source_ids=[sid1], source_type="messages", created_at=time.time(),
        ))
        e._dag.node_meta.write(good, level=1, summary="")
        bad = e._dag.add_node(SummaryNode(
            session_id="cov", depth=0, summary="did some config work",
            token_count=5, source_token_count=20, source_ids=[sid2], source_type="messages", created_at=time.time(),
        ))
        e._dag.node_meta.write(bad, level=2, summary="")
        payload = json.loads(lcm_tools.lcm_doctor({"coverage": True}, engine=e))
        report = payload["coverage"]
        assert report["scored_nodes"] == 2
        below = {n["node_id"]: n for n in report["nodes_below_floor"]}
        assert bad in below and good not in below
        assert below[bad]["level"] == 2
        assert "compile_message_patterns" in below[bad]["missing_sample"]
        check = next(c for c in payload["checks"] if c["check"] == "index_coverage")
        assert check["status"] == "warn"
        # without the flag the doctor payload is upstream's
        plain = json.loads(lcm_tools.lcm_doctor({}, engine=e))
        assert "coverage" not in plain
        assert all(c["check"] != "index_coverage" for c in plain["checks"])
        # the slash command renders it
        from hermes_lcm import command as lcm_command
        text = lcm_command._doctor_coverage_text(e)
        assert "index coverage" in text and f"node {bad}" in text
    finally:
        e.shutdown()


def test_depth_one_coverage_reads_child_summaries(tmp_path):
    e = _engine(tmp_path)
    try:
        c1 = e._dag.add_node(SummaryNode(session_id="cov", depth=0, summary="rewrote escalation.py; decided L3 removal",
                                         token_count=5, source_token_count=5, source_ids=[], source_type="messages", created_at=time.time()))
        parent = e._dag.add_node(SummaryNode(session_id="cov", depth=1, summary="escalation.py rewritten, L3 removal decided",
                                             token_count=5, source_token_count=5, source_ids=[c1], source_type="nodes", created_at=time.time()))
        cov = coverage_doctor.node_coverage(e, e._dag.get_node(parent))
        assert cov["fraction"] == 1.0
    finally:
        e.shutdown()


class _Eng:
    def __init__(self, window, low=262_144, high=1_000_000):
        self.context_length = window
        self._config = type("C", (), {"scale_low_window": low, "scale_high_window": high})()


def test_suggester_hands_large_windows_to_the_curve():
    preset, reason = presets.suggest_preset_for_engine(_Eng(1_000_000))
    assert preset is None and "window-scaled defaults active (t=1.00)" in reason
    preset, reason = presets.suggest_preset_for_engine(_Eng(600_000))
    assert preset is None and "t=0.46" in reason
    # below 512k upstream's branches are untouched
    preset, _ = presets.suggest_preset_for_engine(_Eng(262_144))
    assert preset is not None and preset.name == presets._CODEX_GPT_LONG_CONTEXT.name
