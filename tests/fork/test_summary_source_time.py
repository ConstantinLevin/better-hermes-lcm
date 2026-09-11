"""#37: the host's message timestamp reaches the summariser, and `unknown` reads as unknown.

The store keeps the host's own time in `observed_at` (`observed_at_source =
host_message_timestamp`) and its own write time in `timestamp`/`ingested_at`. The leaf
serializer excluded both, so `The meeting is tomorrow.` was offered to the model with no date
at all and two messages from different days serialised to the same bytes.
"""
import json

import pytest

from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

MILLENNIUM = 946684800.0  # 2000-01-01T00:00:00Z


@pytest.fixture()
def engine(tmp_path):
    e = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "time.db")),
                  hermes_home=str(tmp_path))
    e.on_session_start("t", platform="cli", context_length=262_144)
    try:
        yield e
    finally:
        e.shutdown()


def test_a_host_stamped_message_offers_its_source_time_to_the_summariser(engine):
    serialized = engine._serialize_messages([
        {"role": "user", "content": "The meeting is tomorrow.", "timestamp": MILLENNIUM},
    ])
    assert "2000-01-01T00:00:00Z" in serialized, serialized
    assert "source_time" in serialized, serialized


def test_two_messages_from_different_days_do_not_serialise_identically(engine):
    day_one = engine._serialize_messages([
        {"role": "user", "content": "The meeting is tomorrow.", "timestamp": MILLENNIUM},
    ])
    day_two = engine._serialize_messages([
        {"role": "user", "content": "The meeting is tomorrow.", "timestamp": MILLENNIUM + 86_400},
    ])
    assert day_one != day_two


def test_an_unknown_source_time_is_rendered_as_unknown_and_never_as_the_ingest_time(engine):
    """A stored row whose host sent no usable time: `observed_at` is NULL and `timestamp` is
    the LCM write time. Filling the gap with the write time would invent an event time."""
    serialized = engine._serialize_messages([
        {"role": "user", "content": "no host time here", "store_id": 11,
         "timestamp": MILLENNIUM, "ingested_at": MILLENNIUM,
         "observed_at": None, "observed_at_source": None},
    ])
    assert "source_time=unknown" in serialized, serialized
    assert "ingest_time=2000-01-01T00:00:00Z" in serialized, serialized
    assert "source_time=2000-01-01T00:00:00Z" not in serialized, serialized


def test_a_stored_row_offers_observed_time_as_source_and_write_time_as_ingest(engine):
    serialized = engine._serialize_messages([
        {"role": "user", "content": "recorded later", "store_id": 12,
         "timestamp": MILLENNIUM + 86_400, "ingested_at": MILLENNIUM + 86_400,
         "observed_at": MILLENNIUM, "observed_at_source": "host_message_timestamp"},
    ])
    assert "source_time=2000-01-01T00:00:00Z" in serialized, serialized
    assert "ingest_time=2000-01-02T00:00:00Z" in serialized, serialized
    assert "host_message_timestamp" in serialized, serialized


def test_a_source_time_that_cannot_be_represented_is_offered_verbatim_not_as_unknown(engine):
    """`store._envelope_extra_json` keeps such a value as `timestamp_raw` precisely because it
    is still the only record of when the turn happened."""
    serialized = engine._serialize_messages([
        {"role": "user", "content": "see you then", "timestamp": "next tuesday"},
    ])
    assert "source_time=unparsed(next tuesday)" in serialized, serialized
    assert "source_time=unknown" not in serialized, serialized


def test_an_unrepresentable_stored_source_time_is_recovered_from_the_envelope(engine):
    serialized = engine._serialize_messages([
        {"role": "user", "content": "see you then", "store_id": 13,
         "timestamp": MILLENNIUM, "ingested_at": MILLENNIUM,
         "observed_at": None, "observed_at_source": None,
         "envelope": {"timestamp_raw": "next tuesday"}},
    ])
    assert "source_time=unparsed(next tuesday)" in serialized, serialized
    assert "source_time=unknown" not in serialized, serialized


def test_default_raw_expansion_names_its_time_kinds_and_returns_the_source_time(engine):
    """`lcm_expand(store_id=…)` returned one unqualified `timestamp` — the ingest column — and
    no `observed_at` at all, so the default recovery path could not tell the two apart."""
    engine._store.append(
        "t", {"role": "user", "content": "The meeting is tomorrow.", "timestamp": MILLENNIUM},
        source="cli",
    )
    engine._store.commit()
    store_id = engine._store.get_session_tail("t", limit=1)[-1]["store_id"]

    payload = json.loads(lcm_tools.lcm_expand({"store_id": store_id}, engine=engine))
    assert payload["observed_at"] == MILLENNIUM, payload
    assert payload["observed_at_source"] == "host_message_timestamp", payload
    assert payload["timestamp_kind"] == "lcm_ingest_time", payload
    assert payload["ingested_at"] > MILLENNIUM, payload


def test_raw_expansion_says_unknown_when_the_host_sent_no_usable_time(engine):
    engine._store.append("t", {"role": "user", "content": "no time"}, source="cli")
    engine._store.commit()
    store_id = engine._store.get_session_tail("t", limit=1)[-1]["store_id"]

    payload = json.loads(lcm_tools.lcm_expand({"store_id": store_id}, engine=engine))
    assert payload["observed_at"] is None, payload
    assert payload["observed_at_source"] == "", payload
    assert payload["timestamp_kind"] == "lcm_ingest_time", payload
