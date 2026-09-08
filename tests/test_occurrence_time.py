from datetime import datetime, timezone

from hermes_lcm.occurrence_time import resolve_occurrence_time


def _day(value):
    return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()


def test_explicit_date_is_source_backed_and_not_observation_time():
    result = resolve_occurrence_time(
        "The launch happened on 2024-02-29.",
        observed_at=1_800_000_000,
        session_date="2026-07-19",
    )
    assert result["event_time_source"] == "explicit"
    assert _day(result["event_at"]) == "2024-02-29"
    assert result["event_at"] != result["observed_at"]
    assert result["stored_at"] == 1_800_000_000
    # fork: betterlcm — observed_at stays the host's observation timestamp; the session
    # anchor has its own field instead of overwriting it (audit p05 OT01)
    assert result["observed_at"] == 1_800_000_000
    assert _day(result["session_date_at"]) == "2026-07-19"
    assert result["support"]["quote"] == "2024-02-29"


def test_relative_occurrence_time_variants_use_session_date():
    cases = {
        "today": "2024-03-20",
        "yesterday": "2024-03-19",
        "5 days ago": "2024-03-15",
        "2 weeks ago": "2024-03-06",
        "1 month ago": "2024-02-20",
        "last monday": "2024-03-18",
    }
    for phrase, expected in cases.items():
        result = resolve_occurrence_time(
            f"It happened {phrase}.", observed_at=99, session_date="2024-03-20"
        )
        assert result["event_time_source"] == "relative_to_session"
        assert _day(result["event_at"]) == expected


def test_unknown_is_valid_without_aliasing_observation_time():
    for text, session_date in (("sometime recently", "2024-03-20"), ("yesterday", None)):
        result = resolve_occurrence_time(text, observed_at=123, session_date=session_date)
        assert result["event_time_source"] == "unknown"
        assert result["event_at"] is None
        assert result["observed_at"] == 123  # fork: p05 OT01
        assert result["session_date_at"] == (
            datetime(2024, 3, 20, tzinfo=timezone.utc).timestamp()
            if session_date
            else None
        )
        assert result["stored_at"] == 123


def test_conflicting_explicit_dates_remain_unknown():
    result = resolve_occurrence_time(
        "Either 2024-03-01 or 2024-03-02.", observed_at=123, session_date="2024-03-20"
    )
    assert result["event_time_source"] == "unknown"
    assert result["reason"] == "ambiguous_multiple_explicit_dates"


def test_explicit_and_relative_evidence_together_is_ambiguous():
    """fork: betterlcm (audit p05 OT02) — one explicit date used to win before the relative
    expressions were read, so a proposal date was recorded as the date of the cancellation."""
    result = resolve_occurrence_time(
        "On 2020-01-01 we proposed removal; yesterday we cancelled it.",
        observed_at=123,
        session_date="2024-03-20",
    )
    assert result["event_time_source"] == "unknown"
    assert result["reason"] == "ambiguous_explicit_and_relative_dates"


def test_an_absurd_relative_count_returns_unknown_instead_of_raising():
    """fork: betterlcm (audit p05 OT03) — the arithmetic raised OverflowError on source text
    and aborted the caller's whole enrichment pass."""
    result = resolve_occurrence_time(
        "It happened 999999999999 days ago.", observed_at=123, session_date="2024-03-20"
    )
    assert result["event_time_source"] == "unknown"
    assert result["reason"] == "relative_expression_out_of_range"


def test_an_explicit_date_that_agrees_with_its_relative_wording_stays_definite():
    """fork: betterlcm — only a relative expression that resolves to a DIFFERENT day conflicts;
    "Today (2026-09-08)" agrees with itself (regression found by the verify-2 auditor)."""
    result = resolve_occurrence_time(
        "Today (2026-09-08) we shipped the fix.", observed_at=123, session_date="2026-09-08"
    )
    assert result["event_time_source"] == "explicit"
    assert result["event_date"] == "2026-09-08"
