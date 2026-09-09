"""Fork-defined exceptions (better-hermeslcm)."""


class ExtractionUnavailableError(RuntimeError):
    """The pre-compaction extraction call did not happen or did not finish.

    Distinct from the model answering ``NOTHING_TO_EXTRACT``: upstream returned ``None`` for
    both, so a provider outage was recorded as a successful "nothing worth extracting"
    (audit p05 EX08). Extraction stays non-blocking — the caller logs and moves on — but the
    two outcomes are no longer the same value.
    """


class SummaryUnavailableError(RuntimeError):
    """Every summariser route failed (provider error, timeout, open circuit, spend guard).

    Upstream converged such failures with deterministic truncation (L3), silently writing a
    cut-down fragment into the DAG. This fork never does that: the raw messages stay in the
    active context, the engine arms a compression-failure cooldown the host understands, and
    the turn continues uncompressed. See ``host_cooldown.py``.
    """
