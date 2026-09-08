"""Fork-defined exceptions (betterlcm)."""


class SummaryUnavailableError(RuntimeError):
    """Every summariser route failed (provider error, timeout, open circuit, spend guard).

    Upstream converged such failures with deterministic truncation (L3), silently writing a
    cut-down fragment into the DAG. This fork never does that: the raw messages stay in the
    active context, the engine arms a compression-failure cooldown the host understands, and
    the turn continues uncompressed. See ``host_cooldown.py``.
    """
