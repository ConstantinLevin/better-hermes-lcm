"""Fork-defined exceptions (better-hermes-lcm)."""


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


class GenerationNotTerminatedError(SummaryUnavailableError):
    """The route returned text without positive evidence that the generation finished.

    A subclass, not a sibling, so every caller that already treats "the summariser was
    unavailable" as fail-closed — the leaf rescue, the condensation guard, the host cooldown —
    treats an unterminated generation exactly the same way, with no new handling anywhere.
    The difference is only in what the message says, which is what the rescue predicate and the
    operator read. See ``generation_contract.py`` for what counts as terminal evidence.
    """
