"""What counts as a FINISHED generation at the host boundary (better-hermes-lcm).

The plugin never sees a provider; it sees whatever the host's adapter hands back, and the
adapters fabricate terminal states. The chat-stream accumulator turns a stream that ended
without a terminal chunk into ``finish_reason="stop"``; the Codex Responses adapter projects
``response.incomplete`` and ``response.failed`` into a chat choice with ``stop`` and drops the
status, the incomplete details and the error object on the way; the Bedrock and Anthropic
normalizers map an unknown or missing stop reason to ``stop``. A negative list — "refuse the
known cut reasons" — therefore certifies exactly the responses that carry no evidence at all.

So the test here is positive: a generation is finished only when the response SAYS so, in a
recognised way, and nothing in it contradicts that. Absent evidence, an unrecognised terminal
state, a failure or non-terminal status, an error object, and a completion that consumed the
whole requested cap while claiming to have stopped on its own are all refusals.

What this cannot see, and what no code in this plugin can: a host that fabricates one of the
recognised terminal states. Closing that needs the adapter to carry the provider's own
terminal event through unaltered — a host contract, not a plugin check. Until it does, the
refusals below are the part of the contract this side can enforce.

Refusing is safe: every consumer treats it as "the summariser was unavailable", which in this
fork means the sources stay raw and nothing is published.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

# reasons that mean "the model was cut off", not "the model finished".
TRUNCATED_FINISH_REASONS = frozenset({
    "length", "max_tokens", "max_output_tokens", "content_filter", "incomplete",
})

# reasons that positively declare a finished generation. Anything outside both sets is
# unrecognised, and an unrecognised state is not evidence of termination.
TERMINAL_FINISH_REASONS = frozenset({
    "stop", "end_turn", "stop_sequence", "tool_calls", "function_call", "completed", "complete",
})

# provider statuses that declare a finished response. A response that carries a status at all
# must carry one of these; "failed", "cancelled", "expired", "in_progress" and anything
# unrecognised are refusals.
TERMINAL_STATUSES = frozenset({"completed", "complete", "succeeded", "success", "finished", "ok"})


class GenerationOutcome(NamedTuple):
    """``reason`` is "" for a positively terminal generation.

    ``cut`` separates "the route stopped at a generation limit and the text it produced is a
    genuine prefix of the answer" from "this response is not evidence of a finished
    generation at all". Both are refused where a node would be published; a consumer that may
    keep a labelled partial (the query synthesis) needs to tell them apart.
    """

    reason: str
    cut: bool = False


def _read(obj: Any, name: str) -> Any:
    """Attribute or mapping key.

    The host's generic shape recovery rebuilds a response as a plain dict, and the status it
    was carrying then reaches us as ``response["status"]`` rather than ``response.status``.
    Reading only attributes accepted every failed generation that took that path.
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _first_choice(response: Any) -> Any:
    choices = _read(response, "choices")
    if not choices:
        return None
    try:
        return choices[0]
    except (TypeError, IndexError, KeyError):
        return None


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _completion_tokens(response: Any) -> int:
    usage = _read(response, "usage")
    for name in ("completion_tokens", "output_tokens"):
        value = _read(usage, name)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value
    return 0


def evaluate_generation(response: Any, *, requested_max_tokens: int = 0) -> GenerationOutcome:
    """Why this response is not a finished generation, or ``GenerationOutcome("")``."""
    choice = _first_choice(response)
    if choice is None:
        return GenerationOutcome("the response carried no choices")

    error = _read(response, "error") or _read(choice, "error")
    if error:
        return GenerationOutcome(f"the route reported an error ({str(error)[:160]})")

    details = _read(response, "incomplete_details")
    status = _text(_read(response, "status"))
    if details:
        return GenerationOutcome(
            f"status={status or 'incomplete'}, incomplete_details={str(details)[:160]}"
        )
    if status and status not in TERMINAL_STATUSES:
        return GenerationOutcome(f"status={status}")

    finish_reason = _text(_read(choice, "finish_reason"))
    if finish_reason in TRUNCATED_FINISH_REASONS:
        return GenerationOutcome(f"finish_reason={finish_reason}", cut=True)
    if not finish_reason:
        if status in TERMINAL_STATUSES:
            return GenerationOutcome("")
        return GenerationOutcome(
            "no terminal evidence: the response carried neither a finish_reason nor a "
            "terminal provider status"
        )
    if finish_reason not in TERMINAL_FINISH_REASONS:
        return GenerationOutcome(
            f"finish_reason={finish_reason} is not a recognised terminal state"
        )

    # the response's own accounting can contradict its terminal claim: a generation that used
    # every token it was given did not choose to stop, whatever the finish_reason says.
    completion_tokens = _completion_tokens(response)
    if requested_max_tokens > 0 and completion_tokens >= requested_max_tokens:
        return GenerationOutcome(
            f"finish_reason={finish_reason} but the response used its whole generation cap "
            f"(completion_tokens={completion_tokens} of max_tokens={requested_max_tokens})",
            cut=True,
        )
    return GenerationOutcome("")


def unterminated_generation_reason(response: Any, *, requested_max_tokens: int = 0) -> str:
    return evaluate_generation(response, requested_max_tokens=requested_max_tokens).reason


def generation_text(response: Any) -> str:
    """The assistant text of a response, read across the shapes the adapters produce."""
    choice = _first_choice(response)
    message = _read(choice, "message")
    content = _read(message, "content")
    if isinstance(content, str):
        return content
    return str(content) if content else ""
