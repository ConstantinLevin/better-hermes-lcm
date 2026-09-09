"""Token counting utilities for LCM.

Uses tiktoken when available, falls back to char-based estimate.
"""

import logging
import threading
import json
from functools import lru_cache
from typing import Any, Dict, List

from .message_content import normalize_content_value

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_encoder = None
_encoder_ready = False
_encoder_lock = threading.Lock()
_encoder_thread = None
# Cache-key generation, bumped (under _encoder_lock) when the real encoder is
# adopted. The memoized count is keyed on (text, generation), so an estimator
# result computed before adoption can only ever be inserted under the old
# generation: a cache_clear() alone cannot stop an in-flight LRU miss from
# repopulating the cache with a stale estimate *after* the clear, but a stale
# insert under a dead key is unreachable by post-adoption lookups.
_encoder_generation = 0

# Bound on how long a count_tokens() caller will wait for the FIRST encoder
# load. tiktoken.get_encoding() downloads the BPE file over the network when
# it is not already cached on disk; on restricted-egress hosts that request
# can hang for minutes before failing (measured: ~127s per fresh process in a
# production deployment), and _get_encoder() sits on the host's post-turn
# hook path -- so the hang blocked reply delivery. Callers that hit the bound
# use the char-based estimate; the real encoder is adopted (and the count
# cache cleared) whenever the background load completes.
_ENCODER_FIRST_WAIT_S = 2.0


def _load_encoder():
    """The (possibly network-backed) tiktoken load. Patchable in tests."""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def _encoder_loader() -> None:
    global _encoder, _encoder_ready, _encoder_generation
    enc = None
    try:
        enc = _load_encoder()
    except Exception:
        logger.debug("tiktoken not available, using char-based estimates")
    with _encoder_lock:
        _encoder = enc
        _encoder_ready = True
        if enc is not None:
            # Counts for identical text change estimator -> encoder on
            # adoption; bumping the generation retires every pre-adoption
            # cache key, including ones an in-flight miss has yet to insert.
            _encoder_generation += 1
    if enc is not None:
        # Memory hygiene only (correctness comes from the generation key):
        # evict the now-unreachable old-generation entries.
        _count_tokens_cached.cache_clear()


def _get_encoder():
    """Return the tiktoken encoder, never blocking on unbounded network I/O.

    The load runs in a daemon thread. The first caller waits briefly
    (_ENCODER_FIRST_WAIT_S) so the common already-cached-on-disk case still
    gets exact counts immediately; after that, callers never wait -- they use
    the estimator until the loader finishes.
    """
    global _encoder_thread
    if _encoder_ready:
        return _encoder
    first = False
    with _encoder_lock:
        if _encoder_ready:
            return _encoder
        if _encoder_thread is None:
            _encoder_thread = threading.Thread(
                target=_encoder_loader, name="lcm-tiktoken-load", daemon=True
            )
            _encoder_thread.start()
            first = True
    if first:
        _encoder_thread.join(timeout=_ENCODER_FIRST_WAIT_S)
    return _encoder if _encoder_ready else None


# fork: betterlcm — per-character cost for the fallback estimate, in tokens per character.
# Upstream picked ONE divisor from the proportion of non-ASCII characters, which both
# undercounted dense scripts (100 CJK characters ≈ 150 tokens in practice, estimated as 67)
# and stepped discontinuously at the ratio boundaries: 49 CJK + 51 ASCII estimated 41 tokens,
# 50 + 50 estimated 67 (audit E, E19). Weights are additive, so composition is monotone and
# the estimate moves smoothly. This path only runs when the real tokenizer is unavailable; it
# is an estimate for budget decisions, not a claim about any particular model's tokenizer.
_TOKEN_COST_ASCII = 1.0 / _CHARS_PER_TOKEN
_TOKEN_COST_LATIN_EXT = 0.5      # accented Latin, Greek, Cyrillic, punctuation
_TOKEN_COST_DENSE_SCRIPT = 1.5   # CJK, kana, hangul — roughly one token per character or more
_TOKEN_COST_SYMBOL = 2.0         # emoji and other astral symbols, usually multi-token
_TOKEN_COST_OTHER = 1.0


def _character_token_cost(code_point: int) -> float:
    if code_point < 128:
        return _TOKEN_COST_ASCII
    if code_point < 0x0900:
        return _TOKEN_COST_LATIN_EXT
    if 0x2E80 <= code_point <= 0x9FFF or 0xA960 <= code_point <= 0xD7FF:
        return _TOKEN_COST_DENSE_SCRIPT
    if 0xF900 <= code_point <= 0xFAFF or 0xFE30 <= code_point <= 0xFE4F:
        return _TOKEN_COST_DENSE_SCRIPT
    if 0xFF00 <= code_point <= 0xFFEF:
        return _TOKEN_COST_DENSE_SCRIPT
    if 0x20000 <= code_point <= 0x3FFFF:
        return _TOKEN_COST_DENSE_SCRIPT
    if 0x1F000 <= code_point <= 0x1FBFF or 0x2600 <= code_point <= 0x27BF:
        return _TOKEN_COST_SYMBOL
    return _TOKEN_COST_OTHER


def _fallback_token_estimate(text: str) -> int:
    # Latin text is ~4 chars/token, but CJK and other non-Latin scripts tokenize far denser
    # (~1-2 tokens/char) and emoji denser still. A flat len//4 undercounts them ~3-4x, so
    # preflight under-triggers and assembly can overflow the real budget. ASCII-only text is
    # overwhelmingly common and keeps the cheap legacy estimate, byte for byte.
    length = len(text)
    if length == 0:
        return 0
    if text.isascii():
        return length // _CHARS_PER_TOKEN + 1
    total = 0.0
    for character in text:
        total += _character_token_cost(ord(character))
    return int(total) + 1


def _serialize_for_count(value) -> str:
    """fork: betterlcm — render a non-string value the way a provider would receive it.

    ``len(value) // 4`` on a dict counts its KEYS: a tool call whose ``arguments`` arrive as a
    dict of one 50,000-character command was estimated at ~1 token, and the whole message at
    11 tokens. Every downstream decision — compaction pressure, the fresh-tail token cap, chunk
    sizing, the assembly budget — is derived from these counts, so the undercount silently
    breaks the bounded-prompt guarantee. Serialise instead, and fall back to ``str`` for values
    JSON cannot represent.
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return str(value)


def _count_tokens_core(text) -> int:
    if not isinstance(text, str):
        text = _serialize_for_count(text)  # fork: betterlcm
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return _fallback_token_estimate(text)


def _count_tokens_keyed(text: str, generation: int) -> int:
    # tiktoken encoding is the dominant per-turn cost: assembly and preflight
    # re-count the same content many times per turn. The counting function is
    # stable within one encoder generation (see _encoder_generation), so a
    # (content, generation) key is stable.
    return _count_tokens_core(text)


DEFAULT_TOKEN_CACHE_SIZE = 2048
_count_tokens_cached = lru_cache(maxsize=DEFAULT_TOKEN_CACHE_SIZE)(_count_tokens_keyed)


_token_cache_requests: dict[int, int] = {}
_token_cache_owner_refs: dict[int, Any] = {}


def _forget_token_cache_owner(owner_id: int) -> None:
    """Drop a dead engine's request and re-apply the remaining maximum."""
    _token_cache_requests.pop(owner_id, None)
    _token_cache_owner_refs.pop(owner_id, None)
    set_token_cache_size(DEFAULT_TOKEN_CACHE_SIZE)


def set_token_cache_size(maxsize: int, *, owner: Any = None) -> None:
    """fork: betterlcm — resize the memo (window-weighted: 2048 at 256k, 8192 at 1M).

    The cache is process-global while engines are not, so the size is the MAXIMUM any live
    engine asked for: a 256k clone used to shrink the cache a 1M engine had just grown, and
    the rebuild emptied it — throwing away the other engine's work every time a second engine
    bound a session (verify-3 O8). ``owner`` identifies the requester so its request can be
    replaced rather than accumulated.
    """
    global _count_tokens_cached
    size = max(64, int(maxsize or DEFAULT_TOKEN_CACHE_SIZE))
    if owner is not None:
        owner_id = id(owner)
        _token_cache_requests[owner_id] = size
        if owner_id not in _token_cache_owner_refs:
            try:
                import weakref
                _token_cache_owner_refs[owner_id] = weakref.finalize(
                    owner, _forget_token_cache_owner, owner_id
                )
            except TypeError:  # pragma: no cover - not weak-referenceable
                _token_cache_owner_refs[owner_id] = None
    if _token_cache_requests:
        # fork: betterlcm — the maximum of what the LIVE engines asked for. Flooring it at the
        # default meant an explicit smaller override (a 64-entry cache on a memory-tight host)
        # silently became 2048 (round-2 verify-3 #26); with no live request the default stands.
        size = max(_token_cache_requests.values())
    if _count_tokens_cached.cache_info().maxsize == size:
        return
    _count_tokens_cached = lru_cache(maxsize=size)(_count_tokens_keyed)


# Cap what the LRU may retain by reference. Very large strings are the ones
# least likely to recur identically, and caching them would still let the
# bounded LRU pin unnecessary memory; count those uncached (cost is
# proportional to size either way).
_MAX_CACHEABLE_TOKEN_TEXT_CHARS = 32_768


def count_tokens(text) -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    # Only strings are memoized. Callers may pass non-string, unhashable values
    # (e.g. tool_call arguments as a dict); preserve the legacy tolerance by
    # counting those uncached rather than feeding them to the LRU.
    if isinstance(text, str) and len(text) <= _MAX_CACHEABLE_TOKEN_TEXT_CHARS:
        return _count_tokens_cached(text, _encoder_generation)
    return _count_tokens_core(text)


def count_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single OpenAI-format message."""
    total = 4  # role + overhead
    content = normalize_content_value(msg.get("content")) or ""
    total += count_tokens(content)
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            total += count_tokens(fn.get("name", ""))
            total += count_tokens(fn.get("arguments", ""))
        total += 3  # per-call overhead
    return total


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a message list."""
    return sum(count_message_tokens(m) for m in messages)
