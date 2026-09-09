"""Helpers for search query handling across FTS and LIKE fallback paths."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, List

# fork: betterlcm — these ranges decide whether a query goes to the LIKE scan instead of the
# FTS index, and unicode61 indexes neither CJK nor symbols. A range that is missing here does
# not degrade the search: it silently answers "no matches" for text that is in the store
# (audit p05 SQ02 — `𠀀` against a stored `𠀀𠀁` returned nothing). Supplementary ideographs and
# the extension blocks are therefore included, and the symbol ranges cover the whole
# Miscellaneous-Symbols..Dingbats..Supplemental-Symbols span rather than three islands of it.
_CJK_RE = re.compile(
    r"["
    r"\u2e80-\u2fdf"          # CJK radicals / Kangxi
    r"\u3000-\u303f"          # CJK symbols and punctuation
    r"\u3040-\u30ff"          # kana
    r"\u3100-\u312f"          # bopomofo
    r"\u3190-\u319f"          # kanbun
    r"\u31c0-\u31ef"          # CJK strokes
    r"\u3200-\u4dbf"          # enclosed CJK + extension A
    r"\u4e00-\u9fff"          # unified ideographs
    r"\ua960-\ua97f"          # hangul jamo extended-A
    r"\uac00-\ud7ff"          # hangul syllables + jamo extended-B
    r"\uf900-\ufaff"          # compatibility ideographs
    r"\ufe30-\ufe4f"          # CJK compatibility forms
    r"\uff00-\uffef"          # halfwidth and fullwidth forms
    r"\U00020000-\U0003ffff"  # supplementary ideographs (extensions B..I)
    r"]"
)
_EMOJI_RE = re.compile(
    r"["
    r"\u2190-\u21ff"          # arrows
    r"\u2300-\u23ff"          # miscellaneous technical
    r"\u2460-\u24ff"          # enclosed alphanumerics
    r"\u25a0-\u27bf"          # geometric shapes, misc symbols, dingbats
    r"\u2b00-\u2bff"          # miscellaneous symbols and arrows
    r"\ufe0f"                  # variation selector-16
    r"\U0001F000-\U0001FAFF"  # emoji planes
    r"\U0001FB00-\U0001FBFF"  # legacy computing symbols
    r"]"
)
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
_BOOLEAN_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_RISKY_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9][\-:/][A-Za-z0-9]")
_SPLIT_PUNCT_RE = re.compile(r"[-:/]+")
_STRIP_EDGE_PUNCT = "\"'()[]{}.,;"
# Characters that are special in FTS5 QUERY SYNTAX (as opposed to characters
# FTS5 simply cannot spell in a bareword). Only these have to go on the LIKE
# path, which has no query grammar of its own.
_FTS5_SPECIAL_CHARS = frozenset('"()*^-:{}.')


def _like_safe_char(char: str) -> str:
    """Map one unquoted character to its LIKE-safe form.

    LIKE is a substring match, so the only thing it needs removed is the
    operator punctuation a user typed FOR the index (quoted phrases, prefix
    ``*``). Everything else is signal it can match on — emoji above all, which
    the FTS term form must drop because unicode61 does not index it, and which
    LIKE is therefore the ONLY way to find.
    """
    return " " if char in _FTS5_SPECIAL_CHARS else char


def _fts5_safe_char(char: str) -> str:
    """Map one unquoted character to its FTS5 bareword-safe form.

    An FTS5 bareword accepts only alphanumerics; every other character is
    either query syntax (``"()*^-:{}.``), a string delimiter (``'``), or a
    plain syntax error (``? , & $ ! % = < ;`` ...). A raw natural-language
    question therefore fails ``MATCH`` outright and used to fall through to the
    LIKE full-scan, which blows the recall deadline at scale and returns
    nothing (F31 §3). Substituting a separator instead lets a question reach
    the index in its term form; the default unicode61 tokenizer splits the
    INDEXED text on exactly the same boundary, so no term is lost by the
    substitution.

    ``str.isalnum()`` alone is NOT that boundary: a combining mark is not
    alphanumeric, so a decomposed ``naïve`` (``nai`` + U+0308) split into
    ``nai ve`` while unicode61 indexes the word as ``naive`` — zero rows where
    the raw query matched. Marks therefore stay inside the token, and
    ``sanitize_fts5_query`` composes the query first.
    """
    if char.isalnum() or char.isspace():
        return char
    return char if unicodedata.category(char).startswith("M") else " "


def _lower_operator_tokens(text: str) -> str:
    return " ".join(
        token.lower() if token in _BOOLEAN_OPERATORS else token
        for token in text.split(" ")
    )


def _neutralize_bare_operators(sanitized: str) -> str:
    """Lowercase bare AND/OR/NOT/NEAR outside quoted phrases.

    FTS5 operators are only operators in UPPERCASE, so lowercasing turns them
    back into ordinary barewords. Without this a raw question SILENTLY ACQUIRES
    boolean semantics it never asked for: ``Portland, OR hotel`` sanitized to
    ``Portland OR hotel`` and broadened to a disjunction, while a leading
    ``NOT ready`` was a syntax error that dumped the query onto the LIKE
    full-scan. Raw and deliberate queries share one unmarked entry point, so the
    raw reading has to be the safe one. Quoted phrases are left alone: an
    explicit ``"NEAR"`` is already a literal, not an operator.
    """
    out: list[str] = []
    last = 0
    for match in _QUOTED_PHRASE_RE.finditer(sanitized):
        out.append(_lower_operator_tokens(sanitized[last:match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_lower_operator_tokens(sanitized[last:]))
    return "".join(out)


def sanitize_fts5_query(query: str, *, allow_operators: bool = False) -> str:
    """Reduce a query to FTS5-safe terms, preserving balanced phrase quotes.

    Composed (NFC) first so a decomposed accent is one alphanumeric character
    rather than a base plus a combining mark, which is what unicode61 folds and
    indexes. The LIKE path deliberately does NOT normalize: it is a literal
    substring match against stored bytes, so it must not re-spell the query.

    ``allow_operators`` is the explicit marker for a query a CALLER composed as
    FTS5 syntax (the benchmark harness joins its barewords with ``OR``). It
    keeps bare AND/OR/NOT/NEAR intact. It must never be set for text that came
    from a user or an agent: the default assumes raw prose, which is the only
    safe reading when the two cannot be told apart.
    """
    composed = unicodedata.normalize("NFC", query or "")
    sanitized = _sanitize_query(composed, _fts5_safe_char)
    return sanitized if allow_operators else _neutralize_bare_operators(sanitized)


def sanitize_like_query(query: str) -> str:
    """Strip FTS5 syntax operators, preserving every other character.

    The LIKE path's sanitization has to be WEAKER than the FTS one: a character
    the index cannot spell is still a character LIKE can match. Sharing the FTS
    term form here dropped emoji from the fallback that exists to find them
    (``launch 🚀`` searched only ``%launch%``).
    """
    return _sanitize_query(query, _like_safe_char)


def _sanitize_query(query: str, replace: Callable[[str], str]) -> str:
    """Walk ``query`` outside balanced phrase quotes, mapping chars via ``replace``."""
    if not query:
        return ""

    result: list[str] = []
    quote_buffer: list[str] = []
    in_quote = False
    for char in query:
        if char == '"':
            if in_quote:
                result.append('"')
                result.extend(quote_buffer)
                result.append('"')
                quote_buffer = []
                in_quote = False
            else:
                if result and not result[-1].isspace():
                    result.append(" ")
                in_quote = True
                quote_buffer = []
            continue
        if in_quote:
            quote_buffer.append(char)
            continue
        result.append(replace(char))
    if in_quote and quote_buffer:
        result.extend(replace(char) for char in "".join(quote_buffer))
    return " ".join("".join(result).split())


_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


def contains_risky_fts_ascii(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.count('"') % 2:
        return True
    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", raw)
    return bool(_RISKY_FTS_TOKEN_RE.search(text_without_phrases))


def requires_like_fallback(query: str, sanitized: str | None = None) -> bool:
    """Whether ``query`` must be answered by the LIKE scan instead of the index.

    The test is what SANITIZATION LOSES, not what the FTS5 query grammar cannot
    spell. A compound token (``art-related``, ``api:v2``, ``a/b``) sanitizes to
    ordinary terms the index answers perfectly well, so routing it to the
    full-table LIKE scan just re-imports the scaling ceiling this branch is
    fixing — 6 of the 50 fixed Phase 1B questions carry a hyphen. The risky-ASCII
    check therefore runs against the SANITIZED form, which is what actually
    reaches ``MATCH``.

    Genuine losses stay on LIKE: unicode61 does not segment CJK, it drops emoji
    from the index entirely, and a query that sanitizes to nothing has no terms
    left to match. Those two character classes are tested against the RAW query
    because sanitization is exactly what removes them.
    """
    raw = query or ""
    safe = sanitize_fts5_query(raw) if sanitized is None else (sanitized or "")
    if not safe.strip():
        return True
    if contains_cjk(raw) or contains_emoji(raw):
        return True
    return contains_risky_fts_ascii(safe)


def _token_variants(token: str) -> List[str]:
    cleaned = (token or "").strip().strip(_STRIP_EDGE_PUNCT)
    if not cleaned:
        return []
    if cleaned.upper() in _BOOLEAN_OPERATORS:
        return []

    variants = [cleaned]
    if _SPLIT_PUNCT_RE.search(cleaned):
        parts = [part for part in _SPLIT_PUNCT_RE.split(cleaned) if part]
        if len(parts) > 1:
            variants.extend(parts)

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def extract_search_terms(query: str) -> List[str]:
    text = (query or "").strip()
    if not text:
        return []

    terms: list[str] = []
    for phrase in _QUOTED_PHRASE_RE.findall(text):
        cleaned = phrase.strip()
        if cleaned:
            terms.append(cleaned)

    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", text)
    for token in text_without_phrases.split():
        terms.extend(_token_variants(token))

    if not terms:
        fallback_text = text.strip().strip(_STRIP_EDGE_PUNCT)
        if fallback_text:
            terms.append(fallback_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def describe_query_interpretation(query: str) -> dict:
    """fork: betterlcm — what this query was actually searched for, and what was dropped.

    Term extraction removes bare Boolean words and edge punctuation, and sanitisation strips
    characters the index treats as syntax. Upstream did all of that silently, so a search for
    ``AND 🚀`` or ``*`` quietly became a different search — or no search at all — and the empty
    result read as an exhaustive negative (audit p05 SQ01).
    """
    text = (query or "").strip()
    terms = extract_search_terms(text)
    phrases = extract_quoted_phrases(text)
    kept = {term.casefold() for term in terms} | {phrase.casefold() for phrase in phrases}
    dropped: list[str] = []
    for token in _QUOTED_PHRASE_RE.sub(" ", text).split():
        cleaned = token.strip().strip(_STRIP_EDGE_PUNCT)
        if not cleaned:
            if token not in dropped:
                dropped.append(token)
            continue
        if cleaned.casefold() in kept:
            continue
        if any(cleaned.casefold() in term.casefold() for term in terms):
            continue
        if token not in dropped:
            dropped.append(token)
    return {"terms": terms, "phrases": phrases, "dropped_tokens": dropped}


def extract_quoted_phrases(query: str) -> List[str]:
    return [phrase.strip() for phrase in _QUOTED_PHRASE_RE.findall(query or "") if phrase.strip()]


def escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def count_term_matches(text: str, term: str) -> int:
    haystack = (text or "")
    needle = (term or "")
    if not haystack or not needle:
        return 0
    return haystack.lower().count(needle.lower())


def compute_directness_score(text: str, terms: List[str], phrases: List[str] | None = None) -> float:
    content = text or ""
    if not content:
        return 0.0

    unique_hits = 0
    total_hits = 0
    non_phrase_unique_hits = 0
    non_phrase_total_hits = 0
    normalized_phrases = {(phrase or "").strip().lower() for phrase in (phrases or []) if (phrase or "").strip()}
    for term in terms:
        matches = count_term_matches(content, term)
        if matches > 0:
            unique_hits += 1
            total_hits += matches
            if term.strip().lower() not in normalized_phrases:
                non_phrase_unique_hits += 1
                non_phrase_total_hits += matches

    phrase_hits = 0
    lowered = content.lower()
    for phrase in phrases or []:
        if phrase and phrase.lower() in lowered:
            phrase_hits += 1

    repetition_penalty = max(0, total_hits - unique_hits)
    non_phrase_repetition_penalty = max(0, non_phrase_total_hits - non_phrase_unique_hits)
    score = float((unique_hits * 5) + (phrase_hits * 8))
    if not phrases:
        score -= min(repetition_penalty, 6)
    else:
        score -= min(non_phrase_repetition_penalty, 6)

    if phrases:
        for phrase in phrases:
            normalized_phrase = (phrase or "").strip().lower()
            if not normalized_phrase:
                continue
            phrase_occurrences = lowered.count(normalized_phrase)
            if phrase_occurrences <= 1:
                continue
            segments = re.split(re.escape(normalized_phrase), lowered)
            gap_unique_counts = []
            for segment in segments:
                segment_tokens = [
                    token.lower()
                    for token in _WORD_RE.findall(segment)
                    if any(char.isalpha() for char in token)
                ]
                gap_unique_counts.append(len(set(segment_tokens)))
            interior_gap_counts = gap_unique_counts[1:-1]
            tail_gap_count = gap_unique_counts[-1] if gap_unique_counts else 0
            extra_occurrences = phrase_occurrences - 1
            score -= extra_occurrences * 0.5
            score -= sum(1.5 for count in interior_gap_counts if 0 < count <= 4)
            if all(count == 0 for count in interior_gap_counts) and tail_gap_count <= 2:
                score -= min(extra_occurrences, 3) * 1.0

    return score


def _is_precise_query_shape(terms: List[str], phrases: List[str] | None = None) -> bool:
    if len(terms) == 1:
        return True
    return len(phrases or []) == 1 and len(terms) <= 2


def should_widen_candidate_fetch(terms: List[str], phrases: List[str] | None = None) -> bool:
    return _is_precise_query_shape(terms, phrases)


def should_apply_directness_rank_adjustment(terms: List[str], phrases: List[str] | None = None) -> bool:
    return _is_precise_query_shape(terms, phrases)


def compute_directness_rank_bonus_upper_bound(terms: List[str], phrases: List[str] | None = None) -> float:
    return float((len(terms) * 5) + (len(phrases or []) * 8))


def compute_search_fetch_limit(limit: int, terms: List[str], phrases: List[str] | None = None) -> int:
    base = max(limit * 5, limit, 20)
    if should_widen_candidate_fetch(terms, phrases):
        return max(base, limit * 10, 50)
    return base


def compute_like_fallback_fetch_limit(limit: int, terms: List[str], phrases: List[str] | None = None) -> int:
    """Bound LIKE fallback candidate rows before Python-side scoring/sorting."""
    return compute_search_fetch_limit(limit, terms, phrases)


def compute_search_candidate_cap(limit: int) -> int:
    """Return a hard upper bound on candidate rows inspected per search call."""
    return min(max(limit * 20, limit, 500), 5_000)


AGE_DECAY_RATE = 0.001


def normalize_search_sort(sort: str | None) -> str:
    """Normalize sort parameter to one of: recency, relevance, hybrid."""
    normalized = (sort or "recency").strip().lower()
    return normalized if normalized in {"recency", "relevance", "hybrid"} else "recency"


def build_snippet(text: str, terms: List[str], width: int = 80) -> str:
    content = (text or "")
    if not content:
        return ""
    # fork: betterlcm — match on the ORIGINAL text. Case folding is not length-preserving
    # (``"İ".lower()`` is two characters), so offsets taken in a lowered copy and applied to
    # the original drifted: 100 dotted capital I's before the match produced a snippet holding
    # neither the match nor any source text (audit p05 SQ05).
    #
    # ASCII content — the overwhelmingly common case, and the one this helper runs on for
    # every LIKE-search hit — takes a straight ``find`` on ONE lowered copy instead. For ASCII
    # text and any term, ``lowered.find(term.lower())`` and ``re.IGNORECASE`` agree exactly
    # (a non-ASCII term cannot occur in ASCII text at all), and the regex scan per absent term
    # plus a fresh ``content.lower()`` after each miss cost 104ms on a 2.2M-character source
    # against 5.6ms for the plain scan (round-2 verify-2 #12). Non-ASCII content keeps the
    # exact regex path, but folds the source only once.
    ascii_content = content.isascii()
    lowered = content.lower()
    for term in terms:
        if not term:
            continue
        if ascii_content:
            idx = lowered.find(term.lower())
        else:
            match = re.search(re.escape(term), content, re.IGNORECASE)
            idx = match.start() if match else -1
            if idx < 0:
                # Fold-only equality (ß/ss, İ/i̇) still deserves a snippet; accept the lowered
                # offset only when the original text at that offset really is the term.
                candidate = lowered.find(term.lower())
                if candidate >= 0 and content[candidate:candidate + len(term)].lower() == term.lower():
                    idx = candidate
        if idx >= 0:
            start = max(0, idx - width // 2)
            end = min(len(content), idx + len(term) + width // 2)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet
    return content[:width] + ("..." if len(content) > width else "")
