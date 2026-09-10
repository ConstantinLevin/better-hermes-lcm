"""fork: better-hermes-lcm — ``lcm_doctor coverage``: the executable definition of "no loss".

A summary is acceptable only while a reader can tell from it WHAT the sources contain. This
check extracts index-bearing entities from a node's sources (paths, identifiers, quoted
strings, numbers, URLs, decision/rejection keywords) and reports the fraction that are still
present in the node's summary + index block. It is a heuristic floor, not a proof: a node
scoring low is a node whose sources should be read before trusting the summary.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_PATH_RE = re.compile(r"(?<![\w/])(?:~|\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)+|(?<![\w/])[\w.\-]+\.(?:py|md|json|yaml|yml|toml|txt|sh|ts|js|sql|db|log)\b")
_IDENT_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"   # snake_case
    r"|[a-z]+(?:[A-Z][a-z0-9]+){2,}"                 # camelCase
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"            # PascalCase
    r"|[A-Z][A-Z0-9_]{3,})\b"                        # CONSTANTS
)
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`\n]{3,60})[\"'`]")
_NUMBER_RE = re.compile(r"(?<![\w.])\d{3,}(?:[.,]\d+)?(?![\w.])")
_DECISION_RE = re.compile(
    r"\b(decid\w*|agreed|chose|chosen|rejected|instead of|rather than|must not|never|always|"
    r"constraint\w*|blocked|blocker|error|failed|fixed|resolved|workaround|TODO|open question)\b",
    re.IGNORECASE,
)
_STOP_IDENTS = {"tool_calls", "tool_call_id", "session_id", "store_id", "node_id"}


def extract_index_entities(text: str, *, limit: int = 400) -> List[str]:
    """Distinct entities a future reader would search for, in first-seen order."""
    text = str(text or "")
    seen: Dict[str, None] = {}

    def add(value: str) -> None:
        value = value.strip().strip(".,;:")
        if len(value) < 3 or value.lower() in _STOP_IDENTS:
            return
        seen.setdefault(value, None)

    for match in _URL_RE.finditer(text):
        add(match.group(0))
    for match in _PATH_RE.finditer(text):
        add(match.group(0))
    for match in _IDENT_RE.finditer(text):
        add(match.group(0))
    for match in _QUOTED_RE.finditer(text):
        add(match.group(1))
    for match in _NUMBER_RE.finditer(text):
        add(match.group(0))
    for match in _DECISION_RE.finditer(text):
        add(match.group(0).lower())
    out = list(seen)
    return out[:limit]


def index_entities_with_bound(text: str, *, limit: int = 400) -> tuple[List[str], bool]:
    """fork: better-hermes-lcm — the entity list AND whether the cap cut it short.

    A source holding 1,000 filenames whose summary kept the first 400 scored 100% coverage:
    the cap decided which entities existed, and the score was computed over exactly the ones
    that survived (round-2 verify-4 #34).
    """
    entities = extract_index_entities(text, limit=limit + 1)
    return entities[:limit], len(entities) > limit


def _present(entity: str, haystack: str) -> bool:
    needle = entity.lower()
    if needle in haystack:
        return True
    # decision keywords count by stem (decided/decision)
    stem = needle[:5]
    return len(needle) >= 6 and stem.isalpha() and stem in haystack


def coverage_of(summary_text: str, index_block: str, sources_text: str,
                *, source_truncated: bool = False) -> Dict[str, Any]:
    entities, entities_truncated = index_entities_with_bound(sources_text)
    haystack = (str(summary_text or "") + "\n" + str(index_block or "")).lower()
    present = [e for e in entities if _present(e, haystack)]
    missing = [e for e in entities if not _present(e, haystack)]
    # fork: better-hermes-lcm — no evidence is NOT full coverage. A node whose sources are missing,
    # unreadable or empty yields zero entities; scoring that 1.0 let a broken-provenance node
    # certify as perfect. Report it as unscored and let the caller decide.
    fraction = (len(present) / len(entities)) if entities else None
    return {
        "entities": len(entities),
        "present": len(present),
        "fraction": round(fraction, 3) if fraction is not None else None,
        "scored": fraction is not None,
        "missing_sample": missing[:12],
        # fork: better-hermes-lcm — the score describes only what was EXAMINED (round-2 verify-4 #34)
        "entities_truncated": bool(entities_truncated),
        "source_truncated": bool(source_truncated),
        "coverage_bounded": bool(entities_truncated or source_truncated),
    }


def _source_text_for_node(engine: Any, node: Any, *, max_chars: int = 400_000):
    """Return ``(text, unreadable_source_ids, truncated)``.

    A source the node records but that cannot be read is reported, never silently treated as
    empty; and a source list longer than ``max_chars`` says so, because a score computed over
    the first 400,000 characters is not a score over the sources (round-2 verify-4 #34).
    """
    parts: List[str] = []
    unreadable: List[int] = []
    if node.source_type == "messages":
        rows = engine._store.get_batch(list(node.source_ids))
        for store_id in node.source_ids:
            row = rows.get(store_id)
            if not row:
                unreadable.append(int(store_id))
                continue
            content = row.get("content")
            if isinstance(content, str):
                parts.append(content)
            tool_calls = row.get("tool_calls")
            if tool_calls:
                parts.append(str(tool_calls))
    else:
        for child_id in node.source_ids:
            child = engine._dag.get_node(int(child_id))
            if child is None:
                unreadable.append(int(child_id))
                continue
            parts.append(child.summary)
    text = "\n".join(parts)
    return text[:max_chars], unreadable, len(text) > max_chars


def node_coverage(engine: Any, node: Any) -> Dict[str, Any]:
    meta = None
    store = getattr(engine._dag, "node_meta", None)
    if store is not None:
        try:
            meta = store.read(node.node_id)
        except Exception:
            meta = None
    index_block = str((meta or {}).get("index_block") or "")
    sources_text, unreadable, source_truncated = _source_text_for_node(engine, node)
    result = coverage_of(node.summary, index_block, sources_text,
                         source_truncated=source_truncated)
    result.update({
        "node_id": int(node.node_id),
        "depth": int(node.depth),
        "level": int((meta or {}).get("level", 1) or 1),
        "source_count": len(node.source_ids),
        # fork: a source the node records but the store/DAG cannot produce is a provenance
        # failure, reported separately from semantic coverage.
        "unreadable_source_ids": unreadable,
    })
    return result


def session_coverage(engine: Any, session_id: Optional[str] = None, *, limit: int = 200,
                     floor: float = 0.6) -> Dict[str, Any]:
    session_id = session_id or engine.current_session_id or engine._session_id
    nodes = engine._dag.get_session_nodes(session_id, limit=max(1, int(limit)))
    per_node = [node_coverage(engine, node) for node in nodes]
    scored = [n for n in per_node if n["entities"] > 0]
    below = [n for n in scored if n["fraction"] < floor]
    total_entities = sum(n["entities"] for n in scored)
    total_present = sum(n["present"] for n in scored)
    # fork: better-hermes-lcm — three outcomes, never conflated: scored, unscored (no evidence), and
    # structurally broken (a recorded source that cannot be read).
    unscored = [n for n in per_node if n["entities"] == 0]
    broken = [n for n in per_node if n["unreadable_source_ids"]]
    bounded = [n for n in per_node if n.get("coverage_bounded")]
    truncated_scan = len(nodes) >= max(1, int(limit))
    return {
        "session_id": session_id,
        "floor": floor,
        "nodes": len(per_node),
        "scored_nodes": len(scored),
        "unscored_nodes": [n["node_id"] for n in unscored],
        "nodes_with_unreadable_sources": [
            {"node_id": n["node_id"], "unreadable_source_ids": n["unreadable_source_ids"][:12]}
            for n in broken
        ],
        # a paging limit bounds the WORK, never the claim: say when the scan was partial
        "scan_complete": not truncated_scan,
        # fork: better-hermes-lcm — nodes whose score covers only part of their sources or entities
        "nodes_with_bounded_coverage": [n["node_id"] for n in bounded],
        "aggregate_fraction": round(total_present / total_entities, 3) if total_entities else None,
        "nodes_below_floor": [
            {k: n[k] for k in ("node_id", "depth", "level", "fraction", "entities", "missing_sample")}
            for n in sorted(below, key=lambda n: n["fraction"])[:20]
        ],
        "per_node": [
            {k: n[k] for k in ("node_id", "depth", "level", "fraction", "entities", "present")}
            for n in per_node
        ],
    }


def coverage_check(report: Dict[str, Any]) -> Dict[str, Any]:
    """A ``checks`` entry for lcm_doctor."""
    below = report.get("nodes_below_floor") or []
    broken = report.get("nodes_with_unreadable_sources") or []
    unscored = report.get("unscored_nodes") or []
    bounded = report.get("nodes_with_bounded_coverage") or []
    aggregate = report.get("aggregate_fraction")
    floor = float(report.get("floor", 0.6))
    if broken or bounded or not report.get("scan_complete", True):
        status = "fail" if broken else "warn"
    elif aggregate is None:
        status = "warn"          # nothing could be scored: not evidence of coverage
    else:
        status = "pass" if not below and float(aggregate) >= floor else "warn"
    aggregate_text = "not scored" if aggregate is None else f"{float(aggregate):.0%}"
    detail = (
        f"{report.get('scored_nodes', 0)} of {report.get('nodes', 0)} node(s) scored, aggregate "
        f"{aggregate_text} of index-bearing entities still discoverable; "
        f"{len(below)} under the {floor:.0%} floor"
    )
    if unscored:
        detail += f"; {len(unscored)} node(s) yielded no evidence to score"
    if broken:
        detail += f"; {len(broken)} node(s) reference sources that cannot be read"
    if not report.get("scan_complete", True):
        detail += "; scan was truncated by the node limit"
    return {"check": "index_coverage", "status": status, "detail": detail}
