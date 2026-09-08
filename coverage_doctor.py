"""fork: betterlcm — ``lcm_doctor coverage``: the executable definition of "no loss".

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


def _present(entity: str, haystack: str) -> bool:
    needle = entity.lower()
    if needle in haystack:
        return True
    # decision keywords count by stem (decided/decision)
    stem = needle[:5]
    return len(needle) >= 6 and stem.isalpha() and stem in haystack


def coverage_of(summary_text: str, index_block: str, sources_text: str) -> Dict[str, Any]:
    entities = extract_index_entities(sources_text)
    haystack = (str(summary_text or "") + "\n" + str(index_block or "")).lower()
    present = [e for e in entities if _present(e, haystack)]
    missing = [e for e in entities if not _present(e, haystack)]
    fraction = (len(present) / len(entities)) if entities else 1.0
    return {
        "entities": len(entities),
        "present": len(present),
        "fraction": round(fraction, 3),
        "missing_sample": missing[:12],
    }


def _source_text_for_node(engine: Any, node: Any, *, max_chars: int = 400_000) -> str:
    parts: List[str] = []
    if node.source_type == "messages":
        rows = engine._store.get_batch(list(node.source_ids))
        for store_id in node.source_ids:
            row = rows.get(store_id)
            if not row:
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
            if child is not None:
                parts.append(child.summary)
    text = "\n".join(parts)
    return text[:max_chars]


def node_coverage(engine: Any, node: Any) -> Dict[str, Any]:
    meta = None
    store = getattr(engine._dag, "node_meta", None)
    if store is not None:
        try:
            meta = store.read(node.node_id)
        except Exception:
            meta = None
    index_block = str((meta or {}).get("index_block") or "")
    result = coverage_of(node.summary, index_block, _source_text_for_node(engine, node))
    result.update({
        "node_id": int(node.node_id),
        "depth": int(node.depth),
        "level": int((meta or {}).get("level", 1) or 1),
        "source_count": len(node.source_ids),
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
    return {
        "session_id": session_id,
        "floor": floor,
        "nodes": len(per_node),
        "scored_nodes": len(scored),
        "aggregate_fraction": round(total_present / total_entities, 3) if total_entities else 1.0,
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
    aggregate = float(report.get("aggregate_fraction", 1.0))
    floor = float(report.get("floor", 0.6))
    status = "pass" if not below and aggregate >= floor else "warn"
    return {
        "check": "index_coverage",
        "status": status,
        "detail": (
            f"{report.get('scored_nodes', 0)} node(s) scored, aggregate {aggregate:.0%} of index-bearing "
            f"entities still discoverable; {len(below)} node(s) under the {floor:.0%} floor"
        ),
    }
