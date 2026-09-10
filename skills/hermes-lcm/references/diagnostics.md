# Diagnostics

Use read-only tools before changing configuration or running an apply path.

## Fast path

1. `hermes plugins` — confirm `hermes-lcm` is enabled and the selected context engine is `lcm`.
2. Send one normal message if the session has not been bound since restart.
3. `lcm_status` — runtime identity, database path, context pressure, summary/store counts, filter
   state, rotate snapshot state, and the `window_scaling` block showing every curve-resolved value
   and where it came from.
4. `lcm_inspect` — current-session lineage, message frontier and fresh tail, compaction frontier,
   the latest skip/no-op reason, and externalized-ref readability. Metadata only, no content.
5. `lcm_doctor` — database integrity, core schema and FTS index sync, ingest health, ignore-pattern
   drops, payload storage, orphaned DAG nodes, summary quality, `index_coverage` (whether summaries
   actually name the identifiers in their sources), config validation, source-lineage hygiene,
   lifecycle fragmentation, and context pressure.

`/lcm status` and `/lcm doctor` expose the same views when slash commands are enabled.

## Safe mutation order

For cleanup, repair, source normalization, or rotate:

1. run the read-only preview;
2. inspect the exact candidates and paths;
3. create or confirm a backup;
4. get the user's authorization for that specific apply operation;
5. run one bounded apply, then verify integrity.

Cleanup apply is feature-gated separately (`doctor_clean_apply_enabled`). Never infer permission to
enable it from a request to diagnose something.

## Common states

- **Unbound status after restart** — send a normal message, then check again.
- **Database exists but stays empty** — verify plugin enablement, `context.engine`, the profile, the
  database path, and the ignore/stateless patterns.
- **Compaction reports a no-op** — `lcm_inspect` carries the reason. "raw backlog outside fresh tail
  is below one leaf chunk" is normal, not a fault.
- **Weak exact recall** — confirm the source rows exist, the query and scope are right, summary
  health is sound, and embedding coverage matches the requested mode. A summary that does not name
  a detail is expected; expand to the original instead of retrying the search.
- **Summary and raw evidence conflict** — prefer the newer exact raw evidence and inspect lineage.
- **Path B / context-engine schema log** — expected on hosts where plugin-registry handlers do not
  receive active messages. Context-engine schemas and dispatch remain the healthy route.
