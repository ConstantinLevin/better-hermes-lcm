# Configuration and activation

Hermes-LCM is both a Hermes plugin and a context engine. Both identities must be active:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine configuration. Verify with `hermes plugins`,
then run `lcm_status` after a normal message has bound the session.

`./scripts/install.sh` links an existing checkout into `plugins/hermes-lcm` and
`skills/hermes-lcm`, honouring `HERMES_PROFILE` for a profile-local install. It refuses to replace
a conflicting path rather than overwriting one.

## Most sizes are derived, not configured

Threshold, fresh tail, leaf chunk, pass cap, drain stop, condensation budget, depth, timeouts,
concurrency, and several page sizes are resolved from a weighting curve between two anchors — a
256k context window and a 1M one. **Outside that range nothing changes**: below 256k and above 1M
every weighted value is frozen at its nearest anchor.

The low anchor is *not* simply "what upstream does". It carries upstream's tuning values only where
the setting is a genuine cost/latency preference; where a setting decides how much is lost or how
coarse the index is, both anchors are chosen on merit and are often identical.

`lcm_status` → `window_scaling` prints every resolved value with its source (`curve@t=…`, `env`,
`config_yaml:…`, `explicit`) and the `t` position between anchors. Read it before changing anything.
**An explicit value always wins**, which also means an unnecessary one silently switches the curve
off for that setting.

Worth knowing before tuning:

- `LCM_CONTEXT_THRESHOLD` — when context pressure triggers compaction. Curve: `0.35` at 256k →
  `0.80` at 1M.
- Fresh tail — how much recent conversation is never compacted. The real bound is
  `LCM_FRESH_TAIL_MAX_TOKENS`, `0.15 × window` at both anchors; `LCM_FRESH_TAIL_COUNT` is a flat
  upper bound of 400 messages that can only trim the token-selected tail, never extend it.
- Leaf chunk — how much raw history becomes one summary node, i.e. the granularity of the index.
  `LCM_LEAF_CHUNK_FRACTION` is `0.04 × window` at both anchors. `LCM_LEAF_CHUNK_TOKENS` is the
  separate "do not bother compacting less than this" floor, and the effective floor is the smaller
  of it and one chunk.
- `LCM_DATABASE_PATH` — profile-local SQLite path when the default is unsuitable.
- `LCM_IGNORE_SESSION_PATTERNS` / `LCM_STATELESS_SESSION_PATTERNS` — storage ownership boundaries.
  A session matching either is never stored; leave them empty unless that is what you want.
- Summary and embedding provider settings only after confirming credentials, cost, and data
  handling.

## Off by default, and mostly should stay that way

- `LCM_ENABLE_SLASH_COMMAND=false` — optional `/lcm` commands. Destructive cleanup apply is gated
  separately again (`doctor_clean_apply_enabled`). Never enable a mutation surface just to
  diagnose something.
- `LCM_TEMPORAL_ROLLUPS_ENABLED=false` — with rollups off, `lcm_recent` answers `status: disabled`.
- Assertions, query views, and adaptive retrieval are off, which is why `lcm_query_state` and
  `lcm_retrieve` answer `status: disabled`.
- `LCM_NEW_SESSION_RETAIN_DEPTH=0` — `/new` carries no earlier summaries into the new session. Set
  `2` (or `-1`) to get carry-over back. Nothing is deleted either way.
- `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` and
  `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` cannot be turned on. A configured value is
  refused with a warning, because either one can only be switched into a loss of content the agent
  never saw.

For the complete settings list see `README.md`; for every curve anchor with both endpoints see
`FORK.md`; for what silently switches the curve off see `docs/operator-guide.md`.

Change one variable at a time, then re-check `lcm_status`, context pressure, summary health,
latency, and actual answer quality.
