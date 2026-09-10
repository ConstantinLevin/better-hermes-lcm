# Configuration and activation

Hermes-LCM is a general Hermes plugin and a context engine. Both identities must be active:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine configuration. Verify with `hermes plugins`, then use `lcm_status` after a normal message has bound the session.

## Installation

An existing checkout can install profile-aware plugin and skill links:

```bash
./scripts/install.sh
HERMES_PROFILE=myprofile ./scripts/install.sh
```

The installer exposes both:

- `plugins/hermes-lcm` for plugin loading;
- `skills/hermes-lcm` for normal skill discovery.

It refuses conflicting paths rather than overwriting an existing install.

## High-impact controls

`README.md` is the complete current source for settings, `FORK.md` for the window-weighted
defaults this build resolves, and `docs/operator-guide.md` for what silently suppresses the
curve.

**Most sizes are derived, not configured.** Threshold, fresh tail, leaf chunk, pass cap, drain
stop, condensation budget, depth, timeouts, concurrency and several caps slide with the model's
context window (upstream's values at 256k, the large-window design at 1M). Read
`lcm_status` → `window_scaling` before changing any of them: it prints every resolved value and
whether it came from the curve, an env var or config. An explicit value always wins, which also
means an unnecessary one silently disables the scaling for that setting.

Start with:

- `LCM_CONTEXT_THRESHOLD`: when normal context pressure triggers compaction (curve: 0.35 → 0.80);
- `LCM_FRESH_TAIL_COUNT`: newest messages kept raw (curve: 32 → 400);
- `LCM_LEAF_CHUNK_TOKENS`: floor of raw material before a leaf compaction runs;
- `LCM_DATABASE_PATH`: profile-local SQLite path when the default is unsuitable;
- `LCM_IGNORE_SESSION_PATTERNS` and `LCM_STATELESS_SESSION_PATTERNS`: storage ownership boundaries;
- summary/embedding provider settings only after confirming credentials, cost, and data handling.

Optional slash commands are disabled by default with `LCM_ENABLE_SLASH_COMMAND=false`. Destructive cleanup apply is separately guarded. Do not enable mutation surfaces merely to diagnose a problem.

Change one tuning variable at a time, then re-check `lcm_status`, context pressure, summary health, latency, and actual answer quality.
