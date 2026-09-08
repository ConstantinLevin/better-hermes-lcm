# betterlcm — fork of stephenschoettler/hermes-lcm

Base: upstream commit recorded in `.upstream-base` (`git log -1 $(cat .upstream-base)`).
Branch `betterlcm` carries every fork change; remote `upstream` tracks the original.

## Why this fork exists
Make a 1,000,000-token context window work well under a strict no-loss rule, without
degrading ~256k-window behaviour. Full design: the plan this fork implements is summarised
in `docs/fork-design.md`. Two rules govern every change:

1. **Weighted tuning** — every size that is a *preference* is a linear function of
   `context_length` between two anchors: upstream's value at 256k and the 1M design at 1M.
   Anchors live in one table: `window_scaling.py::WINDOW_SCALED_DEFAULTS`.
2. **Pure optimizations** — loss removal, throughput, index quality — apply at every window
   and must leave the 256k DAG identical to upstream's.

### The four configured exceptions to the no-loss rule

Every other path either keeps the content or marks what it removed. These four are operator
policies that deliberately do not, so they are named here rather than discovered later
(audit verify-4 #19). None is on by default except the auxiliary one.

| exception | what is not recoverable | how to turn it off |
|---|---|---|
| `ignore_message_patterns` | a matching message is never stored; the active context keeps a hash placeholder | leave the setting empty (default) |
| sensitive redaction (`sensitive_patterns_enabled`) | the redacted span is replaced before storage, and two different secrets can produce the same placeholder | leave it disabled (default) |
| ignored / stateless / auxiliary sessions | the conversation is bounded but never stored at all | `ignore_session_patterns` / `stateless_session_patterns` empty; auxiliary (subagent) sessions are excluded by design |
| trajectory protection | the protected payload is redacted in place before hashing | do not enable the trajectory subsystem |

If an operator needs those sessions or spans archived, the fork's answer today is: do not use
the exception. Preserving originals behind protection is a design decision, not a bug fix.

## How to upgrade from upstream (for the next maintainer)
```
git fetch upstream
git merge upstream/main          # or rebase betterlcm onto upstream/main
scripts/test.sh                  # must be green
```
Then redeploy: `~/.hermes/plugins/hermes-lcm` is a clone of this repo (remote `fork`), pinned in
`~/.hermes/plugins/.install-metadata.json` so `hermes plugins update` refuses to touch it.
```
git -C ~/.hermes/plugins/hermes-lcm pull fork betterlcm
python3 - <<'PY'   # re-pin the deployed revision
import json, subprocess, pathlib
p = pathlib.Path.home()/".hermes/plugins/.install-metadata.json"; d = json.loads(p.read_text())
d["hermes-lcm"]["revision"] = subprocess.check_output(["git","-C",str(pathlib.Path.home()/".hermes/plugins/hermes-lcm"),"rev-parse","HEAD"],text=True).strip()
p.write_text(json.dumps(d, indent=2)+"\n")
PY
```
`plugin.yaml` keeps upstream's version string (four upstream tests pin it; the fork is
identified by this file and `git log`). The live `lcm.db` gains the
`lcm_node_meta` table on first use; an upstream build classifies such a DB as newer (drop the
table and the `betterlcm_node_meta_v1` row in `lcm_migration_state` to go back).
Fork code is kept in NEW modules wherever possible so upstream files receive only small,
localized hook calls. The complete list of upstream files touched, with the reason for each
touch, is in `docs/fork-touchpoints.md` — read it before resolving a conflict: it tells you
whether a conflicting hunk is a hook (keep ours, re-apply on top of theirs) or a behavioural
change upstream also made (reconcile). Every fork-only test lives under `tests/fork/`.

## Tracking lossless-claw (the second maintenance duty)
The fork was inspired by [lossless-claw](https://github.com/Martian-Engineering/lossless-claw)
(OpenClaw, TypeScript). On **every lossless-claw release**:
1. Clone the release tag to a temp dir (`git clone --depth 1 --branch vX.Y.Z … /tmp/lossless-claw-vX.Y.Z`).
2. Compare it deeply against this fork, one agent per aspect, code-level with citations on both
   sides: (1) compaction/DAG algorithm, (2) loss avoidance/provenance/recovery, (3) summariser
   prompts/index quality/evaluation, (4) retrieval tools/host integration/operability. The
   prompts used for v1.0.0 are in `docs/claw-comparison/prompts/`; reuse them.
3. Keep only findings where claw is genuinely better *for this fork's purpose* (no loss; 1M
   without degrading 256k) or fixes something the hermes-lcm base does badly. Port them into
   the fork modules, with tests under `tests/fork/`, and record each ported item (and each
   rejected one, with the reason) in `docs/claw-comparison/vX.Y.Z.md`.
4. The same two hard rules apply as for upstream merges: 256k DAG structure stays upstream's,
   and nothing drops content without a marker.

## Fork configuration reference
Every fork setting has a dataclass field, an env var and (for the weighted ones) an anchor in
`window_scaling.py`. `0` / `0.0` means "no override — use the curve"; `lcm_status` →
`window_scaling` shows the resolved value and its source for the current window.

| env var | default | field |
|---|---|---|
| `LCM_SCALE_LOW_WINDOW` | `262144` | `scale_low_window` — window where every weighted value equals upstream's |
| `LCM_SCALE_HIGH_WINDOW` | `1000000` | `scale_high_window` — window where it equals the large-window design |
| `LCM_DRAIN_STOP_FRACTION` | `0.0` | `drain_stop_fraction` (curve: threshold → 0.30 of W) |
| `LCM_LEAF_CHUNK_FRACTION` | `0.0` | `leaf_chunk_fraction` (curve: 1.0 → 0.04 of W) |
| `LCM_LEAF_PASS_CAP` | `0` | `leaf_pass_cap` (curve: 1 → 64) |
| `LCM_LEAF_LOOP_MAX_SECONDS` | `0.0` | `leaf_loop_max_seconds` (curve: 120 → 200) |
| `LCM_SUMMARY_BUDGET_FRACTION` | `0.0` | `summary_budget_fraction` (curve: 0 → 0.20 of W; condensation trigger) |
| `LCM_SUMMARY_CONCURRENCY` | `0` | `summary_concurrency` (curve: 1 → 6) |
| `LCM_SERIALIZE_MESSAGE_MAX_CHARS` | `0` | `serialize_message_max_chars` (curve: 3000 → 4·W) |
| `LCM_EXPAND_PAGE_TOKENS` | `0` | `expand_page_tokens` (curve: 4000 → 32000) |
| `LCM_TOOL_RESPONSE_CHAR_SCALE` | `0.0` | `tool_response_char_scale` (curve: 1 → 4) |
| `LCM_SQLITE_CACHE_KIB` | `0` | `sqlite_cache_kib` (curve: 2048 → 65536) |
| `LCM_TOKEN_CACHE_SIZE` | `0` | `token_cache_size` (curve: 2048 → 8192) |
| `LCM_SUMMARY_FAILURE_COOLDOWN_SECONDS` | `600.0` | `summary_failure_cooldown_seconds` |
| `LCM_ASSEMBLY_MAX_NODES_PER_DEPTH` | `100000` | `assembly_max_nodes_per_depth` (cap hit is marked in the prefix) |
| `LCM_SWEEP_MAX_PASSES` | `12` | `sweep_max_passes` (upstream's constant) |
| `LCM_LEAF_SUMMARY_RATIO` / `_MIN_TOKENS` / `_MAX_TOKENS` | `0.2` / `2000` / `12000` | leaf summary size rule (upstream's literals) |
| `LCM_CONDENSATION_RATIO` / `_MIN_TOKENS` | `0.4` / `1000` | condensation size rule (upstream's literals) |

Upstream settings the curve also drives when not set explicitly: `context_threshold`
(0.35 → 0.80), `summary_timeout_ms` (60 s → 200 s), `expansion_timeout_ms` (120 s → 200 s),
`fresh_tail_count` (32 → 400), `fresh_tail_max_tokens` (off → 0.15·W), `incremental_max_depth`
(3 → 5), `summary_spend_max_calls` (24 → 120), `summary_circuit_breaker_failure_threshold`
(2 → 4), `l2_budget_ratio` (0.5 → 0.8), `large_output_active_replay_stub_threshold_tokens`
(25k → 100k), `expansion_context_tokens` (32k → 125k), `summary_prefix_target_tokens`
(leaf_chunk_tokens → 0.20·W, sweep flag only).

## Layout of fork-only code
- `window_scaling.py`     — the anchor table and the curve; `resolve_window_scaled(config, W)`
- `host_cooldown.py`      — the host's compression-failure cooldown protocol for a plugin engine
- `errors.py`             — `SummaryUnavailableError` (replaces upstream's silent L3 truncation)
- `marked_loss.py`        — every marker text the fork leaves where upstream cut or dropped silently
- `node_meta.py`          — sidecar table `lcm_node_meta` (escalation level + index block)
- `leaf_pipeline.py`      — concurrent leaf summarisation as a lookahead over the serial loop; compaction lock
- `coverage_doctor.py`    — `lcm_doctor coverage` / `/lcm doctor coverage`: index-adequacy check
- `tests/fork/`           — fork tests, organised by the step that introduced them
