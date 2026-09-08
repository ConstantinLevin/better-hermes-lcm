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

## Layout of fork-only code
- `window_scaling.py`     — the anchor table and the curve; `resolve_window_scaled(config, W)`
- `host_cooldown.py`      — the host's compression-failure cooldown protocol for a plugin engine
- `errors.py`             — `SummaryUnavailableError` (replaces upstream's silent L3 truncation)
- `marked_loss.py`        — every marker text the fork leaves where upstream cut or dropped silently
- `node_meta.py`          — sidecar table `lcm_node_meta` (escalation level + index block)
- `leaf_pipeline.py`      — concurrent leaf summarisation as a lookahead over the serial loop; compaction lock
- `coverage_doctor.py`    — `lcm_doctor coverage` / `/lcm doctor coverage`: index-adequacy check
- `tests/fork/`           — fork tests, organised by the step that introduced them
