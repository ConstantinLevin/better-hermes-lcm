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
- `leaf_pipeline.py`      — three-phase leaf loop: preamble / concurrent summarise / persist
- `coverage_doctor.py`    — `lcm_doctor coverage`: index-adequacy check
- `tests/fork/`           — fork tests, organised by the step that introduced them
