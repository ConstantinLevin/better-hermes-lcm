# Release validation

Use `scripts/validate_release.sh` as the local release-confidence lane before tagging or publishing. The script is offline by default: it does not call model providers, does not mutate live Hermes config, routes Python bytecode/cache artifacts under the validation output directory, and writes validation artifacts under a fresh output directory.

**This script is not the fork's evidence.** A green suite is a regression guard. The executable
form of this fork's no-loss doctrine is the two end-to-end anchors in
[`CLAUDE.md`](../CLAUDE.md#verification--what-counts-as-evidence), and both must report 0
unreachable rows, 0 missing facts and 0 facts never offered to the summariser before a release is
a release:

```bash
python3 scripts/e2e_no_loss.py 262144 400
python3 scripts/e2e_no_loss.py 1000000 3000
```

## Command

Prerequisites:

- Run from the repository checkout.
- Use a Python environment with `pytest` installed. If `python` on `PATH` is not the intended interpreter, set `PYTHON=/path/to/python`.
- The benchmark and stress gates are standalone-checkout safe: they provide the minimal Hermes Agent `ContextEngine` base class needed for deterministic local validation when Hermes Agent is not importable.
- The whitespace/conflict-marker gate resolves its range in this order: `LCM_RELEASE_DIFF_BASE` if set, else `origin/main...HEAD` when `origin/main` exists and differs from `HEAD`, else `HEAD^...HEAD`. It then also checks the local working tree and staged diff. **This fork's branch is `better-hermes-lcm`, not `main`**, so on a clone that has no `origin/main` the gate silently falls back to a single-commit range — set `LCM_RELEASE_DIFF_BASE=<rev-or-range>` (for example `origin/better-hermes-lcm...HEAD`) to validate the whole release diff.
- Python validation runs with `PYTHONPYCACHEPREFIX` under the output directory and pytest cache disabled, then records git status before and after validation so release runs do not silently dirty the checkout.
- The low-file-descriptor full gate lowers the limit to 1024 only when the current shell allows it; locked-down hosts keep their existing lower limit instead of failing before pytest starts.

```bash
scripts/validate_release.sh
```

Default smoke mode runs the local gates that should be cheap enough for routine operator use:

- adaptive `git diff --check` over the resolved range above, plus local working-tree and staged diff checks
- `scripts/validate_dependency_contract.py --report-environment` (see [Dependency assurance](dependency-assurance.md))
- Python compile checks for the plugin and release scripts
- shell syntax checks for `scripts/install.sh`, `scripts/update.sh` and the validator itself
- focused pytest over `test_lcm_core`, `test_lcm_command`, `test_packaging_install`, `test_benchmarking_cli`, `test_stress_release_check` and `test_historical_externalization_backfill`
- deterministic benchmark smoke with a synthetic fixture
- deterministic stress smoke

For pre-tag confidence, run:

```bash
scripts/validate_release.sh --full
```

Full mode adds the whole test suite, the low-file-descriptor pytest pass, and the release stress tier. It is intentionally heavier and still avoids provider/network side effects.

## Artifact shape

Each run creates a fresh directory, by default:

```text
/tmp/hermes-lcm-release-validation-YYYYMMDD-HHMMSS/
```

Important files:

- `validation-checklist.md` — scrubbed operator checklist, command summary, and before/after git status
- `logs/*.log` — stdout/stderr for each validation command
- `pycache/` — validation-time Python bytecode cache redirected away from the source tree
- `benchmark-smoke/summary.json` and `benchmark-smoke/metrics.jsonl` — deterministic benchmark artifacts
- `stress-smoke/stress-summary.md` and `stress-smoke/results/stress-results.json` — deterministic stress artifacts

The checklist is safe to paste into a release note or PR validation section after reviewing any local path values. Do not paste raw logs unless they have been scrubbed for local paths, secrets, and unrelated environment details.

## Checklist template

```md
## Release validation

- Command: `scripts/validate_release.sh [--full]`
- Mode: `smoke` or `full`
- Repo: `<branch>@<commit>`
- Output dir: `<validation artifact dir>`

### Gates
- [ ] git diff/whitespace check passed
- [ ] dependency contract validated
- [ ] Python compile checks passed
- [ ] shell syntax checks passed
- [ ] focused or full pytest passed
- [ ] deterministic benchmark smoke passed
- [ ] deterministic stress smoke/release passed
- [ ] git status before/after validation reviewed
- [ ] `scripts/e2e_no_loss.py` clean at both anchors (262144 and 1000000)

### Doctor triage
- [ ] `lcm_doctor` warnings were classified as `safe/ignore`, `inspect`, or `backup-first cleanup`
- [ ] no warning-only class was auto-cleaned without operator review

### Notes
- Skipped gates:
- Warnings reviewed:
- Recommended next release-confidence step:
```

## Warning-only boundaries

Doctor warnings should remain warning-only when the runtime cannot prove that mutation is safe:

- summary quality warnings: inspect retrieval/summary behavior; do not rewrite DAG state automatically
- lifecycle fragmentation: inspect first; only use explicit backup-first lifecycle cleanup for empty lifecycle rows
- payload-storage suspicion: inspect/restore missing payload files before deleting or rewriting anything
- context pressure: usually safe to ignore unless compaction is stuck or repeatedly firing
- legacy blank-source rows: normalized as `unknown` for compatibility; only run source normalization after a backup-first review
