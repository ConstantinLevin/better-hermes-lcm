# Contributing to better-hermes-lcm

This is a maintenance fork of [`stephenschoettler/hermes-lcm`](https://github.com/stephenschoettler/hermes-lcm),
run for one operator's Hermes install. It is public so the work can be read and reused, not
because it is looking for contributors. Issues and pull requests are welcome and will be read;
they may sit for a while.

The branch is `better-hermes-lcm`. There is no `main`.

## Before proposing a change

Read [`CLAUDE.md`](CLAUDE.md). It is short and it is the contract — in particular:

- **Upstream is the floor, never the target.** "That is what upstream does" describes a choice;
  it never justifies one.
- **Nothing becomes unreachable, and anything removed from a view leaves a marker saying what
  went and how to get it back.** A bounded, capped or failed operation is never reported as
  complete.
- **Answer from the code.** Every claim about behaviour is checked at the call site, not against
  a document — this file included.

Fork logic goes in new modules so upstream merges stay reviewable; upstream files get small
hooks marked `# fork: better-hermes-lcm`, with the reason in the comment beside them. That marker
is the merge risk surface: `grep` finds every one of them.

## What counts as validation

A green test run is a regression guard, not evidence. The gate is end-to-end behaviour:

```bash
python3 scripts/e2e_no_loss.py 262144 400
python3 scripts/e2e_no_loss.py 1000000 3000
```

Both must report **0 unreachable rows, 0 missing facts, 0 facts never offered to the
summariser**. A change that cannot produce those numbers is not finished. Note what these runs
do *not* prove — they stub the summariser and never reach condensation; `CLAUDE.md` says which
gaps that leaves.

`bash scripts/test.sh` runs the suite (umask 077 matters — SQLite refuses group-writable
directories).

## In a pull request

Say what changed, why, and what you ran. If the change touches an upstream file, say what the
hook does and what should happen to it when upstream edits the same function.
