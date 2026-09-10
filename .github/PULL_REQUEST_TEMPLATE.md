## What changed
-

## Why
-

## Evidence
- [ ] `python3 scripts/e2e_no_loss.py 262144 400` — 0 unreachable rows / 0 missing facts / 0 never offered
- [ ] `python3 scripts/e2e_no_loss.py 1000000 3000` — same three zeros
- [ ] Anything else you ran, and what it showed

## If this touches an upstream file
- [ ] The hook carries a `# fork: better-hermes-lcm` marker and the reason is in the comment
- [ ] Said here what should happen to it when upstream edits the same function
