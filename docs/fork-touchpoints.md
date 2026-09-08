# Upstream files touched by the fork

Format: `file` — what the hook does — why — how to re-apply on a conflict.
(Maintained by hand; every fork commit that touches an upstream file updates this list.)

| file | hook | why | on conflict |
|---|---|---|---|
| `config.py` | 14 new dataclass fields (block before `from_env`) + 14 `_EnvFieldSpec` entries (tail of `ENV_FIELD_SPECS`) | window-weighted tuning overrides; defaults are "unset" sentinels so upstream behaviour is unchanged | pure additions at the tails — keep ours; if upstream added fields in the same spot, keep both |
