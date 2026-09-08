# Upstream files touched by the fork

Format: `file` — what the hook does — why — how to re-apply on a conflict.
(Maintained by hand; every fork commit that touches an upstream file updates this list.)

| file | hook | why | on conflict |
|---|---|---|---|
| `config.py` | 14 new dataclass fields (block before `from_env`) + 14 `_EnvFieldSpec` entries (tail of `ENV_FIELD_SPECS`) | window-weighted tuning overrides; defaults are "unset" sentinels so upstream behaviour is unchanged | pure additions at the tails — keep ours; if upstream added fields in the same spot, keep both |
| `engine.py` | `WindowScaledSettingsMixin` import + base class; `_init_window_scaled_settings()` in `__init__`; `_resolve_window_scaled_settings()` before both `return True` in `_set_context_length` | curve resolution point | keep our 4 one-liners on top of theirs; if upstream restructured `_set_context_length`, re-add the call on every return path |
| `tools.py` | `"window_scaling": engine.window_scaling_status()` in the status dict | expose resolved values + sources | keep ours |
| `tests/test_lcm_engine.py` | assertions `int(1_000_000 * engine._config.context_threshold)` -> `engine.context_threshold` (4 sites) | they pinned a window-independent default; the design curves it | re-apply the same substitution if upstream adds more |
