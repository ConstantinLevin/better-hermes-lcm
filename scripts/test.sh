#!/usr/bin/env bash
# Run the plugin test suite. pytest is not installed in the Hermes venv; uv overlays it.
# The Hermes venv provides the real `agent.context_engine` ABC that engine.py imports.
# Usage: scripts/test.sh [pytest args...]   (default: tests/ -q)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
UV="${UV:-/home/agent/.hermes/bin/uv}"
PY="${HERMES_PYTHON:-/home/agent/.hermes/hermes-agent/venv/bin/python}"
cd "$HERE"
# The plugin refuses SQLite files whose parent dir is group/other-writable; pytest tmp dirs
# inherit the shell umask (0002 on this box), so force a private umask for the run.
umask 077
exec "$UV" run --no-project --python "$PY" --with pytest --with numpy \
  python -m pytest "${@:-tests/}" -q -p no:cacheprovider
