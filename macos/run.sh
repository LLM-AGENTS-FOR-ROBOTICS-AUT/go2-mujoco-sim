#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is for macOS. See windows/ or scripts/run-simulator.sh." >&2
  exit 1
fi

if [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" == "1" ]]; then
  echo "Open Terminal without 'Open using Rosetta', then retry." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This launcher requires an Apple Silicon (M-series) Mac." >&2
  exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/mjpython" ]] || \
   [[ ! -f "$ROOT/_deps/unitree_mujoco/unitree_robots/go2/scene.xml" ]] || \
   [[ ! -f "$ROOT/_deps/unitree_mujoco/unitree_robots/go2/scene_terrain.xml" ]]; then
  echo "First-time setup is required. This can take a few minutes."
  bash "$ROOT/macos/setup.sh"
fi

# Headless checks need no window; the interactive macOS viewer requires mjpython.
for arg in "$@"; do
  case "$arg" in
    --validate|--help|-h) exec "$ROOT/.venv/bin/python" "$ROOT/go2_viewer.py" "$@" ;;
  esac
done
source "$ROOT/macos/developer-tools.sh"
exec "$ROOT/.venv/bin/mjpython" "$ROOT/go2_viewer.py" "$@"
