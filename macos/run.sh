#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT/macos/platform.sh"

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
