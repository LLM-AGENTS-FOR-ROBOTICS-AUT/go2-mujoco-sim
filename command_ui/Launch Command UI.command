#!/usr/bin/env bash
# Double-click to start the LLM command console (the MuJoCo window and the web page open).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ROOT/.venv/bin/mjpython" ]] || \
   ! "$ROOT/.venv/bin/python" -c "import flask, faster_whisper, torch" >/dev/null 2>&1; then
  echo "First-time setup of the LLM command console. This can take several minutes."
  bash "$ROOT/macos/setup.sh" --agent || { read -r -p "Setup failed. Press Return to close..."; exit 1; }
fi
if "$ROOT/.venv/bin/python" "$ROOT/command_ui/server.py" "$@"; then
  exit 0
else
  status=$?
  echo "The command console could not start. See command_ui/README.md for troubleshooting."
  read -r -p "Press Return to close..." || true
  exit "$status"
fi
