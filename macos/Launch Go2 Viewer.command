#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if bash "$ROOT/macos/run.sh" "$@"; then
  exit 0
else
  status=$?
  echo "The simulator could not start. See macos/README.md for troubleshooting."
  read -r -p "Press Return to close..." || true
  exit "$status"
fi
