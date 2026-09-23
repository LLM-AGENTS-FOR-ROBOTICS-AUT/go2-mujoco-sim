#!/usr/bin/env bash
# Sourced by setup and the graphical launcher. Scope any selection to this process.
if ! /usr/bin/otool -h /usr/bin/true >/dev/null 2>&1; then
  if DEVELOPER_DIR=/Library/Developer/CommandLineTools \
    /usr/bin/otool -h /usr/bin/true >/dev/null 2>&1; then
    export DEVELOPER_DIR=/Library/Developer/CommandLineTools
  else
    echo "MuJoCo's macOS viewer needs Apple's Command Line Tools (otool)." >&2
    echo "Run xcode-select --install, finish the installer, then retry." >&2
    echo "If you use full Xcode, open it and complete its first-launch setup." >&2
    return 1
  fi
fi
