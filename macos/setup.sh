#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/_deps"
UV_VERSION="0.12.18"
UV="$DEPS/tools/uv"
UNITREE_MUJOCO_REV="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
UNITREE_MUJOCO="$DEPS/unitree_mujoco"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS. See windows/ or scripts/setup-ubuntu.sh." >&2
  exit 1
fi
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 11 ]]; then
  echo "macOS 11 (Big Sur) or newer is required." >&2
  exit 1
fi
if [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" == "1" ]]; then
  echo "Open Terminal without 'Open using Rosetta', then rerun setup for Apple Silicon." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "macOS setup requires an Apple Silicon (M-series) Mac. Intel Macs are unsupported." >&2
  exit 1
fi
if ! git --version >/dev/null 2>&1; then
  echo "Git is required. Run xcode-select --install, finish the installer, then retry." >&2
  exit 1
fi

source "$ROOT/macos/developer-tools.sh"

trap 'echo "macOS setup failed. Fix the error above and rerun: bash macos/setup.sh" >&2' ERR
mkdir -p "$DEPS"

# Bootstrap a pinned, repository-local uv without Homebrew, sudo or shell edits.
if [[ ! -x "$UV" ]] || [[ "$("$UV" --version)" != "uv $UV_VERSION"* ]]; then
  echo "Downloading the Python installer (uv $UV_VERSION)..."
  curl --fail --location --silent --show-error --retry 3 \
    "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$DEPS/uv-install.sh"
  UV_UNMANAGED_INSTALL="$DEPS/tools" sh "$DEPS/uv-install.sh"
fi
export UV_PYTHON_INSTALL_DIR="$DEPS/python"
export UV_CACHE_DIR="$DEPS/uv-cache"

if [[ -e "$ROOT/.venv" ]]; then
  if ! "$ROOT/.venv/bin/python" -c \
    'import platform, sys; assert sys.version_info[:2] == (3, 12); assert platform.machine() == sys.argv[1]' \
    "$(uname -m)"; then
    echo "The existing .venv is incompatible. Rename it to .venv-old, then rerun setup." >&2
    exit 1
  fi
else
  echo "Creating an isolated Python 3.12 environment..."
  "$UV" venv --managed-python --python 3.12 "$ROOT/.venv"
fi

echo "Installing MuJoCo..."
"$UV" pip install --python "$ROOT/.venv/bin/python" --only-binary :all: \
  --requirement "$ROOT/macos/requirements.txt"

if [[ ! -d "$UNITREE_MUJOCO/.git" ]]; then
  echo "Downloading Unitree's official Go2 assets..."
  git clone --no-checkout --depth 1 \
    https://github.com/unitreerobotics/unitree_mujoco.git "$UNITREE_MUJOCO"
fi
if [[ "$(git -C "$UNITREE_MUJOCO" rev-parse HEAD)" != "$UNITREE_MUJOCO_REV" ]] || \
   [[ ! -f "$UNITREE_MUJOCO/unitree_robots/go2/scene.xml" ]]; then
  git -C "$UNITREE_MUJOCO" fetch --depth 1 origin "$UNITREE_MUJOCO_REV"
  git -C "$UNITREE_MUJOCO" checkout --detach "$UNITREE_MUJOCO_REV"
fi

echo "Validating the flat and terrain scenes..."
"$ROOT/.venv/bin/python" "$ROOT/go2_viewer.py" --validate
"$ROOT/.venv/bin/python" "$ROOT/go2_viewer.py" --terrain --validate

echo
echo "Setup complete. Start the viewer: bash macos/run.sh"
echo "For rough terrain: bash macos/run.sh --terrain"
echo "You can also double-click the launchers in the macos folder."
