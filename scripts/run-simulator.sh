#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/_deps/unitree_mujoco/simulate/build"
exec ./unitree_mujoco -r go2 -s scene.xml

