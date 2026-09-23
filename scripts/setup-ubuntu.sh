#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/_deps"
UNITREE_SDK2_REV="63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36"
UNITREE_MUJOCO_REV="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
MUJOCO_VERSION="3.3.6"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This installer currently supports x86_64 Ubuntu Linux only." >&2
  exit 1
fi

sudo apt update
sudo apt install -y \
  build-essential cmake curl git tar \
  libboost-all-dev libyaml-cpp-dev libspdlog-dev libfmt-dev libeigen3-dev \
  libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev

mkdir -p "$DEPS"

clone_at_revision() {
  local url="$1"
  local destination="$2"
  local revision="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach "$revision"
}

clone_at_revision \
  https://github.com/unitreerobotics/unitree_sdk2.git \
  "$DEPS/unitree_sdk2" "$UNITREE_SDK2_REV"

cmake -S "$DEPS/unitree_sdk2" -B "$DEPS/unitree_sdk2/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
cmake --build "$DEPS/unitree_sdk2/build" -j"$(nproc)"
sudo cmake --install "$DEPS/unitree_sdk2/build"

clone_at_revision \
  https://github.com/unitreerobotics/unitree_mujoco.git \
  "$DEPS/unitree_mujoco" "$UNITREE_MUJOCO_REV"

if [[ ! -d "$DEPS/mujoco-$MUJOCO_VERSION" ]]; then
  curl -fL \
    "https://github.com/google-deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz" \
    -o "$DEPS/mujoco.tar.gz"
  tar -xzf "$DEPS/mujoco.tar.gz" -C "$DEPS"
fi
ln -sfn "$DEPS/mujoco-$MUJOCO_VERSION" "$DEPS/unitree_mujoco/simulate/mujoco"

cmake -S "$DEPS/unitree_mujoco/simulate" -B "$DEPS/unitree_mujoco/simulate/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/unitree_robotics
cmake --build "$DEPS/unitree_mujoco/simulate/build" -j"$(nproc)"

# Keep the simulator on localhost DDS domain 0 so a separate controller can be
# connected later without exposing traffic on a physical robot network.
sed -i 's/^domain_id:.*/domain_id: 0  # Local simulator domain/' \
  "$DEPS/unitree_mujoco/simulate/config.yaml"
sed -i 's/^use_joystick:.*/use_joystick: 0 # No controller is bundled/' \
  "$DEPS/unitree_mujoco/simulate/config.yaml"

echo
echo "Setup complete."
echo "Start the simulator with: ./scripts/run-simulator.sh"
