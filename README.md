# Unitree Go2 EDU MuJoCo simulator

A clone-and-run package for the official Unitree Go2 model in MuJoCo. It
downloads Unitree's robot assets during setup and does not bundle a locomotion
policy or deployment controller.

The repository stays small: Python environments, compiled libraries, Unitree
SDKs, and MuJoCo are downloaded locally into ignored folders.

## Choose your setup

- **Windows:** easiest way to open the official Go2 model and verify MuJoCo.
  The included viewer holds the robot in a standing pose; it does not run the
  learned walking policy.
- **Ubuntu 22.04/24.04 x86-64:** builds Unitree's native MuJoCo simulator and
  SDK2/DDS bridge, ready for a separate compatible controller.

## Windows simulator quick start

Requirements: Windows 10/11, Git, current graphics drivers, and internet access.

```powershell
git clone https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim.git
cd go2-mujoco-sim
powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1
```

Then open the **windows** folder and double-click **Launch Go2 Viewer.cmd**.
For Unitree's terrain scene, use **Launch Go2 Viewer - Terrain.cmd** in the same
folder. Use the mouse to rotate, pan, and zoom;
close the MuJoCo window or press Escape to exit.

See the [Windows guide](windows/README.md) for launch commands and troubleshooting.

## Native Unitree simulator on Ubuntu

This path builds Unitree's native simulator and SDK2/DDS bridge. It supports
x86-64 Ubuntu and requires `sudo`.

```bash
git clone https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim.git
cd go2-mujoco-sim
chmod +x scripts/*.sh
./scripts/setup-ubuntu.sh
```

```bash
./scripts/run-simulator.sh
```

The native simulator publishes Unitree SDK2 topics on DDS domain `0` over the
loopback (`lo`) interface. A controller such as the lecturer-provided
`go2_deploy` should be kept in its own repository and connected separately.

## Repository layout

```text
go2_viewer.py              Native Windows/Python MuJoCo model viewer
windows/setup.ps1          Windows setup
windows/requirements.txt   Pinned Windows dependencies
windows/*.cmd             One-click Windows launchers
scripts/setup-ubuntu.sh    Full Unitree SDK2/DDS installer
scripts/run-simulator.sh   Ubuntu simulator launcher
```

## Important safety boundary

The Windows launcher controls **only the simulated robot**.

No locomotion model or physical-robot deployment code is included. Do not point
experimental low-level control code at a physical Go2 without an
experienced supervisor, a physical emergency stop, a suspended initial test,
and Unitree's documented procedure for disabling conflicting services.

## Troubleshooting

- Run `.\.venv\Scripts\python.exe .\go2_viewer.py --validate` for a headless check.
- If the viewer is black, update the graphics driver.
- A walking controller is intentionally not included.
- Delete `.venv` and `_deps`, then rerun setup, to rebuild a clean installation.

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream projects and
licenses.
