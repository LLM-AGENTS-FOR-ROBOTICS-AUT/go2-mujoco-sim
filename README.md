# Unitree Go2 EDU MuJoCo simulator

A clone-and-run package for the official Unitree Go2 model in MuJoCo. It
downloads Unitree's robot assets during setup and does not bundle a locomotion
policy or deployment controller.

The repository stays small: Python environments, compiled libraries, Unitree
SDKs, and MuJoCo are downloaded locally into ignored folders.

## Choose your setup

| Platform | Setup and launch files | What runs |
| --- | --- | --- |
| [macOS 11+ — Apple Silicon (M-series)](macos/README.md) | `macos/` | Go2 standing-pose viewer, flat or terrain scene |
| [Windows 10/11 — x64](windows/README.md) | `windows/` | The same Go2 standing-pose viewer |
| Ubuntu 22.04/24.04 — x86-64 | `scripts/` | Native Unitree simulator and SDK2/DDS bridge for a separate controller |

The Windows and macOS viewers share `go2_viewer.py` and the official robot assets.
They do not include a learned walking policy or the Ubuntu DDS bridge.

## macOS simulator quick start

Requires an Apple Silicon (M-series) Mac running macOS 11 (Big Sur) or newer,
Git, Apple's Command Line Tools, and internet access. Python and MuJoCo are
installed automatically; Homebrew and `sudo` are not needed.

```bash
git clone https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim.git
cd go2-mujoco-sim
bash macos/setup.sh
bash macos/run.sh
```

For rough terrain, run `bash macos/run.sh --terrain`. You can also double-click
**Launch Go2 Viewer.command** or **Launch Go2 Viewer - Terrain.command** in the
**macos** folder. Launchers run setup automatically if the installation is missing.

Use the mouse to rotate, pan, and zoom; close the window or press Escape to exit.
If `git` prompts to install Apple's Command Line Tools, finish that installation
and repeat the clone command. See the [macOS guide](macos/README.md) for details.

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
go2_viewer.py              Shared macOS/Windows MuJoCo model viewer
macos/setup.sh             Automatic macOS Python and MuJoCo setup
macos/run.sh               macOS launcher (uses mjpython for the window)
macos/requirements.txt     Pinned Apple Silicon dependencies
macos/*.command            One-click macOS launchers
windows/setup.ps1          Windows setup
windows/requirements.txt   Pinned Windows dependencies
windows/*.cmd             One-click Windows launchers
scripts/setup-ubuntu.sh    Full Unitree SDK2/DDS installer
scripts/run-simulator.sh   Ubuntu simulator launcher
```

## Important safety boundary

The Windows and macOS launchers control **only the simulated robot**.

No locomotion model or physical-robot deployment code is included. Do not point
experimental low-level control code at a physical Go2 without an
experienced supervisor, a physical emergency stop, a suspended initial test,
and Unitree's documented procedure for disabling conflicting services.

## Troubleshooting

- **macOS:** run `bash macos/run.sh --validate` for a headless check.
- **Windows:** run `.\.venv\Scripts\python.exe .\go2_viewer.py --validate`.
- Add `--terrain` to either command to check the terrain scene.
- A walking controller is intentionally not included.
- For installation or graphics issues, use the [macOS](macos/README.md#troubleshooting)
  or [Windows](windows/README.md#troubleshooting) guide.

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream projects and
licenses.
