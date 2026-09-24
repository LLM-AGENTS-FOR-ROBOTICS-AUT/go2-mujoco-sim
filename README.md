# Unitree Go2 EDU MuJoCo simulator

A clone-and-run package for the official Unitree Go2 model in MuJoCo. It
downloads Unitree's robot assets during setup. An optional CPU walking demo
uses the Go2 Walk These Ways policy from go2_deploy, with simulated pose
feedback to track velocity commands and keep straight moves straight.

The repository stays small: Python environments, compiled libraries, Unitree
SDKs, and MuJoCo are downloaded locally into ignored folders.

## Choose your setup

| Platform | Setup and launch files | What runs |
| --- | --- | --- |
| [macOS 11+ — Apple Silicon or Intel](macos/README.md) | `macos/` | Standing viewer; optional walking on M-series macOS 14+ |
| [Windows 10/11 — x64](windows/README.md) | `windows/` | Standing viewer; optional walking setup prepared, execution pending |
| Ubuntu 22.04/24.04 — x86-64 | `scripts/` | Native Unitree simulator and SDK2/DDS bridge for a separate controller |

The Windows and macOS viewers share `go2_viewer.py` and the official robot assets.
The optional walking demo shares `go2_locomotion.py`. These viewers control only
the simulation; the Ubuntu DDS bridge is a separate path.

## Try the walking policy

After cloning, use either setup below. It installs the base simulator as well
as the walking dependencies, downloads a pinned and checksum-verified policy,
and runs the tracking checks. PyTorch runs on the CPU; no training, CUDA, ROS,
or Docker is needed.

**Apple Silicon (M-series), macOS 14+:**

```bash
bash macos/setup.sh --locomotion
bash macos/run.sh --demo
```

**Windows 10/11 x64, in PowerShell:**

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1 -Locomotion
& '.\windows\Launch Go2 Viewer.cmd' --demo
```

Use `--walk` instead of `--demo` for keyboard control: **W/S** forward/backward,
**A/D** sideways, **Q/E** turn, **Space** stop and hold, **R** reset. Tap a key
to set a command; it continues until replaced or stopped. The demo walks,
stops, sidesteps, turns, reverses, then holds its stance after 37 simulated seconds.

This is a **SportClient-style Move/StopMove subset**. The tested configuration
uses a neutral trot, position and heading feedback from MuJoCo, and a smooth
standing hold when stopped. On the local Apple Silicon Mac it passed a
60-second straight walk: 23.91 m travelled, at most 4.3 cm sideways drift and
0.73 degrees heading error. These are results for the complete controller,
not the raw policy alone.

See the [policy comparison, results and limits](docs/locomotion.md). The Windows
setup and tests are prepared but have not been executed for this change yet.
The policy weights from `go2_deploy` were tested through this local adapter;
the original C++/DDS deployment program and physical Go2 EDU were not tested.

## macOS simulator quick start

Supports Apple Silicon (M-series) and Intel Macs, including quad-core Intel Core
i5 models with AVX support, running macOS 11 (Big Sur) or newer. macOS 15
(Sequoia) is supported. Requires Git, Apple's Command Line Tools, and internet
access. Setup detects your processor and installs the matching Python and
MuJoCo packages automatically; Homebrew and `sudo` are not needed.

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
go2_locomotion.py          Optional simulation-only Move/StopMove controller
validate_locomotion.py     Real-policy movement, stopping and idle checks
policy_assets.py           Pinned and verified walking-policy download
requirements-locomotion.txt Optional CPU-only policy runtime
macos/setup.sh             Automatic macOS Python and MuJoCo setup
macos/run.sh               macOS launcher (uses mjpython for the window)
macos/requirements.txt     Pinned Apple Silicon and Intel dependencies
macos/*.command            One-click macOS launchers
windows/setup.ps1          Windows setup
windows/requirements.txt   Pinned Windows dependencies
windows/*.cmd             One-click Windows launchers
scripts/setup-ubuntu.sh    Full Unitree SDK2/DDS installer
scripts/run-simulator.sh   Ubuntu simulator launcher
```

## Important safety boundary

The Windows and macOS launchers control **only the simulated robot**.

The optional model is downloaded during setup; no physical-robot deployment
code is included in the Windows/macOS viewer. Do not point
experimental low-level control code at a physical Go2 without an
experienced supervisor, a physical emergency stop, a suspended initial test,
and Unitree's documented procedure for disabling conflicting services.

## Troubleshooting

- **macOS:** run `bash macos/run.sh --validate` for a headless check.
- **Windows:** run `.\.venv\Scripts\python.exe .\go2_viewer.py --validate`.
- Add `--terrain` to either command to check the terrain scene.
- For walking checks, add `--walk --validate` after installing the optional policy.
- For installation or graphics issues, use the [macOS](macos/README.md#troubleshooting)
  or [Windows](windows/README.md#troubleshooting) guide.

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream projects and
licenses.
