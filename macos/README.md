# macOS setup

Requires **macOS 11 (Big Sur) or newer**, an **Apple Silicon (M-series) Mac**, Git,
Apple's **Command Line Tools**, and internet access for setup. Use a normal
desktop Terminal to open the viewer. Intel Macs are not supported.

## Install and run

Paste these commands into Terminal:

```bash
git clone https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim.git
cd go2-mujoco-sim
bash macos/setup.sh
bash macos/run.sh
```

If `git` prompts to install Apple's Command Line Tools, complete the installation
and retry. You can also start that installation with `xcode-select --install`.

Setup downloads Python 3.12, installs MuJoCo, fetches the pinned official Unitree
Go2 assets, and checks both scenes. No Homebrew, `sudo`, manual Python install,
or environment activation is needed. Rerunning setup reuses the installation.

After setup, launch from the repository root:

```bash
bash macos/run.sh                 # Flat ground
bash macos/run.sh --terrain       # Rough terrain
```

Or double-click **Launch Go2 Viewer.command** or
**Launch Go2 Viewer - Terrain.command** in this folder. The launchers also run
setup automatically on first use, so you can skip the explicit setup command.
After installation, launching works offline.

Use the mouse to rotate, pan, and zoom. Close the window or press Escape to exit.
The robot holds a standing pose; this viewer includes no walking policy or DDS
bridge. The Ubuntu native simulator is a separate setup.

## What gets installed

| Location | Contents |
| --- | --- |
| `.venv/` | Isolated Python environment and MuJoCo dependencies |
| `_deps/python/` | Python 3.12 downloaded by uv |
| `_deps/tools/` | Pinned uv installer tool |
| `_deps/uv-cache/` | Download cache for repeat installations |
| `_deps/unitree_mujoco/` | Official Unitree robot assets at a fixed revision |

These folders are ignored by Git. Setup leaves your system Python and shell
configuration alone. Windows files live in `windows/`; only the viewer code and
robot assets are shared.

The installer uses MuJoCo 3.13.0's prebuilt Apple Silicon package, so it does not
compile MuJoCo. See [MuJoCo's package release](https://pypi.org/project/mujoco/3.13.0/)
and [uv's installer options](https://docs.astral.sh/uv/reference/installer/).

## Troubleshooting

- **Check without opening a window:** run `bash macos/run.sh --validate` and
  `bash macos/run.sh --terrain --validate`. These check model loading and physics,
  so they also work in a session without a display.
- **An error says to use `mjpython`:** launch with `bash macos/run.sh`, not plain
  `python go2_viewer.py`. The launcher handles [MuJoCo's macOS window requirement](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).
- **Command Line Tools / `otool` error:** run `xcode-select --install` and complete
  the installer. If full Xcode is selected but not ready, the launcher uses an
  already-working Command Line Tools installation for this process only; it
  does not change your system's Xcode selection. If neither works, open Xcode
  and complete its first-launch setup, then retry.
- **Apple Silicon Terminal running under Rosetta:** quit Terminal, turn off
  **Open using Rosetta** in Terminal's Finder **Get Info** window, reopen it,
  and run setup again.
- **A download fails or setup is incomplete:** check your internet connection
  and rerun `bash macos/setup.sh`. It reuses completed downloads.
- **The repository moved or an incompatible `.venv` exists:** close the viewer,
  rename `.venv` to `.venv-old`, and rerun setup. Python environments should be
  recreated after moving the repository or switching between operating systems.
- **A double-click launcher is blocked or not executable:** use the Terminal
  commands above. A Git clone preserves executable permissions; ZIP downloads
  may not. To restore them, run `chmod +x macos/*.command macos/*.sh`.
- **Window does not open over SSH:** run from Terminal in your Mac's graphical
  desktop session. Use `--validate` for a headless check.

All commands above assume the repository root as the current directory.
