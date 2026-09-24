# Windows setup

Requires Windows 10/11 (x64), Git, current graphics drivers, and internet access.
Setup installs Python 3.12 for the current user with `winget` if needed, creates
the repository's `.venv`, downloads the pinned Unitree assets into `_deps`, and
validates both scenes.

## Install and run

From PowerShell:

```powershell
git clone https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim.git
cd go2-mujoco-sim
powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1
& '.\windows\Launch Go2 Viewer.cmd'
```

Or double-click **Launch Go2 Viewer.cmd** in this folder. It runs setup on the
first launch if the environment is missing. Use **Launch Go2 Viewer - Terrain.cmd**
for rough terrain, or pass `--terrain` to the regular launcher.

Use the mouse to rotate, pan, and zoom. Close the window or press Escape to exit.
The default viewer holds a standing pose. The optional walking demo below
adds locomotion; the Ubuntu DDS bridge is a separate setup.

## Optional walking demo

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1 -Locomotion
& '.\windows\Launch Go2 Viewer.cmd' --demo
```

Setup also works from a fresh clone. It installs CPU PyTorch and
downloads the pinned Go2 Walk These Ways policy. It uses simulated position
and heading feedback for tracking, and a standing hold for stopping. Use `--walk` for manual keyboard
control: W/S forward/back, A/D sideways, Q/E turn, Space stop, R reset. Commands
persist after releasing a key; press Space to stop. Launching works offline once
setup finishes.

Run `& '.\windows\Launch Go2 Viewer.cmd' --walk --validate` to repeat the
motion checks without a window. The JSON report is saved to
`_deps/locomotion-results.json`. See
[the policy comparison and test results](../docs/locomotion.md).

## Troubleshooting

- **Python installed somewhere else:** run
  `powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1 -Python 'C:\path\to\python.exe'`
  with the path to a Python 3.12 executable.
- **No winget:** install Python 3.12 from [python.org](https://www.python.org/downloads/windows/),
  then rerun setup (use `-Python` if it is outside the default per-user location).
- **Check without opening a window:**
  `.\.venv\Scripts\python.exe .\go2_viewer.py --validate` and
  `.\.venv\Scripts\python.exe .\go2_viewer.py --terrain --validate`.
- **Black viewer:** update the graphics driver.
- **Incomplete installation:** rerun setup. To recreate the Python environment,
  close the viewer, rename `.venv` to `.venv-old`, and rerun setup.

All commands above assume the repository root as the current directory.
Windows setup and launch files live in this folder; `go2_viewer.py` and downloaded
assets are shared at the repository root.
