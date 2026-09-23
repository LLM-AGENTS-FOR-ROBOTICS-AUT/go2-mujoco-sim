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
The robot holds a standing pose; no walking policy or DDS bridge is included in
this viewer.

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
