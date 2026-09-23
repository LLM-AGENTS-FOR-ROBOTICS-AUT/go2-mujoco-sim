@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup is required. This can take a few minutes.
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1"
  if errorlevel 1 goto :failed
)

"%~dp0.venv\Scripts\python.exe" "%~dp0go2_viewer.py" %*
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo The simulator could not start. See README.md for troubleshooting.
pause
exit /b 1
