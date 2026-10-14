@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" goto :setup
".venv\Scripts\python.exe" -c "import flask, faster_whisper, torch" >nul 2>&1 || goto :setup
goto :run

:setup
echo First-time setup of the LLM command console. This can take several minutes.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "windows\setup.ps1" -Agent
if errorlevel 1 goto :failed

:run
".venv\Scripts\python.exe" "command_ui\server.py" %*
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo The command console could not start. See command_ui\README.md for troubleshooting.
pause
exit /b 1
