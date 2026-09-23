$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$deps = Join-Path $root "_deps"
$unitreeMujoco = Join-Path $deps "unitree_mujoco"
$unitreeMujocoRevision = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows, then rerun this script."
}

if (-not (Test-Path -LiteralPath $python)) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Python 3.12 is missing and winget is unavailable. Install Python 3.12 from python.org."
    }
    Write-Host "Installing Python 3.12 for the current user..."
    winget install --id Python.Python.3.12 --exact --scope user --silent `
        --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "Python installation failed with exit code $LASTEXITCODE."
    }
}

New-Item -ItemType Directory -Force -Path $deps | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $unitreeMujoco ".git"))) {
    Write-Host "Downloading Unitree's official MuJoCo robot assets..."
    git clone https://github.com/unitreerobotics/unitree_mujoco.git $unitreeMujoco
}

Write-Host "Selecting the tested Unitree MuJoCo revision..."
git -C $unitreeMujoco fetch --depth 1 origin $unitreeMujocoRevision
if ($LASTEXITCODE -ne 0) {
    throw "Could not download the tested Unitree MuJoCo revision."
}
git -C $unitreeMujoco checkout --detach $unitreeMujocoRevision
if ($LASTEXITCODE -ne 0) {
    throw "Could not select the tested Unitree MuJoCo revision."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating an isolated Python environment..."
    & $python -m venv (Join-Path $root ".venv")
}

Write-Host "Installing MuJoCo..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --requirement (Join-Path $root "requirements-windows.txt")

Write-Host "Validating the robot model and simulator..."
& $venvPython (Join-Path $root "go2_viewer.py") --validate
if ($LASTEXITCODE -ne 0) {
    throw "The Go2 simulator validation failed."
}

Write-Host ""
Write-Host "Setup complete. Double-click 'Launch Go2 Viewer.cmd'."
