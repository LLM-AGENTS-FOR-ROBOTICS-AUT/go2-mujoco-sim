param(
    [string]$Python = "",
    [switch]$Locomotion
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
}
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
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download Unitree's robot assets."
    }
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
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python environment."
    }
}

Write-Host "Installing MuJoCo..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not update pip."
}
& $venvPython -m pip install --requirement (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install MuJoCo."
}

Write-Host "Validating the robot model and simulator..."
& $venvPython (Join-Path $root "go2_viewer.py") --validate
if ($LASTEXITCODE -ne 0) {
    throw "The Go2 simulator validation failed."
}
& $venvPython (Join-Path $root "go2_viewer.py") --terrain --validate
if ($LASTEXITCODE -ne 0) {
    throw "The Go2 terrain validation failed."
}

if ($Locomotion) {
    Write-Host "Installing the optional CPU walking policy..."
    & $venvPython -m pip install --only-binary :all: --requirement (Join-Path $root "requirements-locomotion.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install walking dependencies." }
    & $venvPython (Join-Path $root "policy_assets.py")
    if ($LASTEXITCODE -ne 0) { throw "Could not download and verify the Go2 policy." }
    & $venvPython (Join-Path $root "go2_viewer.py") --walk --validate
    if ($LASTEXITCODE -ne 0) { throw "The Go2 locomotion validation failed." }
    Write-Host "Walking demo ready: windows\Launch Go2 Viewer.cmd --demo"
}

Write-Host ""
Write-Host "Setup complete. Double-click 'windows\Launch Go2 Viewer.cmd'."
