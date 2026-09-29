param(
    [string]$Python = "python",
    [switch]$SkipTorch,
    [switch]$SkipDepthProDownload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DepthCheckpoint = Join-Path $ProjectRoot "checkpoints\depth_pro.pt"
$LocalDepthPro = Join-Path $ProjectRoot "depth_anything_demo\third_party\depth_pro_src\ml-depth-pro-main"
$LocalDepthCheckpoint = Join-Path $LocalDepthPro "checkpoints\depth_pro.pt"

Write-Host "[1/5] Creating Python environment..."
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv (Join-Path $ProjectRoot ".venv")
}

Write-Host "[2/5] Updating installer..."
& $VenvPython -m pip install --upgrade pip setuptools wheel

if (-not $SkipTorch) {
    Write-Host "[3/5] Installing PyTorch (CUDA 12.8 wheels)..."
    & $VenvPython -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "[3/5] Verifying the PyTorch already installed in .venv..."
    & $VenvPython -c "import torch, torchvision"
    if ($LASTEXITCODE -ne 0) {
        throw "-SkipTorch requires torch and torchvision to already be installed in .venv."
    }
}

Write-Host "[4/5] Installing Wujie Aperture and preparing DepthPro..."
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-app.txt")
if (Test-Path -LiteralPath (Join-Path $LocalDepthPro "src\depth_pro\__init__.py")) {
    Write-Host "Using bundled DepthPro source: $LocalDepthPro"
} else {
    & $VenvPython -m pip install "depth_pro @ git+https://github.com/apple/ml-depth-pro.git@9efe5c1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Git clone failed; retrying with the official repository archive..."
        & $VenvPython -m pip install "https://github.com/apple/ml-depth-pro/archive/9efe5c1.zip"
    }
}

if (-not $SkipDepthProDownload -and -not (Test-Path -LiteralPath $DepthCheckpoint) -and -not (Test-Path -LiteralPath $LocalDepthCheckpoint)) {
    Write-Host "[5/5] Downloading official DepthPro checkpoint..."
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DepthCheckpoint) | Out-Null
    Invoke-WebRequest -Uri "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" -OutFile $DepthCheckpoint
} else {
    Write-Host "[5/5] DepthPro checkpoint already exists or download was skipped."
}

& $VenvPython (Join-Path $ProjectRoot "scripts\verify_install.py")
Write-Host "Setup complete. Start with: .\scripts\start.ps1"
