param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Python environment not found. Run .\scripts\setup.ps1 first."
}

$env:BOKEH_HOST = $HostAddress
$env:BOKEH_PORT = [string]$Port
Push-Location $ProjectRoot
try {
    & $VenvPython -m competition_app
} finally {
    Pop-Location
}
