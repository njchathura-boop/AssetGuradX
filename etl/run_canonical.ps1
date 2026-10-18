param([string]$DriveRoot = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($DriveRoot) { $env:ASSETGUARDX_DRIVE_ROOT = $DriveRoot }
Write-Host "=== AssetGuardX canonical hourly rebuild (NO BIGQUERY) ==="
python src/build_canonical_hourly.py
if ($LASTEXITCODE -ne 0) { throw "Canonical hourly build failed." }
