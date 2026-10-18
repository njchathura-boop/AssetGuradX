param([string]$DriveRoot = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($DriveRoot) { $env:ASSETGUARDX_DRIVE_ROOT = $DriveRoot }
Write-Host "=== AssetGuardX labels (NO BIGQUERY) ==="
python src/build_labels.py
if ($LASTEXITCODE -ne 0) { throw "Labels were not built. This is expected until asset_hourly and approved thresholds exist." }
