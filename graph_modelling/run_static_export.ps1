param([string]$DriveRoot = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($DriveRoot) { $env:ASSETGUARDX_DRIVE_ROOT = $DriveRoot }
Write-Host "=== AssetGuardX TigerGraph static topology export (NO BIGQUERY) ==="
python scripts/export_static_topology.py
if ($LASTEXITCODE -ne 0) { throw "TigerGraph export failed." }
