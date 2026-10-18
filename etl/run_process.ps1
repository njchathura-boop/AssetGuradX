param([string]$DriveRoot = "", [switch]$Rebuild)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($DriveRoot) { $env:ASSETGUARDX_DRIVE_ROOT = $DriveRoot }

Write-Host "=== AssetGuardX processing + canonical hourly tables (NO BIGQUERY) ==="

$driveBase = if ($DriveRoot) { $DriveRoot } elseif ($env:ASSETGUARDX_DRIVE_ROOT) { $env:ASSETGUARDX_DRIVE_ROOT } else { "G:\My Drive" }
$archiveRoot = Join-Path $driveBase "AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1"
$rawLogSource = Join-Path $archiveRoot "raw\raw_protocol_logs"
if (Test-Path $rawLogSource) {
    $separateLogs = Get-ChildItem -LiteralPath $rawLogSource -Recurse -File -Filter "*.parquet" -ErrorAction SilentlyContinue
    if ($separateLogs.Count -gt 0) {
        throw "Raw-log extraction shards remain. Complete the raw pipeline with .\run_raw.ps1 -DriveRoot '$driveBase' -Execute before processing."
    }
}

python src/validate_raw_archive.py --deep
if ($LASTEXITCODE -ne 0) { throw "Raw archive incomplete." }

$forceArgs = @()
if ($Rebuild) { $forceArgs += "--force" }

Write-Host "=== Build intermediate tables from unified protocol_events (Drive archive only) ==="
python src/process_raw.py @forceArgs
if ($LASTEXITCODE -ne 0) { throw "Intermediate processing failed." }

python src/build_canonical_hourly.py @forceArgs
if ($LASTEXITCODE -ne 0) { throw "Canonical hourly build failed." }

Write-Host ""
Write-Host "DONE. Intermediate tables and processed/asset_hourly were derived only from the stored Drive archive."
Write-Host "BigQuery was not used. Labels remain a separate later step."
if (-not $Rebuild) { Write-Host "Use -Rebuild to replace existing monthly processed/canonical outputs after an ABI or raw-data correction." }
