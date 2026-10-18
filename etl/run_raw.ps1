param(
    [switch]$Execute,
    [switch]$Force,
    [switch]$SkipVerify,
    [switch]$SkipRpc,
    [string]$DriveRoot = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($DriveRoot) { $env:ASSETGUARDX_DRIVE_ROOT = $DriveRoot }

Write-Host "=== AssetGuardX Raw v2 ==="
Write-Host "Study  : 2023-01-01 through 2023-12-31"
Write-Host "Holdout: 2024-01-01 through 2024-03-31"
Write-Host ("Mode   : " + $(if ($Execute) { "EXECUTE" } else { "DRY-RUN" }))
Write-Host "Design : local scratch -> closed Parquet -> verified Drive copy"
Write-Host "Output : AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1"
Write-Host ""

if ($Execute -and -not $env:ASSETGUARDX_DRIVE_ROOT) {
    throw "Set Google Drive first, e.g. -DriveRoot 'G:\My Drive'."
}

$frozenMarker = if ($env:ASSETGUARDX_DRIVE_ROOT) {
    Join-Path $env:ASSETGUARDX_DRIVE_ROOT "AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1\metadata\RAW_FROZEN.json"
} else { $null }
if ($Execute -and $frozenMarker -and (Test-Path -LiteralPath $frozenMarker)) {
    if ($Force) { throw "This archive is frozen. Do not overwrite it with -Force; create a new release for changed source data." }
    python src/validate_raw_archive.py --deep
    if ($LASTEXITCODE -ne 0) { throw "Frozen archive failed validation; inspect it before extracting." }
    Write-Host "Raw archive is already frozen and valid. No BigQuery or RPC calls were made."
    return
}

if ($Execute -and -not $SkipVerify) {
    Write-Host "=== 1/8 Contract verification ==="
    python src/verify_contracts.py --execute
    if ($LASTEXITCODE -ne 0) { throw "Contract verification failed." }
}

if ($Execute) {
    Write-Host "=== 2/8 Extraction-definition snapshot ==="
    python src/snapshot_metadata.py
    if ($LASTEXITCODE -ne 0) { throw "Metadata snapshot failed." }
}

Write-Host "=== 3/8 BigQuery monthly sources ==="
$argsBQ = @()
if ($Execute) { $argsBQ += '--execute' }
if ($Force) { $argsBQ += '--force' }
python src/extract_bigquery.py --dataset all @argsBQ
if ($LASTEXITCODE -ne 0) { throw "BigQuery extraction failed. Re-run the same command to resume; completed Drive files and complete staged files are reused." }

Write-Host "=== 4/8 Raw protocol logs ==="
if ($Execute) {
    python src/consolidate_protocol_events.py --check-complete
    $unifiedComplete = ($LASTEXITCODE -eq 0)
} else {
    $unifiedComplete = $false
}
if (-not $unifiedComplete) {
    python src/extract_bigquery.py --dataset raw_protocol_logs @argsBQ
    if ($LASTEXITCODE -ne 0) { throw "Raw-log extraction failed. Re-run to resume completed windows." }
    if ($Execute) {
        Write-Host "=== 5/8 Decode and consolidate monthly events ==="
        $decodeArgs = @()
        if ($Force) { $decodeArgs += '--force' }
        python src/decode_raw_logs.py @decodeArgs
        if ($LASTEXITCODE -ne 0) { throw "Raw-log decoding failed." }
        python src/consolidate_protocol_events.py --execute --remove-separate-source
        if ($LASTEXITCODE -ne 0) { throw "Protocol-event consolidation failed." }
        python src/consolidate_protocol_events.py --check-complete
        if ($LASTEXITCODE -ne 0) { throw "Unified monthly protocol archive is incomplete." }
    }
} else {
    Write-Host "Unified protocol events already verified; skipping raw-log download."
}

if ($Execute -and -not $SkipRpc) {
    Write-Host "=== 6/8 Archive-RPC hourly state ==="
    $argsRPC = @('--execute')
    if ($Force) { $argsRPC += '--force' }
    python src/extract_rpc_state.py @argsRPC
    if ($LASTEXITCODE -ne 0) { throw "RPC extraction failed. Re-run the same command to resume." }

    Write-Host "=== 7/8 Raw completeness + event coverage ==="
    python src/validate_raw_archive.py --deep
    if ($LASTEXITCODE -ne 0) { throw "Raw archive is incomplete. Do not freeze yet." }
    python src/audit_event_coverage.py --strict
    if ($LASTEXITCODE -ne 0) { throw "A configured event contract has zero extracted rows. Review before freezing." }

    Write-Host "=== 8/8 Freeze raw archive ==="
    python src/freeze_raw_archive.py
    if ($LASTEXITCODE -ne 0) { throw "Raw freeze manifest failed." }

    Write-Host "`nRAW V2 COMPLETE. From this point, canonical tables and labels use the stored archive. Graph export/load is maintained separately in ..\graph_modelling; BigQuery is not needed unless scope changes."
} elseif (-not $Execute) {
    Write-Host "`nDry-run finished. Contract/RPC calls and downloads were not executed."
}
