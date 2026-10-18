param(
    [string]$DriveRoot = ""
)

$ErrorActionPreference = "Continue"
Write-Host "=== AssetGuardX Raw v2 setup check ==="
Write-Host ""

Write-Host "Python:"
python --version

Write-Host "`nPython dependencies:"
python -c "import google.cloud.bigquery, pyarrow, pandas, yaml, requests, eth_abi, eth_utils; print('Dependencies: OK')"

Write-Host "`nGCP project:"
try { gcloud config get-value project } catch { Write-Warning "gcloud not found or project not configured." }

Write-Host "`nApplication Default Credentials:"
python -c "import google.auth; c,p=google.auth.default(); print('ADC: OK; project=' + str(p))"

if ($DriveRoot) { $env:ASSETGUARDX_DRIVE_ROOT = $DriveRoot }
Write-Host "`nGoogle Drive mount:"
$found = $null
if ($env:ASSETGUARDX_DRIVE_ROOT -and (Test-Path $env:ASSETGUARDX_DRIVE_ROOT)) {
    $found = $env:ASSETGUARDX_DRIVE_ROOT
} else {
    foreach ($letter in 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z') {
        $candidate = "${letter}:\My Drive"
        if (Test-Path $candidate) { $found=$candidate; break }
    }
}
if ($found) { Write-Host "FOUND: $found" } else { Write-Warning "No mounted 'My Drive' detected. Set ASSETGUARDX_DRIVE_ROOT or pass -DriveRoot." }

Write-Host "`nLocal scratch:"
$scratch = if ($env:ASSETGUARDX_SCRATCH_ROOT) { $env:ASSETGUARDX_SCRATCH_ROOT } else { "$env:LOCALAPPDATA\AssetGuardX\scratch" }
Write-Host $scratch
try {
    New-Item -ItemType Directory -Force -Path $scratch | Out-Null
    $drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($scratch).Substring(0,1)) -ErrorAction Stop
    Write-Host ("Scratch free space: {0:N2} GiB" -f ($drive.Free / 1GB))
} catch { Write-Warning "Could not inspect scratch free space: $_" }

Write-Host "`nArchive RPC:"
if ($env:ASSETGUARDX_ETH_RPC_URL) { Write-Host "ASSETGUARDX_ETH_RPC_URL is set." } else { Write-Warning "ASSETGUARDX_ETH_RPC_URL is NOT set." }

Write-Host "`nImportant: v2 writes Parquet to local scratch first, closes it, then copies it to Drive. It never writes an open Parquet file directly to DriveFS."
