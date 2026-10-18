# AssetGuardX data pipeline

This package acquires and processes the selected Ethereum data for the IITM thesis. The study covers 2023-01-01 to 2024-04-01 UTC (end exclusive): calendar year 2023 plus a three-month holdout. The eight target assets are USDC, USDT, DAI, FRAX, LUSD, stETH, wstETH, and WBTC. The selected sources are Uniswap V3, Curve, Aave V2, Lido, Chainlink, token activity, and hourly Ethereum block anchors. This is a selected research universe, not a complete copy of Ethereum.

## Where files live

- Code and `trail.ipynb` live in this local folder.
- Permanent raw and processed Parquet, metadata, manifests, and EDA reports live under `G:\My Drive\AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1`.
- Each output is staged briefly under `%LOCALAPPDATA%\AssetGuardX\scratch`, closed, copied to Drive, and then removed from scratch after the copy is verified locally. A visible file on `G:` does not prove cloud synchronization has finished; check Google Drive for desktop before relying on a second computer.

Do not edit or overwrite a release with `metadata/RAW_FROZEN.json`. Make a new versioned release if the source scope changes.

## Setup

Run from this folder in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
python -m pip install -r requirements.txt
.\check_setup.ps1 -DriveRoot "G:\My Drive"
```

Set `ASSETGUARDX_ETH_RPC_URL` to an archive-capable Ethereum RPC URL and configure Google Application Default Credentials before extraction. Keep keys out of the repository. The RPC and BigQuery calls in this package are read-only, but BigQuery queries can incur charges. Inspect the dry-run estimates and billing controls before `-Execute`.

## Extract raw data

```powershell
.\run_raw.ps1 -DriveRoot "G:\My Drive"
```

This is a dry run. For a new, unfrozen release, execute the same pipeline with:

```powershell
.\run_raw.ps1 -DriveRoot "G:\My Drive" -Execute
```

The runner verifies configured contracts, snapshots the extraction definition, acquires the selected BigQuery sources, fetches selected raw protocol logs in bounded windows, ABI-decodes them, and consolidates them into UTC calendar-month `raw/protocol_events` files. It then extracts hourly historical contract state through RPC, validates the raw archive, audits event coverage, and writes `metadata/RAW_FROZEN.json`. Raw-log query windows stop at calendar-month boundaries, so no later month-repair script is needed.

Completed extraction windows are skipped on a normal rerun. If a frozen, valid archive already exists, `-Execute` makes no new BigQuery or RPC calls. Do not use `-Force` against a frozen release.

If a provider rate-limits one RPC month, the targeted extractor remains available:

```powershell
python src\extract_rpc_state.py --execute --month 2023-01 --batch-size 5 --pause 1.0 --retries 10 --timeout 120
```

Never assume rate limiting is the cause of an unavailable historical state: the endpoint must support the required historical block.

## Inspect and process the stored archive

Run all cells in `trail.ipynb`. It reads the Drive archive and writes `metadata/eda_raw_report.json`. Confirm `READY FOR PROCESSING: True`, `unresolved topic rows = 0`, and inspect the zero-signature assessments. A passing raw-data gate does not validate downstream research products.

After an ABI-only correction, refresh event-derived tables from the frozen archive:

```powershell
python src\process_raw.py --events-only --force
```

This re-decodes previously unknown events in memory and repairs the two archived Curve `NewFee` rows whose arguments were formerly mislabeled; it does not rewrite frozen raw. The active quality table is `processed/raw_log_decode_quality_hourly`. The older `processed/decoded_raw_protocol_logs` file is a legacy derivative from before raw-log consolidation and is not used when unified `raw/protocol_events` exists.

After changing market-price or state logic, rebuild all processed tables and canonical hourly observations:

```powershell
.\run_process.ps1 -DriveRoot "G:\My Drive" -Rebuild
```

This does not query BigQuery or RPC. It creates intermediate protocol/state tables and `processed/asset_hourly` on Drive. `-Rebuild` replaces derived outputs; it does not change the frozen raw source. Without `-Rebuild`, existing processed months are skipped.

The frozen archive retains original topics, data, and original decoder status for audit. The current decoder recognizes the 12 additional event layouts observed in that archive; previously unknown rows become usable only in derived outputs. Nine zero-count Curve parameter-event signatures also have zero matching topics in the selected all-topic raw-log archive. The zero-count Uniswap WBTC/WETH `Flash` event remains **unverified** because that pool has no equivalent raw-log fallback in this release; do not call it confirmed absent. Market values preserve price source, age, and missingness; they are not silently filled from the future.

## Later research stages

`trail.ipynb` checks the structure and arithmetic of `processed/asset_hourly`, but independent market-price reconciliation remains pending. `run_labels.ps1` is intentionally gated by the unfilled thresholds in `config/labels.yaml`. Its current binary labels do not yet implement the proposed two-consecutive-breach/onset definition. Do not present them as final thesis labels until the definition and code agree. TigerGraph export, Docker, schema, graph modelling and loading are maintained in the sibling `graph_modelling` folder. Its static-topology exporter reads the ETL registry and writes CSVs to Drive; the temporal graph loader consumes a bounded canonical graph release. Research readiness remains false until these independent validations are complete.

The frozen extraction is sufficient for the planned market-level models and a historical dependency graph: it contains the eight target assets, selected Uniswap V3 and Curve pools, Aave V2 reserve states/events, Lido/wstETH states, Chainlink ETH/BTC references, token activity, and exact hourly block anchors. It is not a full-wallet archive and cannot support borrower-by-borrower liquidation health or causal cascade claims. Independent price reconciliation and the final episode labels remain required before modeling.

### Optional size-aware historical pool quotes

For additional execution-depth features, `src/extract_quote_sensitivity.py` reads the frozen hourly block anchors and performs read-only historical `eth_call` quotes on the registered Uniswap V3 and Curve pools. It does not submit transactions or modify the frozen source archive. The output is a separate, checksummed release under `supplemental/quote_sensitivity`; it is not included in `RAW_FROZEN.json` and is not a replacement for actual swap logs. Quotes at several input sizes estimate execution-price deterioration, not guaranteed fills.

From this ETL folder in PowerShell, first preview the exact call count:

```powershell
$env:ASSETGUARDX_DRIVE_ROOT = 'G:\My Drive'
python src\extract_quote_sensitivity.py `
  --start-time 2023-01-01T00:00:00Z `
  --end-time 2024-04-01T00:00:00Z `
  --every-hours 6
```

The current registry predicts roughly 117,000 `eth_call` operations at six-hour sampling; the dry run prints the exact count and fails above the configured 150,000-call cap. This is an RPC quota/time consideration, not a BigQuery scan. Confirm the provider permits historical `eth_call` at the requested blocks and check its current allowance before adding `--execute`:

```powershell
python src\extract_quote_sensitivity.py `
  --start-time 2023-01-01T00:00:00Z `
  --end-time 2024-04-01T00:00:00Z `
  --every-hours 6 `
  --batch-size 5 --pause 0.25 --retries 6 --timeout 90 `
  --execute
```

Use `--every-hours 24` to reduce the call count to about one quarter if the RPC allowance is too tight. Keep the emitted release ID; the graph builder accepts it explicitly and verifies the supplement manifest and file checksums.

See `docs/RAW_ARCHIVE_SPEC.md` and `docs/PIPELINE.md` for data definitions and validation details.
