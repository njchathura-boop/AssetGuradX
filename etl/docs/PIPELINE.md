# AssetGuardX processing pipeline

The package deliberately separates **source acquisition** from **research transformations**. The goal is to query Ethereum/BigQuery once for a frozen scope and then make every later experiment reproducible from the Drive archive.

## Stage A — one-time raw acquisition

`contract verification -> extraction-definition snapshot -> BigQuery source events and month-bounded raw logs -> ABI decode -> event consolidation -> RPC hourly state -> validation -> RAW_FROZEN.json`

BigQuery source files are partitioned by month. Each Parquet file is written to local scratch first, closed, and only then copied to Google Drive in bounded chunks. The local file is deleted only after the Drive copy has the expected size. This avoids the previous failure mode in which PyArrow kept an open Parquet writer directly on the DriveFS virtual filesystem and Windows returned error 1450.

Raw BigQuery sources:
- selected protocol decoded events;
- event-level token activity (stablecoin mint/burn, WBTC/wstETH transfers);
- one last-block anchor per UTC hour.

Raw archive-RPC sources:
- Chainlink ETH/USD and BTC/USD round state and timestamps;
- Uniswap V3 `slot0` and active liquidity for selected pools;
- Curve balances, A, fee, virtual price, and configured hourly marginal `get_dy` / `get_dy_underlying` quotes in both directions;
- Aave V2 reserve cash, aToken/debt-token supplies, reserve configuration and normalized indexes;
- Lido stETH pooled-ETH/share/supply state;
- wstETH conversion ratio;
- Lido WithdrawalQueue request/finalization state.

`verify_contracts.py` checks token metadata, Uniswap token0/token1/fee, Curve coins and every configured Curve quote function/index, Aave reserve token addresses, Chainlink feed calls, and Lido state methods before the expensive extraction starts.

The Aave/Curve/Lido raw-log windows stop at UTC calendar-month boundaries. The logs are ABI-decoded and merged directly into the appropriate monthly `raw/protocol_events` file. Consolidation verifies row and fingerprint parity and rejects out-of-month timestamps. `freeze_raw_archive.py` requires deep validation and the v2 calendar-month marker. **No downstream script imports or calls BigQuery.** A new BigQuery extraction is needed only if the research scope itself changes (new asset/pool/protocol/event type).

## Stage B — canonicalization from Drive only

`run_process.ps1` reads the consolidated monthly `raw/protocol_events` files and converts them into typed/intermediate tables. Unknown and malformed raw-log records remain in the canonical archive with explicit status, error, and original topics/data; they are excluded from semantic event tables rather than mis-decoded. Use `-Rebuild` after ABI or source corrections so monthly derivatives are replaced:

```powershell
.\run_process.ps1 -DriveRoot "G:\My Drive" -Rebuild
```

The processing stage creates:
- `token_flows_hourly`
- `protocol_event_counts_hourly`
- `dex_swap_events`
- `uniswap_liquidity_events`
- `curve_liquidity_events`
- `aave_events`
- `aave_auxiliary_token_events`
- `lido_events`
- `lido_withdrawal_events`
- `lido_token_transfer_events`
- `lido_withdrawal_nft_transfer_events`
- `lido_withdrawal_nft_token_approval_events` (ERC-721 approved address and token ID)
- `lido_withdrawal_nft_operator_approval_events` (`ApprovalForAll` boolean)
- `curve_lp_token_transfer_events`
- `curve_lp_token_approval_events`
- `token_approval_events`
- `curve_parameter_events`
- `raw_log_decode_quality_hourly`
- `dex_activity_hourly`
- `oracle_hourly`
- `uniswap_pool_state_hourly`
- `curve_pool_state_hourly`
- `aave_reserve_state_hourly`
- `lido_steth_state_hourly`
- `lido_wsteth_state_hourly`
- `lido_withdrawal_queue_hourly`

It then builds `processed/asset_hourly`. Market observations are reconstructed causally from the selected Uniswap pools and point-in-time Curve quotes, with swap execution prices retained as fallback/reconciliation evidence. Reference values are constructed separately. Missingness, source, and price age are preserved instead of converting missing data to zero. Chainlink age is `snapshot_block_timestamp - oracle_updated_at_unix` in seconds, converted independently of the timestamp's Parquet precision. Negative age is retained as an anomaly rather than clamped to zero. For older archives without the exact snapshot timestamp, the UTC hour end is an explicit approximation.

Direct decoded-event rows and successfully decoded raw-log rows are already in the same canonical monthly source file before intermediate event tables are written. Lido ERC-20 transfers are kept separate from stETH accounting events, and WithdrawalQueue ERC-721 transfers are kept separate from token transfers. This stage is where token decimals, protocol accounting, hourly alignment and exposure reconstruction belong. The canonical raw files retain immutable source evidence.

`run_raw.ps1` orchestrates extraction, decoding, consolidation, validation and freeze for a new release. `run_process.ps1` consumes the frozen Drive archive; it never runs the acquisition stages. Frozen `raw/protocol_events.decode_status` records the decoder version used at acquisition time. Later ABI corrections do **not** overwrite it: `process_raw.py` re-decodes formerly unknown or revised layouts in memory and publishes corrected derived event tables plus `processed/raw_log_decode_quality_hourly`. Inspect the active quality table and the versioned ABI alongside frozen source evidence before claiming decoded-event coverage. The 2023–Q1 2024 release now decodes all 6,861 formerly unknown raw-log rows; two Curve `NewFee` rows were also re-decoded because their original field names were incorrect. A zero event count is separately assessed as zero in the selected raw archive or unverified where no raw-log fallback exists.

`run_canonical.ps1` builds missing canonical months and skips existing month files. To recompute and replace all existing canonical hourly outputs after a code correction, run `python src/build_canonical_hourly.py --force`; this reads the processed Drive tables and does not reacquire source data.

## Stage C — labels from canonical data, no BigQuery

`run_labels.ps1` is intentionally blocked until thesis thresholds are explicitly filled in `config/labels.yaml`.

For each asset/hour it can then derive strict future windows `(t, t+h]` for 6, 12 and 24 hours and maximum future absolute deviation. The current binary label is positive when **any** valid future hour crosses the configured threshold; it does not yet require the proposed two consecutive breaches or identify episode onset. Recovery-duration rules remain separate until the recovery tolerance/event definition is approved. Do not present this output as the final sustained-event thesis label until the event definition and code agree.

Because labels are derived from the frozen chronological archive, thresholds can be changed and labels regenerated without querying Ethereum again.

## Stage D — TigerGraph

Graph-specific code and Docker now live in the sibling `graph_modelling` folder. After inspecting canonical tables, `graph_modelling/run_static_export.ps1` can export stable registry topology (Asset, DEXPool, Protocol, Oracle and relations) to Drive. The timestamped graph release builder/loader is also in `graph_modelling`; it is distinct from the static CSV exporter.

Do **not** repeatedly overwrite a single Asset-to-Pool edge with hourly values. Preserve time explicitly (for example an AssetHour/PoolHour observation ID or timestamped snapshot) so the temporal graph model can reconstruct history.

See [`../../graph_modelling/README.md`](../../graph_modelling/README.md) and [`../../graph_modelling/docs/STATIC_TOPOLOGY.md`](../../graph_modelling/docs/STATIC_TOPOLOGY.md).

## Stage E — modelling transforms

Chronologically split train/validation/test first. Only after the split should transforms such as `log1p` and robust scaling be fit using **training-period statistics only**. This avoids look-ahead leakage.
