# Load a small real-data sample into TigerGraph

This is an integration test of the existing `AssetGuardX` graph, not a thesis
stress result. The default sample reads **one day** of processed Aave V2
reserve-cash observations for DAI, USDC, and WBTC from the Drive archive. It
creates 6 entity vertices and 72 timestamped record vertices, with source and
target edges. The source Parquet file and frozen raw archive are not changed.
No BigQuery or Ethereum RPC calls are made.

Run these commands in PowerShell from `graph_modelling`:

```powershell
$archive = 'G:\My Drive\AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1'
$sample = '.\samples\aave-v2-2023-01-01-24h.json'
python -m assetguardx_graph.cli real-sample --archive-root $archive --date 2023-01-01 --hours 24 --assets USDC DAI WBTC --output $sample
python -m assetguardx_graph.cli graph-check --input $sample
```

The generated JSON is immutable: repeating the command with identical source
data is safe, but it refuses to overwrite different content at the same path.
The script rejects missing/duplicate asset-hours, invalid quality status,
inconsistent decimals, and samples longer than 168 hours. It does not silently
turn gaps into zero values.

The TigerGraph `AssetGuardX` schema and installed `agx_records_v1` query must be
present. For the local Docker database, configure the URL and a limited token in
the **same PowerShell terminal**. For TigerGraph Cloud, use its HTTPS `/restpp`
URL and follow the explicit remote opt-in instructions in the main README.
Never paste a secret into source code, a notebook, or chat.

Then load and query the sample:

```powershell
python -m assetguardx_graph.cli graph-load --input $sample
python -m assetguardx_graph.cli graph-verify --input $sample --at 2023-01-01T23:59:59Z --view exposure
```

`graph-load` checks every record by querying it back before publishing the
release marker. `graph-verify` independently reads the database and checks the
point-in-time snapshot against the local release. On success, it prints
`database_verified: true` and **3 active records** at the example timestamp.
Other releases already in the database can make GraphStudio's overall counts
larger; use the release ID in the command output to distinguish this test.

The records represent **native-token Aave reserve available liquidity**, not
USD-denominated loss, account positions, or liquidation risk. The sample is
only for checking real-data loading, timestamps, provenance, and TigerGraph
round-trip behaviour. It is not a complete exposure graph or MSTG.
