# AssetGuard-X graph modelling and TigerGraph

This folder owns graph-specific work: graph release construction from the ETL
archive, historical graph structure, TigerGraph schema and queries, Docker, and
graph export/load/verification. The separate `etl` folder owns blockchain
extraction and transformation; `forecasting` owns labels, model training and
prediction. ETL outputs remain in Google Drive; TigerGraph's
working database stays in a local Docker volume. Versioned JSON graph releases
and exports can be saved to Drive by passing a Drive path explicitly.

## Folders

- `assetguardx_graph/`: temporal graph contracts, Parquet-to-graph builders, REST adapter, CLI.
- `schema/`: GSQL schema and installed-query definitions.
- `config/`: graph endpoint and bounded-window settings.
- `samples/`: small checked-in sample release for engineering checks.
- `scripts/export_static_topology.py`: ETL-registry-based CSV topology export.
- `compose.yaml`: pinned TigerGraph Community container and persistent local volume.
- `docs/`: Docker setup, real sample workflow, and static-topology explanation.

There are two different export products. The static exporter makes CSVs for
stable assets/pools/protocol relations. The temporal loader reads a canonical
`GraphRelease` JSON, writes timestamped records to TigerGraph, reads them back,
and checks parity before marking the release complete. Do not load raw Ethereum
logs directly as financial graph edges.

## Install Python dependencies

Run in PowerShell from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```

To install the optional read-only graph explorer, also run
`python -m pip install -r requirements-ui.txt` and start it with
`python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1`.

## Start TigerGraph Docker

Docker Desktop must be running with its Linux/WSL2 engine. The Compose file
limits the container to 4 CPUs and 8 GB RAM. It pins the verified TigerGraph
Community image digest and exposes services only on localhost. Host port `19000`
is used for the container's port `9000` because another local service occupied
host port `9000`; RESTPP is on `14240`.

```powershell
docker compose config
docker compose up -d
docker ps --filter "name=assetguardx-tigergraph"
```

Database files are held in the named Docker volume
`assetguardx-graph_assetguardx-tigergraph-data`, not in Google Drive and not in
this source folder. Do not use `docker compose down -v`; `-v` deletes that data.
To enter the container:

```powershell
docker exec -it --user tigergraph assetguardx-tigergraph bash
```

Inside the container, follow [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md) to
secure the initial credentials and install the schema/query. Schema installation
is not automatic when the container starts.

## What goes into the historical graph

The archive already has the core observations needed for a market-level temporal
dependency graph; no new BigQuery extraction is required to build its first
version. A graph release contains stable asset, reference, DEX-pool, Aave-reserve,
Chainlink-feed, and Lido-contract entities. Timestamped records include canonical
market/reference prices, selected-pool liquidity/state, hourly decoded swap
volumes and counts, Aave reserve liquidity/supply/debt/utilization and risk
parameters, hourly supply/withdrawal/borrow/repay flows, liquidation summaries,
Lido conversion/accounting/withdrawal-queue state, and Chainlink
answers/freshness. Each record retains units, observation and
availability times, quality, method, and Parquet/block/event-hash evidence.
The stored market-price series is built from this selected on-chain pool
universe; independent price reconciliation is still a research gate, not a
completed validation claim.

DEX swaps and Aave liquidations are grouped hourly before loading; raw logs do
not become millions of graph vertices. `AGXRecord` is the timestamped record
vertex, connected to stable entity vertices. Thus the event type/direction and
amount live in a versioned record, not as an asset node. Aave reserves are
aggregate market states, not wallet positions; this graph cannot establish an
individual borrower's health, actual bad debt, or a causal liquidation cascade.

The builder accepts an evaluation interval up to 30 days and up to 168 preceding
context hours. The first archive week has no earlier data, so use zero context
there. Each data row becomes available at its actual recorded hourly Ethereum
block anchor; snapshot/cache timestamps use those anchors, not the beginning of
the hour. The existing hourly state inputs remain available even if the optional
size-aware pool-quote supplement has not been collected.

## Build, check, and load a real historical release

Run from `graph_modelling` after installing `requirements-dev.txt`. First make a
small seven-day real-data release; this is deliberately separate from the old
synthetic/Aave-only fixture:

```powershell
$archive = 'G:\My Drive\AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1'
$release = Join-Path $archive 'tigergraph\releases\exposure-2023-01-01-to-2023-01-08.json'
python -m assetguardx_graph.cli real-release `
  --archive-root $archive `
  --start-time 2023-01-01T00:00:00Z `
  --end-time 2023-01-08T00:00:00Z `
  --context-hours 0 `
  --output $release
python -m assetguardx_graph.cli graph-check --input $release
```

`graph-check` is local validation only. Start the local database, install the
schema and `agx_records_v1` query once as described in `docs/DOCKER_SETUP.md`,
then set the endpoint and short-lived token in this same PowerShell terminal.
Never save credentials in the release or repository.

```powershell
$env:ASSETGUARDX_TIGERGRAPH_URL = 'http://127.0.0.1:14240/restpp'
$env:ASSETGUARDX_TIGERGRAPH_TOKEN = '<short-lived-local-rest-token>'
python -m assetguardx_graph.cli graph-load --input $release
python -m assetguardx_graph.cli graph-verify --input $release --at 2023-01-07T23:59:59Z --view exposure
```

The loader batches writes, reads the records back, verifies release identity and
content parity, and only then commits its loaded-window marker. An out-of-window
query must fail explicitly. Do not run simultaneous loaders against this DB.

## Cache verified snapshots

After the seven-day load, make an immutable Parquet cache of hourly point-in-time
records. The command reads the release from TigerGraph once, creates each graph
at its actual availability anchor, compares every cached snapshot hash with the
canonical release, and directly re-queries TigerGraph at the first, middle, and
last timestamps. It rejects any record unavailable at the snapshot time.

```powershell
$cache = Join-Path $archive 'tigergraph\cached_snapshots\exposure-2023-01-01-to-2023-01-08.parquet'
python -m assetguardx_graph.cli graph-cache `
  --input $release `
  --start-time 2023-01-01T00:00:00Z `
  --end-time 2023-01-08T00:00:00Z `
  --every-hours 1 `
  --view exposure `
  --output $cache
```

This writes the Parquet feature table and a JSON manifest with release/file
hashes, entity catalog, per-snapshot hashes, direct-query parity points, and the
future-information check. Do not cache all full-history snapshots as expanded
JSON; Parquet rows are the compact reusable representation. Correlation and
MSTG views are not produced by this builder yet and must not be mistaken for
observed exposure edges.

For later evaluation windows, keep each evaluation interval at most 30 days and
include seven preceding context days. Example: build 2023-01-08 through
2023-02-07 with `--context-hours 168`, then continue in non-overlapping
evaluation intervals. Archive each immutable release/cache in Drive. Before
loading all periods into the 8-GB local Community container, measure the 7-day
and then 30-day database/disk footprint; the DB is a working subset, while the
Drive releases and Parquet caches remain the complete portable history.

The old `real-sample` command is retained only for a tiny Aave reserve-cash
integration fixture. The synthetic demo is engineering data, not thesis data.

## Static topology CSVs

This optional output uses the ETL contract registry and ETL's safe Drive-copy
helpers; it does not query BigQuery or RPC:

```powershell
.\run_static_export.ps1 -DriveRoot 'G:\My Drive'
```

See [`docs/STATIC_TOPOLOGY.md`](docs/STATIC_TOPOLOGY.md) for the CSV contents and
limitations. For current resource requirements and the live database checklist,
see [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md).

## Forecasting is a separate project

Labels, GAT/GRU models, tensor preparation, training, prediction and evaluation
live in [`../forecasting`](../forecasting/README.md), not in this package.
Use `python -m assetguardx_forecasting.cli` from that folder for forecast
commands. This graph package does not depend on PyTorch or the forecasting
package. Forecasting consumes exported releases/caches and reuses these temporal
graph contracts to preserve snapshot hashes and availability checks.
