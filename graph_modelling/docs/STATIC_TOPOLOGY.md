# Static topology CSV export

`scripts/export_static_topology.py` creates stable, non-temporal CSVs from the
ETL asset and contract registries. It is a small structural export for graph
setup and is different from the timestamped `GraphRelease` data sent through
the TigerGraph REST adapter.

## What it writes

The script creates CSV files for asset, protocol, pool, and oracle vertices, and
for relations such as asset-in-pool, pool-operated-by, lending-protocol,
asset-oracle, and wrapped-as. It reads `etl/config/contracts.yaml` and
`etl/config/extraction.yaml`, stages through the ETL scratch helper, and copies
the files into the configured Google Drive project directory under
`tigergraph/static/`.

It does not query BigQuery or Ethereum RPC. It does not export timestamps,
hourly prices, reserve states, account positions, loss weights, or labels, and it
does not load the CSVs into the temporal GSQL schema in this folder.

## Run

After checking that the ETL registry and canonical archive correspond to the
intended research release, run in PowerShell from `graph_modelling`:

```powershell
.\run_static_export.ps1 -DriveRoot 'G:\My Drive'
```

The larger temporal path starts from processed Drive Parquet, creates an
immutable bounded JSON `GraphRelease`, runs `graph-check`, then loads and
verifies it with `graph-load` and `graph-verify`. See the
[real sample guide](REAL_SAMPLE.md).
