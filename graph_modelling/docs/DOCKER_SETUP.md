# Historical graph integration v1

This is a bounded-window implementation, not a full-history production server.
The schema and query have **not yet been compiled against a live TigerGraph
server**. Python transport tests are not a substitute for this integration gate.

## Install and secure the local pilot

1. Check Docker Desktop is running and measure free local disk before loading a
   large graph. The database volume is local; do not put it in streamed Drive.
2. Start Docker Desktop's Linux/WSL2 engine. Allow four CPUs and at least 8 GB RAM;
   stop the database during GPU training if the laptop cannot support both.
3. The Compose file pins the official TigerGraph Community Edition 4.2.5 image
   by digest (`tigergraph/community@sha256:190c505bf5c22c90e597b901ea8fcfaf2da7e883b688d7d48c8fe60af0a20ddb`).
   This is the free Community image, not the Savanna service or an Enterprise trial.
4. From `graph_modelling`, validate and start the local container:

   ```powershell
   docker compose config
   docker compose up -d
   docker exec -it --user tigergraph assetguardx-tigergraph bash
   ```

5. Inside the container, change the Linux user and database passwords, start
   services with `gadmin start all`, then install on a **fresh graph**:

   ```text
   passwd
   /home/tigergraph/tigergraph/app/cmd/gsql ALTER PASSWORD tigergraph
   /home/tigergraph/tigergraph/app/cmd/gadmin start all
   /home/tigergraph/tigergraph/app/cmd/gsql /home/tigergraph/mydata/schema/schema_v1.gsql
   /home/tigergraph/tigergraph/app/cmd/gsql /home/tigergraph/mydata/schema/queries_v1.gsql
   ```

   Confirm authentication is enabled and issue a limited-privilege REST token
   using the installed version's security guide. Never commit the token.
   Set it in the VS Code terminal as `ASSETGUARDX_TIGERGRAPH_TOKEN`.
6. Verify server version, edition, authentication and successful query installation.
   Compose only starts a container; it does not certify these checks.

The database is stored in a named **local Docker volume**, never Google Drive.
Only localhost is exposed, and SSH is not exposed. Do not run `down -v`: it
deletes the working database volume. Rebuild from archived canonical releases.
The container's port 9000 is mapped to host port 19000 because another local
Python service already listens on 127.0.0.1:9000. RESTPP remains at 14240.

This deployment follows the official [Docker guide](https://www.tigergraph.com/docs/tigergraph-server/4.2/getting-started/docker).
API structures follow [REST upsert](https://www.tigergraph.com/docs/tigergraph-server/4.2/api/upsert-rest)
and [installed-query endpoints](https://www.tigergraph.com/docs/tigergraph-server/4.2/api/built-in-endpoints).

## Optional TigerGraph Cloud path

Docker is not required when a TigerGraph Cloud cluster is available. Create a
small Cloud workspace, install `schema_v1.gsql` and `queries_v1.gsql` through
the Cloud GraphStudio/GSQL tools available for that plan, and create a database
secret. Cloud RESTPP uses HTTPS on port 443 and supports database-secret or
bearer-token authentication.
Cloud free-tier availability and lifecycle limits are account/plan dependent;
do not treat it as the only copy of the research release. Archive canonical
releases and exports on Google Drive.

The Python adapter rejects remote endpoints by default. Opt in explicitly in
the VS Code terminal, without changing tracked configuration or committing a
token:

```powershell
$env:ASSETGUARDX_TIGERGRAPH_URL = "https://<cluster-domain>:443/restpp"
$env:ASSETGUARDX_TIGERGRAPH_ALLOW_REMOTE = "true"
$env:ASSETGUARDX_TIGERGRAPH_SECRET = "<database-secret>"
```

The adapter sends that value as `Authorization: GSQL-Secret`. A short-lived
RESTPP bearer token may be used instead by setting
`ASSETGUARDX_TIGERGRAPH_TOKEN` and omitting the secret variable.

Then run the same `graph-check`, `graph-load`, and `graph-export` commands
below. The database remains the working copy in TigerGraph; Drive stores the
immutable fixture/release and exported snapshots. A raw Parquet log manifest
must first be decoded into a canonical `GraphRelease`; it must not be loaded
directly as economic graph edges.

## Seven-day engineering pilot

The following creates **synthetic test data**, not real Ethereum data or thesis evidence:

```powershell
$archive = 'G:\My Drive\AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1'
$graphFixture = Join-Path $archive 'tigergraph\samples\synthetic-seven-day-demo-v1.json'
python -m assetguardx_graph.cli offline-demo --output $graphFixture
python -m assetguardx_graph.cli graph-check --input $graphFixture
python -m assetguardx_graph.cli graph-load --input $graphFixture
$snapshot = Join-Path $archive 'tigergraph\exports\synthetic-seven-day-demo-exposure.json'
python -m assetguardx_graph.cli graph-export --input $graphFixture --at 2023-01-01T00:00:00Z --view exposure --output $snapshot
```

Run `graph-load` twice to check idempotence. Query an out-of-window date to check
explicit failure. Measure `docker stats --no-stream`, disk growth and runtime.
The synthetic pilot checks mechanics only; benchmark a verified seven-day real
graph before attempting thirty days plus seven preceding context days.

## Data contract and immutability

- Stable entities have explicit asset/pool/market/oracle/wrapper kinds.
- `AGXRecord` vertices hold timestamped states and directional relationships.
  `AGX_SOURCE` and `AGX_TARGET` connect records to release-scoped entities.
- Content-addressed IDs incorporate the release and full record. A release ID
  cannot be reused for different content through the adapter.
- Availability time and observation time are checked independently. Query
  results are reduced to the newest eligible version *before* expiry is checked.
  A missing or expired recent value does not resurrect an older one.
- A loaded-window marker is published only after canonical record read-back parity.
  Failed/partial loads may be retried with the same immutable input.
- One loader per database at a time is required. Distributed concurrent release
  publishing is not implemented. Restrict direct database writes to the loader.
- Query `agx_records_v1` supplies canonical as-of records. Python functions provide
  neighbourhoods and exposure changes using those snapshots. Dedicated optimized
  GSQL path/evidence/coverage queries are still scheduled work.
- Export carries stable node ordering, typed relation records, freshness masks,
  observation ages, schema identity and release/snapshot hashes. Export fails if
  the database and canonical input disagree. This is the required training path;
  offline test/demo graphs must not be represented as DB-validated research inputs.

The v1 payload keeps decimal amounts as strings to avoid precision loss.
Timestamps use UTC whole-second precision. Canonical per-window JSON releases
are immutable rebuild inputs; verified point-in-time snapshots can be cached as
partitionable Parquet feature rows with per-snapshot hashes and a manifest.
Keep TigerGraph loads bounded and benchmark resource use before accumulating
many historical releases in its 8-GB local volume.

## Live acceptance checklist

- [ ] Community image/digest and local resource footprint recorded.
- [ ] Credentials changed and REST authentication enforced.
- [ ] GSQL schema/query compile on the selected version.
- [ ] Real server supports the expected RESTPP response shape and idempotent loads.
- [ ] Entity links and exported record contents match canonical inputs.
- [ ] Interrupted load cannot publish a successful release marker.
- [ ] Unloaded dates fail; availability and release isolation tests pass live.
- [ ] Real seven-day release benchmark and thirty-day resource gate pass.
- [ ] Archived exports can be read from a second runtime after Drive sync.
