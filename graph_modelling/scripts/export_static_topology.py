from __future__ import annotations

import csv
import sys
from pathlib import Path

# Reuse the ETL package's verified Drive-copy and storage-root helpers while
# keeping this graph export command physically separate from ETL execution.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ETL_SOURCE = WORKSPACE_ROOT / "etl" / "src"
sys.path.insert(0, str(ETL_SOURCE))

from utils import (  # noqa: E402
    copy_to_drive_with_retries,
    load_yaml,
    resolve_storage_root,
    staging_path,
)


def write_csv(cfg, root: Path, name: str, header: list[str], rows: list[list]):
    stage = staging_path(cfg, "tigergraph", "static", name)
    with stage.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    out = root / "tigergraph" / "static" / name
    copy_to_drive_with_retries(stage, out, cfg)
    print(f"[OK] {name}: {len(rows):,} rows")


def main():
    cfg = load_yaml("config/extraction.yaml")
    reg = load_yaml("config/contracts.yaml")
    root = resolve_storage_root(cfg, require=True)
    assets = reg["assets"]
    contracts = reg["contracts"]

    asset_rows = [
        [
            s,
            m["address"],
            m["decimals"],
            m.get("family", "support"),
            int(bool(m.get("target_asset"))),
        ]
        for s, m in assets.items()
    ]
    protocols = sorted({m["protocol"] for m in contracts.values()})
    protocol_rows = [[p] for p in protocols]
    pool_rows = []
    oracle_rows = []
    asset_pool = []
    pool_protocol = []
    asset_protocol = []
    asset_oracle = []
    wrappers = []

    for key, m in contracts.items():
        if m.get("role") == "dex_pool":
            pool_rows.append(
                [key, m["address"], m["protocol"], m.get("state_kind", "")]
            )
            pool_protocol.append([key, m["protocol"]])
            if m.get("protocol") == "uniswap_v3":
                for pos, t in enumerate([m["token0"], m["token1"]]):
                    asset_pool.append([t, key, pos])
            else:
                for pos, c in enumerate(m.get("coins", [])):
                    asset_pool.append([c["symbol"], key, pos])
        elif m.get("role") == "oracle":
            oracle_rows.append([key, m["address"], m.get("asset", "")])
        elif m.get("role") == "data_provider" and m.get("protocol") == "aave_v2":
            for reserve in m.get("reserves", []):
                asset_protocol.append([reserve["symbol"], "aave_v2", "LENDING_RESERVE"])

    asset_oracle += [
        ["WBTC", "chainlink_btc_usd", "REFERENCE_PRICE"],
        ["stETH", "chainlink_eth_usd", "UNDERLYING_REFERENCE"],
        ["wstETH", "chainlink_eth_usd", "UNDERLYING_REFERENCE"],
    ]
    wrappers.append(["stETH", "wstETH", "WRAPPED_AS"])

    write_csv(
        cfg,
        root,
        "vertices_asset.csv",
        ["asset_id", "address", "decimals", "family", "target_asset"],
        asset_rows,
    )
    write_csv(cfg, root, "vertices_protocol.csv", ["protocol_id"], protocol_rows)
    write_csv(
        cfg,
        root,
        "vertices_pool.csv",
        ["pool_id", "address", "protocol", "state_kind"],
        pool_rows,
    )
    write_csv(
        cfg, root, "vertices_oracle.csv", ["oracle_id", "address", "feed"], oracle_rows
    )
    write_csv(
        cfg,
        root,
        "edges_asset_in_pool.csv",
        ["asset_id", "pool_id", "position"],
        asset_pool,
    )
    write_csv(
        cfg,
        root,
        "edges_pool_operated_by.csv",
        ["pool_id", "protocol_id"],
        pool_protocol,
    )
    write_csv(
        cfg,
        root,
        "edges_asset_lending_protocol.csv",
        ["asset_id", "protocol_id", "relation"],
        asset_protocol,
    )
    write_csv(
        cfg,
        root,
        "edges_asset_oracle.csv",
        ["asset_id", "oracle_id", "relation"],
        asset_oracle,
    )
    write_csv(
        cfg,
        root,
        "edges_asset_wrapper.csv",
        ["from_asset", "to_asset", "relation"],
        wrappers,
    )
    print(
        "\nStatic TigerGraph topology exported. Do not load hourly state as a single overwriting edge yet; temporal observations are built after canonicalization."
    )


if __name__ == "__main__":
    main()
