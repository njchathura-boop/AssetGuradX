# AssetGuardX-Raw v2 specification

AssetGuardX-Raw v2 is the permanent source archive for the fixed Ethereum Mainnet scope. Its purpose is to preserve enough source-level information once so canonical features, labels, graph structures and alternative modelling choices can later be rebuilt without BigQuery.

## Period and scope

- Core study: 2023-01-01 00:00 UTC through 2023-12-31 23:59 UTC.
- Temporal holdout: 2024-01-01 through 2024-03-31.
- End is exclusive at 2024-04-01 00:00 UTC.
- Target assets: USDC, USDT, DAI, FRAX, LUSD, stETH, wstETH, WBTC.
- ETH/WETH and 3CRV are support entities used to reconstruct market/reference relationships.
- Aave V2 is the representative lending protocol. Compound is intentionally excluded.

## BigQuery raw tables

### `protocol_events`

Selected decoded events preserve block number/time, transaction hash, log index when available, contract address/key, event signature and the original decoded argument payload as JSON.

Coverage:
- Uniswap V3: Swap, Mint, Burn, Flash for USDC/WETH, WBTC/WETH and wstETH/WETH selected pools.
- Curve: swap, liquidity add/remove and selected A/fee change events for 3pool, stETH/ETH, FRAX/USDC and LUSD/3CRV.
- Aave V2: deposit, withdraw, borrow, repay, liquidation, reserve updates, collateral-use toggles, rate-mode/rebalance and flash-loan activity.
- Lido stETH: staking/accounting/rebase/reward/withdrawal-receipt events.
- Lido WithdrawalQueue: request, finalization and claim events from its configured activation time onward.

### `token_activity_events`

Event-level transfers are deliberately selective:
- USDC/USDT/DAI/FRAX/LUSD: mint and burn only;
- WBTC and wstETH: all transfers.

The source events are retained so hourly flow aggregates can be regenerated or audited later. The package does not build a full wallet-to-wallet Ethereum graph.

### `hourly_blocks`

The last Ethereum block observed in each UTC hour. These block numbers and block timestamps are the point-in-time anchors for all historical `eth_call` snapshots.

## Archive-RPC raw table: `rpc_hourly`

One row per active entity/hour (or per Aave reserve/hour), with exact hourly anchor block and block timestamp.

State preserved:
- Chainlink ETH/USD and BTC/USD: round ID, answer and update timestamp;
- Uniswap V3 pools: `slot0` and active liquidity;
- Curve pools: coin balances, A, fee, virtual price;
- Curve market sensors: one-target-token marginal quotes in **both directions** using configured `get_dy` or `get_dy_underlying`; these are later converted using correct token decimals and same/past-hour USD anchors;
- Aave V2 selected reserves: available underlying cash, aToken supply, stable/variable debt supply, reserve configuration, normalized income and debt indexes;
- Lido stETH: total pooled ether, total shares and total supply;
- wstETH: stETH-per-wstETH conversion ratio;
- Lido WithdrawalQueue: last request/finalized IDs, locked ETH, unfinalized request count/stETH and bunker-mode state.

RPC rows with any failed required call are marked partial. Deep validation treats partial rows as a hard error before freezing.

## Data-integrity rules

- Contract/pool/feed configuration is programmatically verified before extraction.
- Every month is independent and resumable.
- A readable complete Drive Parquet is skipped on resume.
- A complete local staged Parquet left by a failed Drive upload is reused without re-running BigQuery.
- An unreadable/empty Drive file is regenerated for that month only.
- RPC completeness is checked against every configured active entity and hourly block.
- Event-level duplicates are checked where source keys permit it.
- The exact YAML/configuration used for extraction is copied into `metadata/extraction_definition_snapshot.json`.
- `metadata/RAW_FROZEN.json` inventories the completed archive.

## Intentionally excluded

- Compound: excluded by agreed scope; Aave represents lending.
- Full transactions table: selected decoded protocol events already preserve the actions needed by this economic-exposure model.
- Full receipts table: not a required model source for the fixed event-based scope.
- Traces: internal EVM call paths are not the intended economic-exposure graph.
- Full wallet graph / all ERC-20 transfers.
- Every DEX pool on Ethereum.

A later scope expansion may require a new targeted extraction for the newly added source, but it does not invalidate the frozen v2 archive.
