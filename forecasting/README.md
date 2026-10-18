# Temporal graph attention forecasting

The forecasting implementation lives in `forecasting/assetguardx_forecasting`.
The eight targets are USDC, USDT, DAI, FRAX, LUSD, stETH, wstETH and WBTC.
Their reference prices come from the canonical hourly tables already included
in the graph. Independent market-price reconciliation is still required.


## Separation of responsibilities

- `../etl/`: blockchain acquisition, decoding and processed hourly tables.
- `../graph_modelling/`: historical graphs, TigerGraph, Docker and snapshot exports.
- `./`: forecasting datasets/labels, GAT and GRU models, training, prediction and evaluation.

The dependency is one-way: forecasting imports the graph package's temporal
record/snapshot contracts and immutable artifact writer. Graph construction and
TigerGraph loading do not import forecasting and do not require PyTorch.
No database connection is needed for training on exported inputs.

### Forecasting files

- `assetguardx_forecasting/forecast_data.py`: graph tensors, labels and event census.
- `assetguardx_forecasting/labeling.py`: sustained-onset definitions and split purging.
- `assetguardx_forecasting/gat.py`: attention layers, GRU, normalization and family heads.
- `assetguardx_forecasting/forecasting.py`: training, checkpoint reload, inference and evaluation.
- `assetguardx_forecasting/cli.py`: six `forecast-*` commands.
- `assetguardx_forecasting/demo.py`: explicitly synthetic test fixture.
- `config/forecast.yaml`: provisional thresholds, horizons and chronological splits.
- `tests/`: numerical, temporal-safety, schema and checkpoint regression checks.

## Architecture and inputs

The network uses an edge- and relationship-aware GATv2-style scoring function,
two graph layers by default, a 32-dimensional hidden representation with four
attention heads, and a GRU reading 72 hourly snapshots. Small output heads are
shared within the stablecoin, liquid-staking and BTC-wrapper families. Only
WBTC belongs to the last family; that head is not evidence of generalization
to other BTC wrappers.

Economic entities are the training nodes. A self record becomes a node feature;
a cross-entity record becomes a typed message edge. Numeric amounts and ages
are accompanied by validity masks. Relation names include units. Reverse edges
are explicitly distinct message directions; they do not represent reverse
financial exposure. Evidence/record vertices remain in TigerGraph's audit
representation rather than becoming arbitrary extra ML nodes.

Amounts use a signed log transform. Means and standard deviations are fitted
only on valid training observations, then frozen. Node/relationship vocabulary
is frozen by `schema_until`; later unknown features or entities are counted
and ignored rather than expanding the model using the test period. Snapshots
are evaluated at their actual Ethereum availability anchors. Any missing hour
breaks a training sequence. No features are filled using future records.

The implementation is native PyTorch and needs no PyTorch Geometric extension
wheels. This makes the Windows CPU pilot easy to run. Its attention score applies
a nonlinear transform before the final attention projection, following the
[GATv2 principle](https://arxiv.org/abs/2105.14491), with additional relationship
embeddings and edge attributes. This is a project adaptation, not a claim to
reproduce the paper's benchmark results. Attention scores are internal weights,
not causal explanations or monetary sensitivities.

## Labels and evaluation policy

`config/forecast.yaml` is a **provisional research design**, separate from the
still-unapproved ETL label configuration. Defaults are 0.5% for stablecoins and
WBTC and 1% for stETH/wstETH. The target is a new episode starting within the
next 6, 12 or 24 hours, confirmed by two consecutive valid breaches. Recovery
requires three valid hours below half the entry threshold. Unknown coverage
resets knowledge of the episode state. Active or already-breaching anchors are
excluded from early-warning labels. Missing/stale prices become unknown labels.

A horizon ending on the first breach needs the following confirmation hour.
Split purging therefore uses that hour's actual availability time. Severity
targets retain maximum downside and premium within the strict forecast window.
The network's cumulative hazards ensure `p6 <= p12 <= p24`; cumulative severity
outputs are also monotone. These are probabilities of an onset **within** a
window, not the probability of a breach at exactly hour 6, 12 or 24.

The default exploratory split is January-August 2023 training, September-October
selection/early stopping, November-December calibration, and January-March
2024 testing. The current 2023-to-March-2024 archive cannot support the original
five-year thesis split. These dates are explicit defaults for the available
archive; freeze the final policy before inspecting final model performance.

Training uses a six-hour stride; this reduces overlapping training windows but
does not make them independent. Validation/inference use every hour. The trainer
requires both pooled label classes at every horizon and refuses meaningless
all-negative training. Per-asset AP is undefined when a split has only one class.
The census counts asset episodes and clusters of onsets within 24 hours; it does
not certify statistical independence. Small event counts limit scientific claims.

Training never scores the test set. `forecast-evaluate` is a separate explicit
command. Temperature calibration, when requested, uses the separate calibration
interval and never the test interval. Reports include Brier score, AP, prevalence,
severity errors, reliability bins, event recall, warning lead time and false-alarm
runs per covered asset-month. Confidence intervals across independent event/time
blocks, a full LightGBM baseline and finalized scientific validation are further
research work. The pilot never sets `research_ready` to true.

## Install and test the engineering workflow

Run from `forecasting`, with the same Python environment for every command:

```powershell
Set-Location 'C:\Users\njcha\Desktop\My Files\IITM\FINAL_YEAR_PROJECT\forecasting'
python -m pip install -r requirements.txt
$archive = 'G:\My Drive\AssetGuardX_IITM\thesis_2023_Q1_2024_raw_v2_1'
$models = Join-Path $archive 'forecasting'
python -m assetguardx_forecasting.cli forecast-smoke --output-dir (Join-Path $models 'synthetic-smoke-v1')
```

The requirements install both local projects in editable mode; use this folder
as the working directory during installation. This keeps one copy of the graph
contract without mixing the training implementation into the graph project.
For automated tests, install `requirements-dev.txt` and run `python -m pytest`.
The source/config layout currently supports this documented editable workflow.

Existing Drive artifacts under `modelling/synthetic-gat-smoke-v1` are unchanged
and remain readable by the new package. New commands above write under
`forecasting/` in the same archive. This code move does not relocate, overwrite
or delete immutable archive artifacts.

This creates an explicitly synthetic fixture, dataset, a trained checkpoint,
training report and 24 prediction rows (eight assets by three horizons). It
tests forward/backward propagation, labels, reload and inference. It downloads
no blockchain data and needs no running TigerGraph. The directory must be new;
choose `synthetic-smoke-v2` for a separate run. Synthetic metrics are not thesis
results. The current installed PyTorch build was CPU-only; CUDA use requires a
CUDA-capable PyTorch environment and its own resource/determinism check.

## Prepare real data and count episodes

First inspect the entire stored market series without loading more data into
TigerGraph or downloading anything:

```powershell
python -m assetguardx_forecasting.cli forecast-census --archive-root $archive
```

This is a read-only, provisional census of the processed market table. It reports
observed onsets, active/unknown episode hours, valid-price hours and eligible
forecast labels. Its timestamps are market hour labels; it does not manufacture
prediction availability times or write a trainable graph dataset.

After `graph-cache` has produced the hourly verified snapshots, prepare all
available cache files together. Each input needs its `.parquet.manifest.json`.
Preparation checks file, record and snapshot hashes and rejects future records,
expired records, disagreement between overlapping windows or mixed synthetic
and real sources. Readback evidence is preserved in the resulting manifest.

```powershell
$caches = @(Get-ChildItem (Join-Path $archive 'tigergraph\cached_snapshots') -Filter '*.parquet' -File | Select-Object -ExpandProperty FullName)
if ($caches.Count -eq 0) { throw 'Create verified graph-cache files first.' }
$dataset = Join-Path $models 'datasets\graph-hourly-v1.npz'
python -m assetguardx_forecasting.cli forecast-prepare --cache $caches --output $dataset
python -m assetguardx_forecasting.cli forecast-census --input $dataset
```

A seven-day or 37-day release does not cover the default training, validation
and test intervals. Such a release is useful for preparation/census tests. The
trainer gives a clear error until enough periods and event classes are present.
Cache subsequent bounded graph windows using the [graph workflow](../graph_modelling/README.md).

For an engineering preparation test while Docker is unavailable, use a canonical
release instead. This path is explicitly marked `database_verified: false`:

```powershell
$release = Join-Path $archive 'tigergraph\releases\exposure-2023-01-08-to-2023-02-07.json'
$pilot = Join-Path $models 'datasets\canonical-37-day-pilot-v1.npz'
python -m assetguardx_forecasting.cli forecast-prepare --releases $release --output $pilot
python -m assetguardx_forecasting.cli forecast-census --input $pilot
```

Short-period training needs explicit split overrides AND a vocabulary freeze
at or before that earlier training end (`forecast-prepare --schema-until ...`).
It also needs `--allow-canonical` and `--allow-provisional` where applicable.
These switches describe an engineering pilot; they do not certify research data.

## Train and compare models

Once the dataset contains the configured periods, run:

```powershell
$run = Join-Path $models 'runs\gat-seed42-v1'
python -m assetguardx_forecasting.cli forecast-train --input $dataset --output-dir $run --architecture gat --epochs 20 --batch-size 8 --device cpu --allow-provisional
```

Use the same data, split dates, seed, epoch budget and dimensions for `--architecture
mean` (uniform incoming aggregation with the same relationship/edge messages)
and `--architecture gru` (no graph message passing). The report includes a
training-prevalence probability baseline. Compare selected models with multiple
fixed seeds after development. No model is assumed to outperform its controls.

Each run records its source hashes, frozen tensor schema, training-only scaler,
label definitions, split boundaries, code hash, seed, model settings, runtime
version and parameter count. `INCOMPLETE.json` remains after interrupted training;
an incomplete run cannot be used for prediction. Use a new directory to retry;
automatic mid-epoch resume is not implemented in this first trainer.

## Predict and evaluate

```powershell
$checkpoint = Join-Path $run 'model.pt'
$predictions = Join-Path $models 'predictions\gat-latest-v1.parquet'
python -m assetguardx_forecasting.cli forecast-predict --input $dataset --checkpoint $checkpoint --latest --output $predictions
```

Without `--latest`, use `--start-time` and `--end-time` to select historical
inference times. The predictor uses only its lookback sequence at each time;
future labels are not model inputs. At the dataset tail it can still forecast
when the future outcome is unknown. Active/breaching/unknown episode states
receive an explicit status and null forecasts. The file includes asset, family,
actual as-of timestamp, horizon, onset probability, downside/premium severity,
training event support, provenance and calibration status. This is historical
model inference; continuous live acquisition is not part of this command.

When development is frozen, score the held-out period explicitly:

```powershell
python -m assetguardx_forecasting.cli forecast-evaluate --input $dataset --checkpoint $checkpoint --split test --output-dir (Join-Path $models 'evaluation\locked-gat-v1')
```

Permanent datasets/checkpoints/predictions belong on Drive. Source code and the
Python environment stay local. TigerGraph uses its existing local Docker volume;
training consumes archived tensors and does not query the database per minibatch.
