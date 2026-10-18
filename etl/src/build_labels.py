from __future__ import annotations

import numpy as np
import pandas as pd

from utils import load_yaml, publish_dataframe, resolve_storage_root


def future_max_complete(s: pd.Series, h: int) -> pd.Series:
    """Maximum over the strict future window (t, t+h], requiring all h hourly values.

    The explicit shift matrix is intentionally used here. h is at most 24 for this
    project, so clarity and correct alignment are more important than micro-optimizing.
    """
    future = pd.concat([s.shift(-k).rename(k) for k in range(1, h + 1)], axis=1)
    complete = future.notna().sum(axis=1).eq(h)
    out = future.max(axis=1, skipna=True)
    return out.where(complete, np.nan)


def main():
    cfg = load_yaml('config/extraction.yaml')
    lab = load_yaml('config/labels.yaml')
    root = resolve_storage_root(cfg, require=True)
    files = sorted((root/'processed'/'asset_hourly').glob('year=*/month=*/*.parquet'))
    if not files:
        raise SystemExit('processed/asset_hourly does not exist yet. Run run_process.ps1 first.')

    thresholds = lab.get('thresholds', {})
    missing = [a for a,v in thresholds.items() if v is None]
    if missing:
        raise SystemExit(
            'Label thresholds are intentionally unset. Approve values in config/labels.yaml first: '
            + ', '.join(missing)
        )

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    required = {'asset','hour','relative_deviation'}
    if not required.issubset(df.columns):
        raise SystemExit(f'asset_hourly must contain {sorted(required)}')

    df['hour'] = pd.to_datetime(df['hour'], utc=True)
    df = df.sort_values(['asset','hour']).reset_index(drop=True)
    out_parts = []
    horizons = [int(x) for x in lab['horizons_hours']]

    for asset, g in df.groupby('asset', sort=False):
        g = g.copy().set_index('hour').asfreq('h')
        g['asset'] = asset
        abs_dev = pd.to_numeric(g['relative_deviation'], errors='coerce').abs()
        threshold = float(thresholds[asset])
        for h in horizons:
            fm = future_max_complete(abs_dev, h)
            g[f'future_max_abs_deviation_{h}h'] = fm
            y = (fm >= threshold).where(fm.notna())
            g[f'y_{h}h'] = pd.array(y, dtype='Int8')
        out_parts.append(g.reset_index())

    labels = pd.concat(out_parts, ignore_index=True)
    out = root/'processed'/'labels'/'labels.parquet'
    pub = publish_dataframe(labels, cfg, 'processed_labels','all',out,cfg['output'].get('compression','zstd'))
    print(f'[OK] labels: {len(labels):,} rows -> {out} ({pub["bytes"]:,} bytes)')
    print('Future windows are strict (t, t+h] and require complete hourly observations.')
    print('Recovery-duration logic remains intentionally separate until recovery_tolerance and event-definition rules are approved.')


if __name__ == '__main__':
    main()
