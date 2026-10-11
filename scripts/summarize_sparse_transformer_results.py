import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _preferred_output_root():
    candidates = [
        os.environ.get('SPARSE_TRANSFORMER_OUTPUT_ROOT', '').strip(),
        '/mnt/d/Store/Sparse_Transformer',
        '/mnt/d/Store/Sparse-Transformer',
        'outputs',
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return 'outputs'


def _safe_read_json(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _station_dirs(output_root: str):
    root = Path(output_root)
    if not root.exists():
        return []
    out = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / 'summary.json').exists():
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def _build_station_table(output_root: str):
    rows = []
    for sd in _station_dirs(output_root):
        sid = str(sd.name).zfill(8)
        s = _safe_read_json(sd / 'summary.json') or {}
        m = s.get('metrics', {}) if isinstance(s, dict) else {}

        row = {
            'station_id': sid,
            'test_loss': float(s.get('test_loss', np.nan)) if isinstance(s, dict) else np.nan,
            'nse': float(m.get('nse', np.nan)) if isinstance(m, dict) else np.nan,
            'rmse': float(m.get('rmse', np.nan)) if isinstance(m, dict) else np.nan,
            'mae': float(m.get('mae', np.nan)) if isinstance(m, dict) else np.nan,
            'nse_threshold': float(m.get('nse_threshold', np.nan)) if isinstance(m, dict) else np.nan,
            'nse_pass': bool(m.get('nse_pass')) if isinstance(m, dict) and (m.get('nse_pass') is not None) else np.nan,
            'data_source': str(s.get('data_source')) if isinstance(s, dict) and (s.get('data_source') is not None) else '',
            'feature_count': len(s.get('feature_names', [])) if isinstance(s, dict) and isinstance(s.get('feature_names', []), list) else np.nan,
            'has_causal': int((sd / 'causal_intervention_results.json').exists()),
            'has_probe': int((sd / 'physical_probe_results.json').exists()),
            'has_interactions': int((sd / 'variable_interactions_ranking.csv').exists()),
            'has_faithfulness': int((sd / 'faithfulness_curve.csv').exists()),
            'has_model': int((sd / 'sparse_transformer.pt').exists()),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _summary_from_station_table(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame([{'metric': 'n_stations', 'value': 0}])

    nse = df['nse'].dropna().astype(float)
    rmse = df['rmse'].dropna().astype(float)
    mae = df['mae'].dropna().astype(float)

    n_nse_pass = int((df['nse_pass'] == True).sum())
    n_total = int(len(df))

    rows = [
        {'metric': 'n_stations', 'value': float(n_total)},
        {'metric': 'n_nse_available', 'value': float(len(nse))},
        {'metric': 'nse_mean', 'value': float(nse.mean()) if len(nse) else np.nan},
        {'metric': 'nse_median', 'value': float(nse.median()) if len(nse) else np.nan},
        {'metric': 'nse_min', 'value': float(nse.min()) if len(nse) else np.nan},
        {'metric': 'nse_max', 'value': float(nse.max()) if len(nse) else np.nan},
        {'metric': 'nse_pass_count', 'value': float(n_nse_pass)},
        {'metric': 'nse_pass_rate', 'value': float(n_nse_pass / n_total) if n_total else np.nan},
        {'metric': 'rmse_mean', 'value': float(rmse.mean()) if len(rmse) else np.nan},
        {'metric': 'mae_mean', 'value': float(mae.mean()) if len(mae) else np.nan},
        {'metric': 'coverage_has_causal', 'value': float(df['has_causal'].mean())},
        {'metric': 'coverage_has_probe', 'value': float(df['has_probe'].mean())},
        {'metric': 'coverage_has_interactions', 'value': float(df['has_interactions'].mean())},
        {'metric': 'coverage_has_faithfulness', 'value': float(df['has_faithfulness'].mean())},
        {'metric': 'coverage_has_model', 'value': float(df['has_model'].mean())},
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize Explainable Sparse Transformer station results from outputs/<station_id>/summary.json'
    )
    parser.add_argument('--output-root', type=str, default=None, help='root outputs directory')
    parser.add_argument('--station-out', type=str, default=None, help='per-station CSV output')
    parser.add_argument('--summary-out', type=str, default=None, help='global summary CSV output')
    args = parser.parse_args()

    output_root = args.output_root or _preferred_output_root()
    station_out = args.station_out or os.path.join(output_root, 'sparse_transformer_station_metrics.csv')
    summary_out = args.summary_out or os.path.join(output_root, 'sparse_transformer_summary.csv')

    station_df = _build_station_table(output_root)
    if station_df.empty:
        raise SystemExit(f'No station summary.json found under: {output_root}')

    summary_df = _summary_from_station_table(station_df)

    station_path = Path(station_out)
    summary_path = Path(summary_out)
    station_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    station_df.to_csv(station_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary_df.to_string(index=False))
    print(f'Using output root: {output_root}')
    print(f'Wrote station metrics: {station_path}')
    print(f'Wrote summary metrics: {summary_path}')


if __name__ == '__main__':
    main()
