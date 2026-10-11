import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def _nan_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def _bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = pv[order] * (m / ranks)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    idx_valid = np.where(valid)[0]
    q[idx_valid[order]] = q_sorted
    return q


def _paired_effect_size(diff):
    n = diff.size
    if n < 2:
        return float('nan')
    std = np.std(diff, ddof=1)
    if not np.isfinite(std) or std == 0.0:
        return float('nan')
    return float(np.mean(diff) / std)


def _sign_consistency(diff):
    nz = diff[np.abs(diff) > 0]
    if nz.size == 0:
        return float('nan'), float('nan'), 0
    pos = int((nz > 0).sum())
    n = int(nz.size)
    p_val = stats.binomtest(pos, n=n, p=0.5, alternative='two-sided').pvalue
    return pos / n, float(p_val), n


def _bootstrap_ci(diff, n_boot=2000, alpha=0.05, seed=42):
    if diff.size == 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = diff.size
    mean_samples = np.empty(n_boot, dtype=float)
    median_samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = diff[rng.integers(0, n, n)]
        mean_samples[i] = np.mean(sample)
        median_samples[i] = np.median(sample)
    lo = 100 * (alpha / 2.0)
    hi = 100 * (1.0 - alpha / 2.0)
    return (
        float(np.percentile(mean_samples, lo)),
        float(np.percentile(mean_samples, hi)),
        float(np.percentile(median_samples, lo)),
        float(np.percentile(median_samples, hi)),
    )


def _metric_pairs(columns):
    cols = set(columns)
    pairs = []
    for c in sorted(columns):
        if not c.endswith('_base'):
            continue
        stem = c[:-5]
        mod = f'{stem}_mod'
        if mod in cols:
            pairs.append((stem, c, mod))
    return pairs


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
    # Fallback keeps prior behavior if no candidate exists yet.
    return 'outputs'


def _candidate_roots(user_root=None):
    roots = []
    if user_root:
        roots.append(user_root)
    roots.extend([
        os.environ.get('SPARSE_TRANSFORMER_OUTPUT_ROOT', '').strip(),
        '/mnt/d/Store/Sparse_Transformer',
        '/mnt/d/Store/Sparse-Transformer',
        'outputs',
        '.',
    ])
    out = []
    seen = set()
    for r in roots:
        if not r:
            continue
        rr = os.path.normpath(r)
        if rr in seen:
            continue
        seen.add(rr)
        out.append(rr)
    return out


def _resolve_pattern_files(user_pattern=None, user_root=None):
    if user_pattern:
        files = sorted(glob.glob(user_pattern))
        return files, user_pattern

    for root in _candidate_roots(user_root=user_root):
        patterns = [
            os.path.join(root, 'random_forest_model_performance*.csv'),
            os.path.join(root, 'outputs', 'random_forest_model_performance*.csv'),
            os.path.join(root, 'hbv_model_performance*.csv'),
            os.path.join(root, 'outputs', 'hbv_model_performance*.csv'),
        ]
        for pat in patterns:
            files = sorted(glob.glob(pat))
            if files:
                return files, pat
    return [], ''


def _resolve_output_root_from_files(files, fallback_root):
    if not files:
        return fallback_root
    # Usually files are at <root>/*_model_performance*.csv.
    # If they are at <root>/outputs/*_model_performance*.csv, use <root>/outputs.
    first = os.path.normpath(files[0])
    parent = os.path.dirname(first)
    return parent if parent else fallback_root


def summarize(files, out_path=None, station_out_path=None, n_boot=2000, alpha=0.05, seed=42):
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['_source_file'] = os.path.basename(f)
            df_list.append(df)
        except Exception:
            continue

    if len(df_list) == 0:
        raise RuntimeError('No valid result files found')

    all_df = pd.concat(df_list, ignore_index=True)
    if 'station_id' not in all_df.columns:
        raise RuntimeError('Missing station_id in input files')
    all_df['station_id'] = all_df['station_id'].astype(str).str.zfill(8)

    pairs = _metric_pairs(all_df.columns)
    if not pairs:
        raise RuntimeError('No metric pairs found. Need columns like *_base and *_mod')

    agg_cols = [base for _, base, _ in pairs] + [mod for _, _, mod in pairs]
    station_means = all_df.groupby('station_id', as_index=True)[agg_cols].mean(numeric_only=True)

    station_frames = []
    summary_rows = []
    for metric_name, base_col, mod_col in pairs:
        pair = station_means[[base_col, mod_col]].dropna().astype(float)
        n = int(pair.shape[0])
        diff = (pair[mod_col] - pair[base_col]).to_numpy()

        row = {
            'metric': metric_name,
            'n': n,
            'base_min': _nan_float(pair[base_col].min() if n else float('nan')),
            'base_mean': _nan_float(pair[base_col].mean() if n else float('nan')),
            'base_max': _nan_float(pair[base_col].max() if n else float('nan')),
            'mod_min': _nan_float(pair[mod_col].min() if n else float('nan')),
            'mod_mean': _nan_float(pair[mod_col].mean() if n else float('nan')),
            'mod_max': _nan_float(pair[mod_col].max() if n else float('nan')),
            'mean_diff_mod_minus_base': _nan_float(np.mean(diff) if n else float('nan')),
            'median_diff_mod_minus_base': _nan_float(np.median(diff) if n else float('nan')),
            'cohen_d_paired': _paired_effect_size(diff),
        }

        sign_ratio, sign_p, sign_n = _sign_consistency(diff)
        row['positive_diff_ratio'] = sign_ratio
        row['sign_test_p_value'] = sign_p
        row['sign_test_n_nonzero'] = sign_n

        if n >= 2:
            tstat, p_t = stats.ttest_rel(pair[mod_col], pair[base_col], nan_policy='omit')
            row['t_stat'] = _nan_float(tstat)
            row['p_value_ttest'] = _nan_float(p_t)
        else:
            row['t_stat'] = float('nan')
            row['p_value_ttest'] = float('nan')

        if n >= 1:
            nz = np.abs(diff) > 0
            if nz.sum() == 0:
                row['wilcoxon_stat'] = float('nan')
                row['p_value_wilcoxon'] = float('nan')
            else:
                try:
                    w = stats.wilcoxon(diff[nz], zero_method='wilcox', alternative='two-sided')
                    row['wilcoxon_stat'] = _nan_float(w.statistic)
                    row['p_value_wilcoxon'] = _nan_float(w.pvalue)
                except Exception:
                    row['wilcoxon_stat'] = float('nan')
                    row['p_value_wilcoxon'] = float('nan')
        else:
            row['wilcoxon_stat'] = float('nan')
            row['p_value_wilcoxon'] = float('nan')

        if n >= 2:
            ci_mean_lo, ci_mean_hi, ci_med_lo, ci_med_hi = _bootstrap_ci(
                diff=diff,
                n_boot=n_boot,
                alpha=alpha,
                seed=seed,
            )
            row['ci_mean_diff_low'] = ci_mean_lo
            row['ci_mean_diff_high'] = ci_mean_hi
            row['ci_median_diff_low'] = ci_med_lo
            row['ci_median_diff_high'] = ci_med_hi
        else:
            row['ci_mean_diff_low'] = float('nan')
            row['ci_mean_diff_high'] = float('nan')
            row['ci_median_diff_low'] = float('nan')
            row['ci_median_diff_high'] = float('nan')

        summary_rows.append(row)

        station_df = pair.rename(columns={base_col: 'base', mod_col: 'mod'}).copy()
        station_df['metric'] = metric_name
        station_df['diff_mod_minus_base'] = station_df['mod'] - station_df['base']
        station_frames.append(
            station_df.reset_index()[['station_id', 'metric', 'base', 'mod', 'diff_mod_minus_base']]
        )

    summary = pd.DataFrame(summary_rows)
    summary['q_value_ttest_bh'] = _bh_fdr(summary['p_value_ttest'].to_numpy())
    summary['q_value_wilcoxon_bh'] = _bh_fdr(summary['p_value_wilcoxon'].to_numpy())
    summary = summary.sort_values(by='metric', ascending=True).reset_index(drop=True)

    station_level = pd.concat(station_frames, ignore_index=True)
    station_level = station_level.sort_values(by=['metric', 'station_id'], ascending=True)

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(out_path, index=False)

    if station_out_path:
        station_dir = os.path.dirname(station_out_path)
        if station_dir:
            os.makedirs(station_dir, exist_ok=True)
        station_level.to_csv(station_out_path, index=False)

    return summary, station_level

def main():
    parser = argparse.ArgumentParser(description='Summarize NSE across baseline/modified benchmark result files')
    parser.add_argument('--output-root', type=str, default=None,
                        help='root folder for outputs (auto-detects /mnt path if omitted)')
    parser.add_argument('--pattern', type=str, default=None,
                        help='glob pattern for result CSVs')
    parser.add_argument('--out', type=str, default=None,
                        help='output summary CSV path')
    parser.add_argument('--station-out', type=str, default=None,
                        help='optional per-station diff CSV path')
    parser.add_argument('--n-boot', type=int, default=2000,
                        help='bootstrap samples for confidence intervals')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='alpha for confidence intervals')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for bootstrap')
    args = parser.parse_args()

    preferred_root = args.output_root or _preferred_output_root()
    files, resolved_pattern = _resolve_pattern_files(user_pattern=args.pattern, user_root=preferred_root)
    if not files:
        raise SystemExit(
            'No files match any candidate pattern. '
            'Tried user pattern or auto patterns like <root>/random_forest_model_performance*.csv '
            'and <root>/outputs/random_forest_model_performance*.csv '
            '(also supports hbv_model_performance*.csv for backward compatibility)'
        )

    output_root = _resolve_output_root_from_files(files, fallback_root=preferred_root)
    out_path = args.out or os.path.join(output_root, 'benchmark_nse_summary.csv')
    station_out = args.station_out or os.path.join(output_root, 'benchmark_metric_station_diffs.csv')

    summary, station_level = summarize(
        files,
        out_path=out_path,
        station_out_path=station_out,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary.to_string(index=False))
    print(f'Using file pattern: {resolved_pattern}')
    print(f'Using output root: {output_root}')
    print(f'Wrote summary to: {out_path}')
    if station_out:
        print(f'Wrote station-level diffs to: {station_out} (rows={len(station_level)})')

if __name__ == '__main__':
    main()