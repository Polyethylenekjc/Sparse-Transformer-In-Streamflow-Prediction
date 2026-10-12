# 521-station shared-parameter LSTM baseline

Computational and independent metric audits: **PASS**. Each of the 521 stations
has one fresh seed-42 model, shared hyperparameters and a 15-day history. There
is no multi-seed or multi-window LSTM experiment in these final results.

24 shared configurations were evaluated on the same six random tuning stations,
with validation-only selection. Winner trial17: hidden256, layers2,
LR0.002713975339231919, weight decay0.0001, dropout0. The mean validation NSE is
0.615620109. Tuning plus all521 final fits took 2149.71 seconds (35m50s).

Test NSE median **0.643255**, mean **0.190351**; 13 stations have negative NSE.
Station08190500 has NSE -210.885 and is retained. Its observed test mean is
0.512cfs and SD3.618cfs, against training mean34.667cfs and SD570.674cfs;
this explains sensitivity to biased predictions in a low-variance regime.
It does not justify deleting the station or tuning against its test results.

- [Chinese report](report/实验报告.md) and [audited summary](report/audit.json).
- [Workbook](report/LSTM共享参数基线.xlsx): summary, all521 station models,
  and all144 candidate/station fits.
- Figures include complete test distributions, Bayesian search and final
  learning curves; vectorPDF/SVG and 600dpiPNG are provided.
- [Raw numeric records](raw-records.tar.gz): all validation predictions, phase
  logs, selected parameters, SQLite/TPE state, per-station final predictions,
  same-date reference predictions and metadata. SHA256 is in the adjacent manifest.
- [Original input CSVs](../../data): 521 Forcing and521 Streamflow files remain
  tracked in the repository; their hashes are frozen in tuning_protocol.json.
- [Performance records](verification-records.tar.gz) and performance-summary.json
  document benchmarks and telemetry. GPU median93%, CPU quota median99.80%,
  GPU memory peak7439MiB. The microbenchmark identified14 workers as the best
  tested concurrency; it is not an end-to-end speedup estimate.
- All665 checkpoints have complete local/server backups. Their per-archive and
  per-member SHA256 indices are in checkpoint-backup-manifest.json. Public weight
  upload awaits separate permission; the numerical record archive excludes weights.

To reproduce the analysis:

```bash
mkdir -p experiments/lstm_global_20261012/raw
tar -xzf experiments/lstm_global_20261012/raw-records.tar.gz -C experiments/lstm_global_20261012/raw
python experiments/report_global_lstm.py --input experiments/lstm_global_20261012/raw --output experiments/lstm_global_20261012/report --reference experiments/lstm_global_20261012/reference_manifest.json
```

All negative/undefined results remain visible. Spread is across stations, not
training-seed uncertainty. Supplied data lack daily original-observation flags;
the evaluation cannot be described as original-observation-only. No claim of
universal superiority, significance or guaranteed globally optimal parameters
is made. This baseline supersedes the preliminary per-station/multi-seed grids.
