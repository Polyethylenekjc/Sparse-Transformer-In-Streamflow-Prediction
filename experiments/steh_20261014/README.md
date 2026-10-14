# STEH population experiment — final records, 2026-10-14

521 catchments have separate with-Q and no-Q seed42 models. Twelve prespecified new catchments add seeds43/44, giving 1,090 trained models. All use a common validation-selected hyperparameter configuration, catchment-specific weights and training-only normalization. The LSTM is the already published 521-model, shared-TPE, single-seed local baseline under `../lstm_global_20261012`.

The original nearest-degree circuit criterion remains primary: 203/521 with-Q and 167/521 no-Q models pass. Three fixed seeds jointly pass in only 2/12 confirmation catchments per setting. The two fixed structural-control calipers are sensitivity analyses: 291/310 with-Q and 246/264 no-Q primary models pass. Both are reported; neither is selected by test performance. Functional similarity does not establish fully converged, unique or causal physical circuits.

## Contents

- `report/`: independently audited all-basin tables, eight figures in PDF/SVG/600dpi PNG, workbook, source/claim traces, numerical/reference sensitivity and Chinese manuscript-readiness assessment.
- `raw/numerical-records.part-001` through `part-092`: 955,052,650-byte gzip tar, split at 10 MiB. All 30,994 numerical files are preserved, including tuning, development, previous numerical failures, predictions, probe scores and the final 43,600 control forecasts.
- `numerical-parts.json`, `numerical-manifest.json`: ordered compressed-part hashes and every extracted member hash.
- `performance-records.tar.gz`, `performance-manifest.json`: measured performance records and failure/recovery logs.
- `source-snapshot.tar.gz`, `source-snapshot-manifest.json`: exact frozen training source bytes plus final numerical-recovery/audit/report tools. Fourteen training-source hashes are verified against the study manifest. Use this snapshot when platform newline conversion would otherwise alter file hashes.
- `checkpoint-manifest.json`, `local-backup-audit.json`: 1,342 checkpoints were backed up and verified locally, including all 1,090 final models. Weight bytes are not public while separate permission remains pending.

## Restore numerical records

Run from this directory with Python 3.12+ (safe tar data filter):

```python
import hashlib, json, pathlib, tarfile
root = pathlib.Path('.')
manifest = json.loads((root / 'numerical-parts.json').read_text())
archive = root / manifest['archive']['archive']
with archive.open('wb') as target:
    for part in manifest['parts']:
        block = (root / 'raw' / part['path']).read_bytes()
        assert len(block) == part['bytes']
        assert hashlib.sha256(block).hexdigest() == part['sha256']
        target.write(block)
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['archive']['sha256']
with tarfile.open(archive) as tar:
    tar.extractall(root / 'restored', filter='data')
```

Restored records retain `outputs/` roots. Final control results are in `outputs/steh_control_sensitivity_20261014_sequential`. The earlier vectorized run in `outputs/steh_control_sensitivity_20261012_v2` failed numerical verification for 74 jobs and is historical evidence, not the final sensitivity analysis. All final controls were recomputed sequentially per mask with parallel catchments; the protocol, checkpoints, validation-selected cores and scientific thresholds were not changed.

The three original cross-forward metric failures were recovered separately against their own native/binary forecast references. Original failed records remain in the archive. Across all 1,090 models, switching the full-model NSE reference from binary to native changes zero original joint-criterion classifications; the pointwise differences remain disclosed.

## Interpretation and manuscript limits

All catchments, negative metrics, extreme values, undefined wins and scientific failures are retained. The LSTM extreme at 08190500 is not clipped or specially retrained. Census descriptions are not seed uncertainty. Confirmation intervals target eligible catchments and three fixed seeds; the four caliper intervals have their own Bonferroni family and conditional N492 target. CUM/API/CHG are input-derived proxies; their probe scores support decodability rather than independent physical states. Saved probe R² records were count/status/source checked but cannot be independently recomputed without unsaved probe forecasts/representations.

The current Word manuscript has not been rewritten to these results. Old RF improvements, paired interactions and scenario perturbations were not revalidated by this run. The assessment lists the required replacements and limits; it is not a formal peer-review panel or a claim of publication readiness.
