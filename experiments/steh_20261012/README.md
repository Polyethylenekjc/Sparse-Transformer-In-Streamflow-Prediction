# STEH population experiment, 2026-10-12

This directory preserves method development, validation-only shared tuning and
the frozen population experiment plan. Population training and the final
scientific audit were still running when these records were first published.
Do not interpret this plan as completed or scientifically successful results.

## Frozen design

- All 521 basins: one fresh seed42 STEH model with discharge history and one
  without it. Compare the with-Q primary result to the existing single-seed LSTM
  on identical target dates. The no-Q setting is an input ablation.
- Twelve prespecified new basins sampled without replacement from the 503 basins
  outside the original development and six tuning basins: additionally fit seeds
  43 and 44 in both input settings. The total is 1,090 fresh training tasks.
- 15-day history, next-day target, chronological 70/15/15 split, train-only
  standardization, 10+10+10 training epochs and batch128 remain fixed.
- Shared Optuna TPE search: 12 candidates on the same six random LSTM tuning
  basins, equal weight for both STEH input settings, validation NSE only. The
  selected trial3 mean validation NSE is 0.5058959846. This is the best evaluated
  candidate, not proof of a global optimum.
- Freeze the validation-selected circuit before any test score is read for each
  fresh model. Joint backward reduction uses 512 validation days, tolerance .05,
  and at most 12 rounds. Preserve all failed test outcomes.
- Fixed circuit criterion: test NSE loss <=.05; removing the selected circuit
  reduces NSE by >=.02; strict win fraction >=.8 against structural-control
  draws excluding the core itself. Draws may repeat each other. Twenty controls match layer, type and count exactly, degree
  approximately; disclose overlaps and identical controls.
- Compare seeds using validation input-effect ranks and centered linear CKA,
  not arbitrary neuron identifiers. Residual, embedding, input and readout
  pathways remain present; this is not an isolated closed causal graph.

## Preserved development records

`development-records.tar.gz` contains all numerical records, validation
predictions, logs, Optuna state, development circuit interventions and functional
stability from `outputs/steh_tuning_20261012` and
`outputs/steh_refined_20261012`, plus the frozen population manifest/config.
It excludes model checkpoint files. SHA256:
`e8039275cd6e579b10649b6ce8a3164583dfcadefb1e0743ee435f1ac53da764`.

Original fixed-fraction circuits failed all72 development fits. Joint validation
reduction passed9/72 (7 with-Q, 2 no-Q). These original12 basins had already been
evaluated and remain **method development**, not independent confirmation.

`frozen-source.tar.gz` preserves the exact server source bytes. A few unchanged
files differ from the local checkout only by newline encoding; the archived
bytes match the hashes in `population-manifest.json`. Legacy plotting modules
are included even though the new population runner does not call them.

Original 1,042 forcing/streamflow CSV inputs remain in the repository's
[`data`](../../data) directory. LSTM final results are in
[`lstm_global_20261012`](../lstm_global_20261012). Its 665 checkpoint files are
backed up locally with all compressed/member SHA256 values verified; weights
are not included here while separate public-upload authorization is pending.

## Reproduction and reporting

Run `experiments/run_steh_population.py --workers 14` on the documented server
environment. It launches circuit evaluation after successful training.
Run `experiments/report_steh_population.py` on completed training/circuit records
and the LSTM raw records. The audit recomputes daily-prediction metrics, checks
dates, frozen circuit identity, control outcomes and phase-checkpoint selection.
`experiments/build_steh_workbook.mjs` creates the associated workbook.

Population results describe all521 basins in this dataset. Between-basin SD is
not seed uncertainty. Confirmation inference treats basin as the unit, requires
all three fixed seeds to pass, and uses an exact finite-population SRS interval
for the 503 eligible basins. No inference covers all possible initializations.

Near-threshold gates and finite-epoch churn are reported separately; a completed
run or decreasing loss is not proof of structural convergence. Do not revise
test thresholds or hide extreme NSE values to improve manuscript figures.

## Fixed control sensitivity protocol

The early review of the first 12 ordered basins identified a limitation of
nearest-degree controls: the same retained units can be selected repeatedly.
The original primary criterion remains unchanged. An additional sensitivity
analysis reports **both** fixed robust-degree calipers 1 and 2, with 20 draws
per caliper, exact layer/type/count matching, and the same validation-frozen
core. Neither caliper is selected using test performance. Control designs are
saved before forecasts, and every individual control forecast is retained.
Overlaps, repeated masks, nonidentical-draw wins and unique-mask wins are
reported separately. GPU-vectorized predictions are checked against sequential
predictions; numerically ambiguous comparisons are recomputed sequentially.

For this added method, the first 12 early-review basins are development data.
Conditional on avoiding those basins, the same prespecified random confirmation
sample is an SRS from 492 eligible basins. Success still requires all three
fixed seeds to pass. Four exact hypergeometric 98.75% marginal intervals
(two input settings by two calipers) provide at least 95% joint coverage.
This sensitivity family is reported alongside the original primary analysis.

The complete added protocol is `control-sensitivity-protocol-v2.json`.
Run `experiments/steh_control_sensitivity.py` with the explicit output
`outputs/steh_control_sensitivity_20261012_v2` after primary circuit completion,
then `experiments/audit_steh_controls.py`. Computational integrity does not
imply that either scientific criterion will pass.

The development refinement used archived `development_refinement_v1.py`,
SHA256 2de2a9586762ef8663c2068c21b0ff646f1d842a460ca18d33b762109edc6a0a,
which matches its development protocol. It is a source snapshot; use the main
experiment script for the frozen population analysis. Final export preserves
original development predictions/failures, control sensitivity records,
performance telemetry and checkpoint backups, with per-member SHA256 manifests.

## Numerical forward diagnostic

`ste-roundoff-diagnostic.json` documents a read-only check at development basin
01052500, with-Q. Native FP32 evaluation of `hard + probs - probs.detach()`
produces gate roundoff up to5.96046e-8; exact binary overrides can consequently
produce a small forecast difference. The maximum train-scaled prediction
change was7.72476e-5 and NSE difference1.70734e-7. Evaluating the mathematical
equivalent `hard + (probs - probs.detach())` **in memory only** produced
bitwise equality with binary overrides. Frozen training source was not edited.
The report audits native and binary metrics separately and discloses pointwise
differences instead of silently widening tolerances. Primary forecasting uses
native gates; circuit interventions and their full reference use binary gates.

`partial-report-audit.json` and `partial-report-records.csv` describe only the
first24 completed development-review models. They are not population results
or independent confirmation. `architecture-counts.json` distinguishes all
allocated parameters, matrix parameters and parameters used by15day inputs.
Positional embeddings allocate2048positions; only15are used here. Matrix
sparsity does not itself establish inference speed with dense kernels.
The requested local LSTM comparison does not establish superiority over pooled
regional or ensemble LSTMs; see https://hess.copernicus.org/articles/28/4187/2024/.
KGE=0 is not an inherent skill threshold; benchmarks are explicit, as discussed
at https://hess.copernicus.org/articles/23/4323/2019/.
