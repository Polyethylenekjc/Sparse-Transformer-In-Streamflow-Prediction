# STEH scientific quality checks

The objective is credible predictive and mechanistic evidence. Different basins
may legitimately learn different circuits. Same-basin sensitivity to training
initialization is evaluated separately. Raw hidden-unit identifiers cannot be
matched across independently initialized models.

1. Preserve the original 12-basin/3-seed results and every circuit failure.
2. Develop validation-guided joint backward reduction on these frozen models.
   Every accepted removal must retain validation NSE within .05 of the complete
   gated model. The tolerance, 512 validation dates, 12-round limit and random
   control count are frozen before test evaluation. This is a bounded search,
   not a globally minimal or isolated closed graph.
3. Freeze each selected core before computing test predictions. Evaluate the
   complete model, retained core, removed core and 20 structural controls on the
   same complete test dates. Report exact layer/type/count matching, approximate
   degree matching, overlap, and the number of distinct controls. An identical
   control provides no independent comparison.
4. Keep the original scientific criterion: core NSE loss <=.05, core removal
   loss >=.02, and strict core win fraction >=.8 against distinct controls.
   Computational completion does not imply that this criterion passes.
5. Compare seeds using validation input-effect ranks and linear CKA of centered
   hidden representations on identical dates. CKA is invariant to hidden-unit
   permutations and global scaling. Shared residual/readout pathways remain a
   limitation; high CKA alone does not establish causal circuit equivalence.
6. Tune STEH globally on the same six random stations used for the LSTM search,
   using validation NSE only. Both with-Q and no-Q settings are equally weighted.
   Preserve the 10+10+10 epoch phase budget, chronological split, 15-day window,
   batch128 and train-only standardization. Save all configurations, validation
   predictions, phase curves and selected checkpoint hashes.
7. Separate original-basin method development from confirmation on new,
   prespecified stations. Do not repeatedly revise methods based on held-out
   test scores. Main all-station performance and sampled cross-seed stability
   serve different inferential roles.
8. Report basin-level paired performance, medians, complete distributions,
   negative cases and confidence intervals where the sampling design permits.
   Average seeds within a basin before population-level inference. A single-seed
   LSTM baseline does not estimate initialization variance.

Observed development result: the original 25%-candidate core failed all72 fits.
Joint validation reduction completed all72 in 115.83 seconds and passed9
(7 with-Q, 2 no-Q). These observations do not yet support a universal compact
faithful-circuit claim. Neither failures nor thresholds are changed for plotting.

Scripts: `experiments/refine_steh.py`, `experiments/tune_steh.py`.
The final scope and new confirmation sample are frozen separately before runs.
