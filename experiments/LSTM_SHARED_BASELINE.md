# Shared-parameter LSTM baseline

Each station receives one freshly initialized LSTM. All stations use one shared
configuration, seed 42, a 15-day input window and a one-day forecast horizon.
There is no seed ensemble or multi-window experiment in this baseline.

The chronological split, input construction, train-only standardization,
batch size 128, 30 training epochs, AdamW, gradient clipping and FP32 policy
match the existing revision experiment. Meteorological inputs and one Q-history
channel are supplied to both the with-Q STEH and this LSTM. Each station keeps
its own fitted scaler and weights; model weights are not shared between stations.

Bayesian tuning uses Optuna's seeded TPE sampler. Six stations are selected
uniformly without replacement from the 521 paired CSV stations, using sampling
seed 20261012. Every candidate is evaluated on the same six stations. The
objective is their unweighted mean validation NSE. The first four candidates
are the original LSTM grid; up to 24 configurations are evaluated, with a
30-minute search budget checked between candidate batches. The winner is the
best observed validation configuration, not a guaranteed global optimum.
The test period is never used for tuning or model selection. After freezing
the winner, all 521 stations are trained again from fresh initialization.

The search space covers hidden size 32/64/128/256, one or two recurrent layers,
learning rate 1e-4 to 3e-3 on a logarithmic scale, inter-layer dropout 0/0.1/0.3
(zero for one layer), and weight decay 0/1e-5/1e-4. Candidate checkpoints use the
lowest validation MSE within their fixed 30 epochs. Two candidate configurations
are proposed at a time; results are returned to TPE in candidate-index order.

```bash
conda activate steh
python experiments/global_lstm.py --scope all --workers 14
python experiments/report_global_lstm.py \
  --input experiments/lstm_global_20261012/raw \
  --output experiments/lstm_global_20261012/report \
  --reference experiments/lstm_global_20261012/reference_manifest.json
```

`--prepare-only` freezes the protocol without training. Completed fits can be
resumed; changing the frozen source/configuration/data requires a new output
directory. The initial 12-station, three-seed LSTM trial grid and interrupted
station-specific tuning are preliminary records and are superseded by this
shared-parameter baseline. They must not be combined with the final 521 models.

Repository `data/Forcing` and `data/Streamflow` contain the original supplied
CSV inputs. Experiment raw records contain every tuning station's validation
predictions and training curves, the full trial ranking, shared selection,
per-station metadata, final daily predictions and same-date persistence/AR
references. Model weights are losslessly archived as GitHub release assets to
keep binary checkpoints out of normal Git history. SHA256 manifests accompany
the archives. No SSH password or GitHub credential is included.

Scientific graphics follow the user-specified
[academic-research-skills visualization agent](https://github.com/Imbad0202/academic-research-skills/blob/main/academic-paper/agents/visualization_agent.md),
commit d1e51716e53fc371727ea6164dd709fc8ea28c97. Figure/table provenance includes
source hashes, transformation hash, caption claim, manuscript claim links and
known limitations. No manuscript claims are inserted automatically.
