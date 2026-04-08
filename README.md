# Explainable Sparse Transformer for Flood Prediction

This project implements an explainable sparse Transformer for streamflow/flood prediction. In addition to forecasting, it is designed to analyze whether the model learns physically meaningful hydrologic patterns.

## Features

- Time-series Transformer backbone (multi-head attention + residual + MLP)
- Top-K weight sparsity with scheduled annealing (`update_sparsity`)
- Top-K activation sparsity on MLP outputs
- Learnable circuit masks (input features + attention heads + MLP neurons)
- Circuit extraction and ablation tools:
  - `prune_circuit`
  - `mean_ablation`
  - `head_ablation_test`
- Explainability interfaces:
  - `get_attention_statistics`
  - `extract_hidden_states`
  - `linear_probe`
  - `causal_intervention_test`

## Key Files

- `src/flood_transformer/config.py`: experiment configuration
- `src/flood_transformer/sparsity.py`: sparsity utilities and STE logic
- `edge_importance_ranking.csv`: real edge-ablation interaction ranking
- `variable_interactions.json`: pairwise input interaction matrix (real interventions)
- `variable_interactions_ranking.csv`: ranked variable-pair interactions
- `node_importance_ranking.csv`: unified node ranking (head + neuron)
- `faithfulness_curve.csv`: performance curve while keeping Top-K nodes
- `cherry_samples.json`: representative high-flow and low-flow samples

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py \
  --station_id 01013500 \
  --forcing_dir data/Forcing \
  --streamflow_dir data/Streamflow \
  --task regression \
  --device cpu
```

### Optional Advanced Args

```bash
python run_experiment.py \
  --station_id 01013500 \
  --epochs_dense 5 \
  --epochs_sparse 5 \
  --epochs_mask 5 \
  --debug \
  --log_every 20 \
  --sparsity_anneal_mode power_law \
  --sparsity_anneal_exponent 2 \
  --weight_topk_mode neuronwise \
  --minimum_alive_per_neuron 2 \
  --circuit_threshold 0.5 \
  --edge_ablation_topk 30
```

## Outputs

By default, outputs are written to `outputs/`:

- `head_importance_ranking.csv`: attention-head importance
- `neuron_importance_ranking.csv`: MLP-neuron importance
- `circuit_graph.png`: minimal circuit graph
- `attention_statistics.png`: attention distribution before/after flood events
- `physical_probe_results.json`: correlations with physical proxy variables
- `causal_intervention_results.json`: causal intervention results
- `summary.json`: output artifact index and metrics summary

## Interactive Visualization

Run:

```bash
streamlit run streamlit_circuit_viz.py
```

Capabilities:

- Interactive circuit graph (input abbreviations -> heads -> neurons -> discharge output)
- Head/Neuron importance rankings
- Attention distribution visualization
- Causal intervention and probe result views
- Variable interaction matrix and Top variable-pair ranking
- Figure-4-style pathway view (Top-K nodes, inactive gray-out, layer separators, faithfulness curve)

## Debugging and Metrics

- Use `--debug` for step-level logs (loss, mask loss, sparsity, validation loss)
- Rich progress bars are enabled by default; disable with `--no_rich`
- Regression mode computes NSE (Nash-Sutcliffe Efficiency)
- Set pass threshold with `--nse_threshold` (default: `0.5`)

## CAMELS Map Analysis

Run:

```bash
streamlit run streamlit_us_map.py
```

This page can:

- Build station-level feature tables from `outputs/*`
- Cluster stations by `top_factor`, `top_struct_factor`, and `probe`
- Visualize clusters/metrics on a U.S. map with interactive filtering
- Perform association tests against geography/climate metadata (Kruskal/Chi-square)

### CAMELS Static Attributes

- Set `CAMELS attribute root` in the sidebar to an external CAMELS attribute directory
- The app recursively scans CSV/TXT files containing `camels` or `attr` in their names
- It auto-detects station ID keys (for example `gauge_id`/`gage_id`) and merges on `gauge_id`
- If the path is unavailable, the page falls back to in-repo station and explanation features only

## Input-Mask Sparsification Tips

Input features can also be sparsified via learnable masks:

- `--lambda_input_mask_l1`: L1 regularization strength for input-node masks
- `--input_threshold`: threshold for keeping input nodes in the extracted circuit

Suggested tuning path:

1. Keep `--input_threshold 0.5`, increase `--lambda_input_mask_l1` from `1e-5 -> 1e-4 -> 5e-4`.
2. Track `node_type=input` counts in `outputs/node_importance_ranking.csv` and NSE in `summary.json`.
3. If NSE drops noticeably (for example > 0.03), reduce `--lambda_input_mask_l1` or lower `--input_threshold` to `0.4`.
4. If too many inputs remain, keep L1 fixed and raise `--input_threshold` to `0.55~0.7` for post-hoc pruning.

