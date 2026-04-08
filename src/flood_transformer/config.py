from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Central configuration for model, training, sparsity, and explainability settings."""

    # Data settings
    data_root: str = "data"
    forcing_dir: str = "data/Forcing"
    streamflow_dir: str = "data/Streamflow"
    station_id: Optional[str] = None
    seq_len: int = 15
    pred_horizon: int = 1

    # Input-field candidates (matched with tolerant fallback logic)
    rainfall_col_candidates: List[str] = field(default_factory=lambda: ["P", "prcp", "precip", "rain"])
    temp_col_candidates: List[str] = field(default_factory=lambda: ["T", "tmean", "temp"])
    pet_col_candidates: List[str] = field(default_factory=lambda: ["PET", "pet", "evap"])
    q_up_col_candidates: List[str] = field(default_factory=lambda: ["Q_up", "qup", "upstream_q"])
    soil_col_candidates: List[str] = field(default_factory=lambda: ["Soil", "soil", "soil_moisture"])
    streamflow_col_candidates: List[str] = field(default_factory=lambda: ["Q", "streamflow", "flow"])
    include_streamflow_history_input: bool = True

    # Task settings
    task_type: str = "regression"  # regression | classification
    flood_quantile_threshold: float = 0.9
    nse_threshold: float = 0.5

    # Model settings
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1

    # Sparsity settings
    enable_weight_sparsity: bool = True
    enable_activation_sparsity: bool = True
    target_weight_sparsity: float = 0.9
    target_activation_sparsity: float = 0.8
    sparsity_warmup_ratio: float = 0.2
    sparsity_anneal_mode: str = "cosine"  # cosine | linear | power_law
    sparsity_anneal_exponent: int = 2
    weight_topk_mode: str = "neuronwise"  # global | neuronwise
    minimum_alive_per_neuron: int = 2

    # Circuit-mask regularization
    enable_learnable_masks: bool = True
    lambda_mask_l1: float = 1e-4
    lambda_input_mask_l1: float = 1e-4
    circuit_threshold: float = 0.5
    input_threshold: float = 0.5
    # 0 means evaluate all internal sparse-circuit edges with real ablation.
    edge_ablation_topk: int = 0
    in_node_bridge_max_nodes: int = 32

    # Training settings
    seed: int = 42
    epochs_dense: int = 10
    epochs_sparse: int = 10
    epochs_mask: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Output settings
    output_dir: str = "outputs"
    device: str = "cpu"

    # Debugging and logging
    debug: bool = False
    use_rich_progress: bool = True
    log_every: int = 20
