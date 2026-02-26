from __future__ import annotations

import argparse
import json
import os

from src.flood_transformer.config import ExperimentConfig
from src.flood_transformer.train import run_full_experiment


def _resolve_station_id(station_id: str | None, forcing_dir: str, streamflow_dir: str) -> str:
    if station_id:
        return station_id

    forcing_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(forcing_dir)
        if f.endswith(".csv")
    }
    stream_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(streamflow_dir)
        if f.endswith(".csv")
    } if os.path.exists(streamflow_dir) else set()

    common = sorted(forcing_ids.intersection(stream_ids))
    if common:
        return common[0]
    if forcing_ids:
        return sorted(forcing_ids)[0]
    raise FileNotFoundError("未找到可用站点，请检查 forcing/streamflow 目录")


def build_args():
    p = argparse.ArgumentParser(description="Explainable Sparse Transformer for Flood Prediction")
    p.add_argument("--station_id", type=str, default=None, help="站点 ID，例如 01013500")
    p.add_argument("--forcing_dir", type=str, default="data/Forcing", help="气象强迫数据目录")
    p.add_argument("--streamflow_dir", type=str, default="data/Streamflow", help="流量数据目录")
    p.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--nse_threshold", type=float, default=0.5, help="NSE 达标阈值")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seq_len", type=int, default=15)
    p.add_argument("--epochs_dense", type=int, default=10)
    p.add_argument("--epochs_sparse", type=int, default=10)
    p.add_argument("--epochs_mask", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target_weight_sparsity", type=float, default=0.9)
    p.add_argument("--target_activation_sparsity", type=float, default=0.8)
    p.add_argument("--lambda_mask_l1", type=float, default=1e-4)
    p.add_argument("--circuit_threshold", type=float, default=0.5)
    p.add_argument("--edge_ablation_topk", type=int, default=30)
    p.add_argument("--sparsity_anneal_mode", type=str, default="cosine", choices=["cosine", "linear", "power_law"])
    p.add_argument("--sparsity_anneal_exponent", type=int, default=2)
    p.add_argument("--weight_topk_mode", type=str, default="neuronwise", choices=["global", "neuronwise"])
    p.add_argument("--minimum_alive_per_neuron", type=int, default=2)
    p.add_argument("--debug", action="store_true", help="输出调试日志")
    p.add_argument("--no_rich", action="store_true", help="禁用 rich 进度条")
    p.add_argument("--log_every", type=int, default=20, help="每多少 step 打印一次调试信息")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument(
        "--include_q_input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否将历史流量 Q 作为输入特征（默认开启）",
    )
    return p.parse_args()


def main():
    args = build_args()
    station_id = _resolve_station_id(args.station_id, args.forcing_dir, args.streamflow_dir)
    output_dir = args.output_dir
    if os.path.basename(os.path.normpath(output_dir)) != station_id:
        output_dir = os.path.join(output_dir, station_id)

    config = ExperimentConfig(
        station_id=station_id,
        forcing_dir=args.forcing_dir,
        streamflow_dir=args.streamflow_dir,
        include_streamflow_history_input=args.include_q_input,
        task_type=args.task,
        nse_threshold=args.nse_threshold,
        device=args.device,
        seq_len=args.seq_len,
        epochs_dense=args.epochs_dense,
        epochs_sparse=args.epochs_sparse,
        epochs_mask=args.epochs_mask,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target_weight_sparsity=args.target_weight_sparsity,
        target_activation_sparsity=args.target_activation_sparsity,
        lambda_mask_l1=args.lambda_mask_l1,
        circuit_threshold=args.circuit_threshold,
        edge_ablation_topk=args.edge_ablation_topk,
        sparsity_anneal_mode=args.sparsity_anneal_mode,
        sparsity_anneal_exponent=args.sparsity_anneal_exponent,
        weight_topk_mode=args.weight_topk_mode,
        minimum_alive_per_neuron=args.minimum_alive_per_neuron,
        debug=args.debug,
        use_rich_progress=(not args.no_rich),
        log_every=args.log_every,
        output_dir=output_dir,
    )

    outputs = run_full_experiment(config)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
