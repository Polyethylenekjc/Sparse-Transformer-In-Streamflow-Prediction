from __future__ import annotations

import argparse
import json
import os
from typing import List, Set

from src.flood_transformer.config import ExperimentConfig
from src.flood_transformer.train import run_full_experiment


def _list_station_ids(forcing_dir: str, streamflow_dir: str, require_streamflow: bool = True) -> List[str]:
    forcing_ids: Set[str] = {
        os.path.splitext(f)[0]
        for f in os.listdir(forcing_dir)
        if f.endswith(".csv")
    }
    if not require_streamflow:
        return sorted(forcing_ids)

    if not os.path.exists(streamflow_dir):
        return sorted(forcing_ids)

    stream_ids: Set[str] = {
        os.path.splitext(f)[0]
        for f in os.listdir(streamflow_dir)
        if f.endswith(".csv")
    }
    return sorted(forcing_ids.intersection(stream_ids))


def build_args():
    p = argparse.ArgumentParser(description="Run flood sparse-transformer experiment for all stations")
    p.add_argument("--forcing_dir", type=str, default="data/Forcing")
    p.add_argument("--streamflow_dir", type=str, default="data/Streamflow")
    p.add_argument("--output_root", type=str, default="outputs", help="每个站点输出到 output_root/站点ID")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--seq_len", type=int, default=15)
    p.add_argument("--pred_horizon", type=int, default=1)
    p.add_argument("--epochs_dense", type=int, default=10)
    p.add_argument("--epochs_sparse", type=int, default=10)
    p.add_argument("--epochs_mask", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nse_threshold", type=float, default=0.5)
    p.add_argument("--target_weight_sparsity", type=float, default=0.9)
    p.add_argument("--target_activation_sparsity", type=float, default=0.8)
    p.add_argument("--lambda_mask_l1", type=float, default=1e-4)
    p.add_argument("--circuit_threshold", type=float, default=0.5)
    p.add_argument("--edge_ablation_topk", type=int, default=30)
    p.add_argument("--sparsity_anneal_mode", type=str, default="cosine", choices=["cosine", "linear", "power_law"])
    p.add_argument("--sparsity_anneal_exponent", type=int, default=2)
    p.add_argument("--weight_topk_mode", type=str, default="neuronwise", choices=["global", "neuronwise"])
    p.add_argument("--minimum_alive_per_neuron", type=int, default=2)
    p.add_argument("--include_q_input", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no_rich", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--require_streamflow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_stations", type=int, default=0, help=">0 时只跑前 N 个站点")
    return p.parse_args()


def main():
    args = build_args()

    if not os.path.exists(args.forcing_dir):
        raise FileNotFoundError(f"forcing_dir 不存在: {args.forcing_dir}")

    station_ids = _list_station_ids(
        forcing_dir=args.forcing_dir,
        streamflow_dir=args.streamflow_dir,
        require_streamflow=args.require_streamflow,
    )
    if args.max_stations > 0:
        station_ids = station_ids[: args.max_stations]

    if len(station_ids) == 0:
        raise RuntimeError("未找到可运行的站点")

    os.makedirs(args.output_root, exist_ok=True)

    results = []
    for idx, station_id in enumerate(station_ids, start=1):
        out_dir = os.path.join(args.output_root, station_id)
        print(f"[{idx}/{len(station_ids)}] Running station {station_id} -> {out_dir}")

        config = ExperimentConfig(
            station_id=station_id,
            forcing_dir=args.forcing_dir,
            streamflow_dir=args.streamflow_dir,
            include_streamflow_history_input=args.include_q_input,
            task_type=args.task,
            nse_threshold=args.nse_threshold,
            device=args.device,
            seq_len=args.seq_len,
            pred_horizon=args.pred_horizon,
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
            output_dir=out_dir,
        )

        try:
            output = run_full_experiment(config)
            status = "ok"
            err = ""
        except Exception as e:
            output = {}
            status = "failed"
            err = str(e)

        results.append(
            {
                "station_id": station_id,
                "status": status,
                "output_dir": out_dir,
                "error": err,
                "artifacts": output,
            }
        )

    run_summary_path = os.path.join(args.output_root, "multi_station_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok
    print(json.dumps({"total": len(results), "ok": ok, "failed": failed, "summary": run_summary_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
