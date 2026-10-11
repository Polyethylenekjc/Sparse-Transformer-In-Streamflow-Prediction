from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import traceback
from typing import Dict, List

import pandas as pd

# Allow running this file directly via `python scripts/xxx.py`.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from recompute_real_edge_importance import recompute_real_edge_importance_for_station


def _discover_stations(output_root: str) -> List[str]:
    if not os.path.isdir(output_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(output_root)):
        p = os.path.join(output_root, name)
        if not os.path.isdir(p):
            continue
        # minimal required artifacts
        if os.path.exists(os.path.join(p, "sparse_transformer.pt")) and os.path.exists(os.path.join(p, "circuit_structure.json")):
            out.append(name)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch recompute REAL edge importance for all stations.")
    p.add_argument("--output_root", default="/mnt/d/Store/Sparse_Transformer")
    p.add_argument("--forcing_dir", default="data/Forcing")
    p.add_argument("--streamflow_dir", default="data/Streamflow")
    p.add_argument("--device", default="cpu")

    p.add_argument("--seq_len", type=int, default=15)
    p.add_argument("--pred_horizon", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--include_q_input", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--circuit_threshold", type=float, default=0.5)
    p.add_argument("--max_edges", type=int, default=0, help="0 means all internal edges per station")
    p.add_argument("--out_csv", default="edge_importance_ranking_real.csv")
    p.add_argument("--compute_in_node_bridge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--in_node_bridge_max_nodes", type=int, default=32)
    p.add_argument("--in_node_out_json", default="in_node_interactions.json")
    p.add_argument("--in_node_out_csv", default="in_node_interactions_ranking.csv")
    p.add_argument("--bundle_out_json", default="real_importance_bundle.json")

    p.add_argument("--stations", nargs="*", default=None, help="Optional specific station ids")
    p.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", type=int, default=4, help="Parallel workers for multi-process recompute")
    return p.parse_args()


def _run_one(task: Dict[str, object]) -> Dict[str, str]:
    sid = str(task["station_id"])
    try:
        out_path = recompute_real_edge_importance_for_station(
            station_id=sid,
            output_root=str(task["output_root"]),
            forcing_dir=str(task["forcing_dir"]),
            streamflow_dir=str(task["streamflow_dir"]),
            device_name=str(task["device"]),
            seq_len=int(task["seq_len"]),
            pred_horizon=int(task["pred_horizon"]),
            batch_size=int(task["batch_size"]),
            include_q_input=bool(task["include_q_input"]),
            circuit_threshold=float(task["circuit_threshold"]),
            max_edges=int(task["max_edges"]),
            out_csv=str(task["out_csv"]),
            compute_in_node_bridge=bool(task["compute_in_node_bridge"]),
            in_node_bridge_max_nodes=int(task["in_node_bridge_max_nodes"]),
            in_node_out_json=str(task["in_node_out_json"]),
            in_node_out_csv=str(task["in_node_out_csv"]),
            bundle_out_json=str(task["bundle_out_json"]),
        )
        return {
            "station_id": sid,
            "status": "ok",
            "output_csv": out_path,
            "in_node_csv": os.path.join(str(task["output_root"]), sid, str(task["in_node_out_csv"])),
            "bundle_json": os.path.join(str(task["output_root"]), sid, str(task["bundle_out_json"])),
            "error": "",
        }
    except Exception as e:
        return {"station_id": sid, "status": "failed", "output_csv": "", "in_node_csv": "", "bundle_json": "", "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    args = parse_args()

    stations = args.stations if args.stations else _discover_stations(args.output_root)
    if len(stations) == 0:
        print("[warn] no stations found to process")
        return

    print(f"[info] stations_to_process={len(stations)}")

    tasks = [
        {
            "station_id": sid,
            "output_root": args.output_root,
            "forcing_dir": args.forcing_dir,
            "streamflow_dir": args.streamflow_dir,
            "device": args.device,
            "seq_len": args.seq_len,
            "pred_horizon": args.pred_horizon,
            "batch_size": args.batch_size,
            "include_q_input": args.include_q_input,
            "circuit_threshold": args.circuit_threshold,
            "max_edges": args.max_edges,
            "out_csv": args.out_csv,
            "compute_in_node_bridge": args.compute_in_node_bridge,
            "in_node_bridge_max_nodes": args.in_node_bridge_max_nodes,
            "in_node_out_json": args.in_node_out_json,
            "in_node_out_csv": args.in_node_out_csv,
            "bundle_out_json": args.bundle_out_json,
        }
        for sid in stations
    ]

    rows = []
    workers = max(1, int(args.num_workers))
    if workers == 1:
        for idx, t in enumerate(tasks, start=1):
            sid = str(t["station_id"])
            print(f"\n[{idx}/{len(tasks)}] processing station={sid}")
            result = _run_one(t)
            if result["status"] == "failed":
                print(f"[error] station={sid} failed: {result['error']}")
                if not args.continue_on_error:
                    rows.append(result)
                    break
            rows.append(result)
    else:
        print(f"[info] running with num_workers={workers}")
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_run_one, t): str(t["station_id"]) for t in tasks}
            done_n = 0
            for fut in cf.as_completed(fut_map):
                done_n += 1
                sid = fut_map[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"station_id": sid, "status": "failed", "output_csv": "", "error": f"{type(e).__name__}: {e}"}

                if result["status"] == "ok":
                    print(f"[{done_n}/{len(tasks)}] ok station={sid}")
                else:
                    print(f"[{done_n}/{len(tasks)}] failed station={sid}: {result['error']}")
                    if not args.continue_on_error:
                        for f in fut_map:
                            if not f.done():
                                f.cancel()
                rows.append(result)

    report_path = os.path.join(args.output_root, "real_edge_recompute_report.csv")
    pd.DataFrame(rows).to_csv(report_path, index=False)
    ok = sum(1 for r in rows if r["status"] == "ok")
    fail = len(rows) - ok
    print(f"\n[done] ok={ok}, failed={fail}, report={report_path}")


if __name__ == "__main__":
    main()
