"""Run independent frozen experiment jobs in persistent spawned processes."""
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from .common import digest, write_json
from .runner import run, job_path, summarize


def execute(payload):
    cfg, manifest, index = payload
    folder = job_path(Path(cfg["output_root"]), manifest["jobs"][index])
    folder.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(station=None, only=None, max_jobs=None,
                              job_index=index, no_summary=True, fail_fast=True)
    started = time.perf_counter()
    try:
        with (folder/"worker.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"WORKER pid={os.getpid()} job_index={index}", flush=True)
                run(cfg, manifest, args)
        return {"index": index, "status": "complete", "seconds": time.perf_counter()-started}
    except Exception:
        return {"index": index, "status": "failed", "seconds": time.perf_counter()-started,
                "traceback": traceback.format_exc()}


def run_parallel(cfg, manifest, args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OMP_NUM_THREADS"] = str(cfg["torch_threads"])
    os.environ["MKL_NUM_THREADS"] = str(cfg["torch_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(cfg["torch_threads"])
    selected = [(i, j) for i, j in enumerate(manifest["jobs"])
                if (not args.station or j["station"] == args.station)
                and (not args.only or j["model"] == args.only)]
    if args.max_jobs is not None:
        selected = selected[:args.max_jobs]
    if not selected:
        raise ValueError("No jobs match requested filters")
    pending = []
    for index, job in selected:
        path = job_path(Path(cfg["output_root"]), job)/"summary.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("run_sha256") != digest({"job": job, "manifest": manifest}):
                raise RuntimeError(f"Incompatible completed job: {path}")
        else:
            pending.append(index)
    started = time.perf_counter()
    report = {"workers": args.workers, "planned": len(selected), "already_complete": len(selected)-len(pending),
              "results": []}
    print(f"PARALLEL: {args.workers} workers; {len(pending)} pending jobs", flush=True)
    # Spawn avoids sharing CUDA state or RNG across independent basin models.
    with mp.get_context("spawn").Pool(min(args.workers, max(1,len(pending)))) as pool:
        for result in pool.imap_unordered(execute, [(cfg,manifest,i) for i in pending], chunksize=1):
            report["results"].append(result)
            report["elapsed_seconds"] = time.perf_counter()-started
            write_json(Path(cfg["output_root"])/"parallel_progress.json", report)
            print(f"JOB {result['index']} {result['status']} {result['seconds']:.1f}s; {len(report['results'])}/{len(pending)}", flush=True)
            if result["status"] == "failed" and args.fail_fast:
                pool.terminate()
                summarize(cfg,manifest)
                raise RuntimeError("Job failed with --fail-fast; see parallel_progress.json and worker.log")
    summarize(cfg,manifest)
    if any(r["status"] == "failed" for r in report["results"]):
        raise RuntimeError("Some jobs failed; inspect failure.json and worker.log, then rerun to retry")
