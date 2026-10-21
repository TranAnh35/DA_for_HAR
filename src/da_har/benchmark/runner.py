"""Benchmark runner: task x method x seed -> JSON.

Usage (from the repository root)::

    conda run -n da_har python -m src.da_har.benchmark.runner \
        --dataset hhar --shift cross_device --methods source_only coral dann \
        --seeds 42 43 44 --epochs 15

Results are written to ``results/da_benchmark/<run_id>.json``.  The JSON is
self-describing: it stores the task definition, the protocol audit, the
source-selection rule, per-seed metrics and seed-level aggregates.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .domains import SHIFT_TYPES, build_tasks
from .methods import ADAPTATION_METHODS
from .stats import paired_bootstrap, summarize
from .trainer import TrainConfig, train_task

DEFAULT_OUT_DIR = os.path.join("results", "da_benchmark")


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc = [r["metrics"]["target_accuracy"] for r in records]
    f1 = [r["metrics"]["target_macro_f1"] for r in records]
    src_val = [r["metrics"]["source_val_macro_f1"] for r in records]
    acc_mean, acc_std, n = summarize(acc)
    f1_mean, f1_std, _ = summarize(f1)
    sv_mean, sv_std, _ = summarize(src_val)
    return {
        "n_seeds": n,
        "target_accuracy_mean": acc_mean,
        "target_accuracy_std": acc_std,
        "target_macro_f1_mean": f1_mean,
        "target_macro_f1_std": f1_std,
        "source_val_macro_f1_mean": sv_mean,
        "source_val_macro_f1_std": sv_std,
        "seeds": [int(r["seed"]) for r in records],
    }


def run(
    dataset: str,
    shift: str,
    methods: Sequence[str],
    seeds: Sequence[int],
    config: TrainConfig,
    *,
    max_windows: Optional[int] = None,
    out_dir: str = DEFAULT_OUT_DIR,
    quiet: bool = False,
) -> Dict[str, Any]:
    specs = build_tasks(dataset, shift, max_windows=max_windows)
    if not specs:
        raise SystemExit(f"No tasks available for dataset={dataset!r} shift={shift!r}")

    started = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{dataset}_{shift}_{started}"
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset,
        "shift": shift,
        "created": started,
        "selection_rule": "source_val_macro_f1",
        "is_target_oracle": False,
        "methods": list(methods),
        "seeds": list(seeds),
        "config": asdict(config),
        "tasks": [],
    }

    for spec in specs:
        task = spec.materialize()
        if not quiet:
            print(
                f"\n[{spec.name}] source={task.source.num_windows} windows "
                f"target={task.target.num_windows} windows "
                f"slots={len(task.source.channels)}"
            )
        task_entry: Dict[str, Any] = {
            "task": spec.name,
            "summary": task.summary(),
            "results": [],
            "aggregates": {},
            "paired_vs_source_only": {},
        }
        for method in methods:
            for seed in seeds:
                cfg = TrainConfig(**{**asdict(config), "seed": int(seed)})
                result = train_task(task, method_name=method, config=cfg)
                task_entry["results"].append(result)
                if not quiet:
                    m = result["metrics"]
                    print(
                        f"  {method:<12} seed={seed:<5} "
                        f"src_val_f1={m['source_val_macro_f1']:.4f} "
                        f"tgt_acc={m['target_accuracy']:.4f} "
                        f"tgt_f1={m['target_macro_f1']:.4f}"
                    )
        for method in methods:
            records = [r for r in task_entry["results"] if r["method"] == method]
            task_entry["aggregates"][method] = _aggregate(records)
        if "source_only" in methods:
            base = {
                r["seed"]: r["metrics"]["target_macro_f1"]
                for r in task_entry["results"]
                if r["method"] == "source_only"
            }
            for method in methods:
                if method == "source_only":
                    continue
                diffs = [
                    r["metrics"]["target_macro_f1"] - base[r["seed"]]
                    for r in task_entry["results"]
                    if r["method"] == method and r["seed"] in base
                ]
                task_entry["paired_vs_source_only"][method] = paired_bootstrap(diffs)
        payload["tasks"].append(task_entry)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    payload["output_path"] = out_path
    if not quiet:
        print(f"\nWrote {out_path}")
        _print_summary(payload)
    return payload


def _print_summary(payload: Dict[str, Any]) -> None:
    print("\n=== Summary (target macro-F1, mean +/- std over seeds) ===")
    for task in payload["tasks"]:
        print(f"{task['task']}:")
        for method, agg in task["aggregates"].items():
            print(
                f"  {method:<12} acc={agg['target_accuracy_mean'] * 100:6.2f} "
                f"f1={agg['target_macro_f1_mean'] * 100:6.2f} "
                f"+/- {agg['target_macro_f1_std'] * 100:4.2f} "
                f"(n={agg['n_seeds']})"
            )
        for method, stats in task["paired_vs_source_only"].items():
            if not np.isfinite(stats["mean"]):
                continue
            print(
                f"  delta[{method} vs source_only] = {stats['mean'] * 100:+.2f} "
                f"[{stats['ci_low'] * 100:+.2f}, {stats['ci_high'] * 100:+.2f}] "
                f"p={stats['p_value']:.3f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=False, default="uci_har")
    parser.add_argument("--shift", required=False, default=None, choices=list(SHIFT_TYPES))
    parser.add_argument("--methods", nargs="+", default=["source_only", "coral", "dann"],
                        choices=list(ADAPTATION_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--da-weight", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Deterministic per-domain subsample for smoke runs.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tasks:
        for spec in build_tasks(args.dataset, args.shift or "cross_subject"):
            print(f"{spec.name:<40} shift={spec.shift}")
        return 0
    if args.shift is None:
        raise SystemExit("--shift is required (unless --list-tasks is used)")
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        da_weight=args.da_weight,
        hidden=args.hidden,
        device=args.device,
        seed=args.seeds[0],
    )
    run(
        args.dataset,
        args.shift,
        args.methods,
        args.seeds,
        config,
        max_windows=args.max_windows,
        out_dir=args.out_dir,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
