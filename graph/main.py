#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import ExperimentConfig, load_dataset_configs
from train import run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run PVNN Section 6.2 experiments.")
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "disease", "airport"])
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder that contains dataset subfolders.")
    parser.add_argument("--model-types", nargs="+", default=["pvnn"], help="Model types to run (pvnn/hnn/hnn++/lnn/knn/fc).")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=40)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--curvature", type=float)
    parser.add_argument("--frechet-iters", nargs="*", type=int, help="List of GyroBN Frechet iterations (-1 for inf).")
    parser.add_argument("--time-test", action="store_true", help="Collect per-epoch timing stats.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to store raw results.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = PROJECT_ROOT / "configs" / "datasets.yaml"
    dataset_configs = load_dataset_configs(cfg_path)
    exp_cfg = ExperimentConfig(
        datasets=args.datasets,
        model_types=[m.lower() for m in args.model_types],
        data_root=args.data_root,
        runs=args.runs,
        base_seed=args.base_seed,
        time_test=args.time_test,
        frechet_iters=args.frechet_iters,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        override_dropout=args.dropout,
        override_weight_decay=args.weight_decay,
        override_curvature=args.curvature,
    )
    summary = run_experiments(exp_cfg, dataset_configs, args.data_root)
    for dataset, record in summary.items():
        if dataset == "timing":
            continue
        for label, runs in record["runs"].items():
            scores = [run["acc"] for run in runs]
            if scores:
                mean_acc = np.mean(scores)
                std_acc = np.std(scores)
                tag = f"{dataset.upper()} (iter={label})"
                print(f"{tag:<18}: {mean_acc:.2f}% +/- {std_acc:.2f}% ({len(scores)} runs)")
    if "timing" in summary:
        for dataset, (avg_time, n_epochs) in summary["timing"].items():
            print(f"[TIME] {dataset}: {avg_time:.3f}s over {n_epochs} epochs")
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

