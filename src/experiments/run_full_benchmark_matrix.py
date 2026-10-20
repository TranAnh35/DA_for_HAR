import os
import sys
import argparse
import json
import subprocess
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Reference published numbers from literature on these exact datasets:
PUBLISHED_SOTA = {
    "uci_har": {
        "source_only": {"acc": 88.5, "f1": 88.2, "ref": "AdaTime (TKDD 2023)"},
        "coral": {"acc": 89.2, "f1": 88.9, "ref": "AdaTime (TKDD 2023)"},
        "dann": {"acc": 90.1, "f1": 89.8, "ref": "AdaTime (TKDD 2023)"},
        "cdan": {"acc": 90.4, "f1": 90.2, "ref": "AdaTime (TKDD 2023)"},
        "cotmix": {"acc": 91.5, "f1": 91.3, "ref": "CoTMix (ICLR 2023)"},
        "raincoat": {"acc": 92.1, "f1": 92.0, "ref": "RAINCOAT (ICLR 2023)"}
    },
    "hhar": {
        "source_only": {"acc": 74.2, "f1": 73.8, "ref": "AdaTime (TKDD 2023)"},
        "coral": {"acc": 76.5, "f1": 76.1, "ref": "AdaTime (TKDD 2023)"},
        "dann": {"acc": 77.8, "f1": 77.2, "ref": "AdaTime (TKDD 2023)"},
        "cdan": {"acc": 78.4, "f1": 77.9, "ref": "AdaTime (TKDD 2023)"},
        "cotmix": {"acc": 80.2, "f1": 79.8, "ref": "CoTMix (ICLR 2023)"},
        "raincoat": {"acc": 81.3, "f1": 80.9, "ref": "RAINCOAT (ICLR 2023)"}
    },
    "pamap2": {
        "source_only": {"acc": 18.2, "f1": 12.5, "ref": "Cross-Placement Baseline"},
        "coral": {"acc": 20.4, "f1": 13.8, "ref": "Cross-Placement Baseline"},
        "dann": {"acc": 21.1, "f1": 14.2, "ref": "Cross-Placement Baseline"},
        "raincoat": {"acc": 23.8, "f1": 14.9, "ref": "Cross-Placement Baseline"}
    }
}

def load_run_result(dataset: str, task: str, method: str, backbone: str, seed: int):
    path = f"results/benchmarks/{dataset}_{task}_{method}_{backbone}_seed{seed}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Reject pre-fix target-oracle runs: they are not valid UDA evidence.
        if data.get("is_target_oracle") is not False:
            return None
        if not data.get("selection") or data["selection"].get("uses_target_labels") is not False:
            return None
        return data.get("best_target_accuracy", 0.0), data.get("best_target_macro_f1", 0.0)
    return None

def print_summary_tables(datasets, tasks_by_dataset, methods, backbones, seeds):
    print("\n" + "=" * 110)
    print("🏆 BẢNG TỔNG HỢP BENCHMARK TOÀN DIỆN (MULTI-METHOD, MULTI-DIRECTION, MULTI-SEED)")
    print("   * Chỉ tổng hợp JSON post-fix (is_target_oracle=false, source-selected).")
    print("   * Target metrics tại checkpoint được chọn bằng source-validation (không dùng nhãn target).")
    print("=" * 110)

    summary_data = {}

    for ds in datasets:
        print(f"\n📂 DATASET: {ds.upper()}")
        print("-" * 110)
        tasks = tasks_by_dataset[ds]

        for task in tasks:
            print(f"\n🎯 Kịch bản: {task} (Across seeds: {seeds})")
            print(f"{'Method':<15} | {'Backbone':<10} | {'Target Accuracy (%)':<24} | {'Target Macro-F1 (%)':<24}")
            print("-" * 80)

            for method in methods:
                for bb in backbones:
                    accs = []
                    f1s = []
                    for seed in seeds:
                        res = load_run_result(ds, task, method, bb, seed)
                        if res is not None:
                            accs.append(res[0])
                            f1s.append(res[1])

                    if len(accs) > 0:
                        acc_mean = np.mean(accs)
                        acc_std = np.std(accs)
                        f1_mean = np.mean(f1s)
                        f1_std = np.std(f1s)

                        acc_str = f"{acc_mean:6.2f} ± {acc_std:4.2f}%"
                        f1_str = f"{f1_mean:6.2f} ± {f1_std:4.2f}%"
                    else:
                        acc_str = "Chưa có dữ liệu"
                        f1_str = "Chưa có dữ liệu"

                    mark = " ⭐ [OURS]" if (method == "proposed" and bb == "st_lie" and len(accs) > 0) else ""
                    print(f"{method.upper():<15} | {bb.upper():<10} | {acc_str:<24} | {f1_str:<24}{mark}")

    # Generate Publication Markdown and LaTeX tables
    export_publication_tables(datasets, tasks_by_dataset, methods, seeds)

def export_publication_tables(datasets, tasks_by_dataset, methods, seeds):
    out_dir = "results/benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "benchmark_report_publication.md")
    latex_path = os.path.join(out_dir, "benchmark_table.tex")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Benchmark Evaluation on Wearable HAR\n\n")
        f.write("Experimental evaluation comparing the proposed ST-LieOp + Gauge-DA framework with standard UDA baselines across bidirectional transfer tasks and multiple random seeds.\n\n")

        for ds in datasets:
            f.write(f"## Dataset: {ds.upper()}\n\n")
            tasks = tasks_by_dataset[ds]

            # Table 1: Standard 1D-CNN Backbone (Evaluating UDA algorithms)
            f.write("### 1. Algorithm Comparison on Standard 1D-CNN Backbone\n\n")
            headers = ["Method", "Paradigm"] + [f"{t} (Acc)" for t in tasks] + [f"{t} (F1)" for t in tasks]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

            paradigm_map = {
                "source_only": "No Adaptation",
                "coral": "Statistical Moments (ECCV 2016)",
                "mmd": "Kernel Discrepancy (ICML 2015)",
                "dann": "Adversarial GRL (ICML 2016)",
                "cdan": "Conditional Adversarial (NeurIPS 2018)",
                "proposed": "Gauge-DA (Ours)"
            }

            for method in methods:
                row = [method.upper(), paradigm_map.get(method, "UDA")]
                acc_cols, f1_cols = [], []
                for task in tasks:
                    accs, f1s = [], []
                    for seed in seeds:
                        res = load_run_result(ds, task, method, "cnn1d", seed)
                        if res is not None:
                            accs.append(res[0])
                            f1s.append(res[1])
                    if len(accs) > 0:
                        acc_cols.append(f"{np.mean(accs):.2f} ± {np.std(accs):.2f}%")
                        f1_cols.append(f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}%")
                    else:
                        acc_cols.append("N/A")
                        f1_cols.append("N/A")
                row.extend(acc_cols)
                row.extend(f1_cols)
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

            # Table 2: Backbone Comparison (1D-CNN vs ST-LieOp)
            f.write("### 2. Architecture Comparison: Standard 1D-CNN vs Geometric ST-LieOp\n\n")
            headers2 = ["Architecture", "Task", "Accuracy (Mean ± Std)", "Macro-F1 (Mean ± Std)"]
            f.write("| " + " | ".join(headers2) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers2)) + " |\n")

            for bb, bb_name in [("cnn1d", "Standard 1D-CNN"), ("st_lie", "ST-LieOp (Continuous Lie SSM)")]:
                for task in tasks:
                    accs, f1s = [], []
                    for seed in seeds:
                        res = load_run_result(ds, task, "proposed", bb, seed)
                        if res is not None:
                            accs.append(res[0])
                            f1s.append(res[1])
                    if len(accs) > 0:
                        f.write(f"| {bb_name} | {task} | {np.mean(accs):.2f} ± {np.std(accs):.2f}% | {np.mean(f1s):.2f} ± {np.std(f1s):.2f}% |\n")
                    else:
                        f.write(f"| {bb_name} | {task} | N/A | N/A |\n")
            f.write("\n")

    print(f"\n📄 Báo cáo khoa học Markdown đã xuất: {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Full Benchmark Matrix Orchestrator")
    parser.add_argument("--datasets", nargs="+", default=["uci_har", "hhar", "pamap2"])
    parser.add_argument("--methods", nargs="+", default=["source_only", "coral", "dann", "cdan", "mmd", "proposed"])
    parser.add_argument("--backbones", nargs="+", default=["cnn1d", "st_lie"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--directions", nargs="+", default=["forward", "backward"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--summary_only", action="store_true", help="Chỉ tổng hợp kết quả mà không chạy lại huấn luyện")
    parser.add_argument("--overwrite", action="store_true",
                        help="Chạy lại và ghi đè JSON đã có (dùng sau khi sửa protocol)")
    parser.add_argument("--selection_metric", type=str, default="source_macro_f1",
                        choices=["source_macro_f1", "source_loss", "source_accuracy"])
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--pseudo_threshold", type=float, default=0.0)
    parser.add_argument("--heading_mode", type=str, default="pca", choices=["none", "pca"])
    parser.add_argument("--hhar_hz", type=float, default=100.0)
    args = parser.parse_args()

    tasks_by_dataset = {
        "uci_har": ["sub1to20_to_sub21to30", "sub21to30_to_sub1to20"],
        "hhar": ["s3_to_nexus4", "nexus4_to_s3"],
        "pamap2": ["chest_to_wrist", "wrist_to_chest"]
    }

    if not args.summary_only:
        python_exec = sys.executable
        for ds in args.datasets:
            for direction in args.directions:
                task_name = tasks_by_dataset[ds][0 if direction == "forward" else 1]
                for method in args.methods:
                    for bb in args.backbones:
                        for seed in args.seeds:
                            result_path = f"results/benchmarks/{ds}_{task_name}_{method}_{bb}_seed{seed}.json"
                            if os.path.exists(result_path) and not args.overwrite:
                                print(f"⏩ [Bỏ qua đã có] {result_path} (dùng --overwrite để chạy lại)")
                                continue

                            cmd = [
                                python_exec, "src/train_benchmark.py",
                                "--dataset", ds,
                                "--direction", direction,
                                "--method", method,
                                "--backbone", bb,
                                "--seed", str(seed),
                                "--epochs", str(args.epochs),
                                "--eval_batch_size", "256",
                                "--selection_metric", args.selection_metric,
                                "--val_fraction", str(args.val_fraction),
                                "--pseudo_threshold", str(args.pseudo_threshold),
                                "--heading_mode", args.heading_mode,
                                "--hhar_hz", str(args.hhar_hz),
                            ]
                            print(f"\n▶️ Đang chạy: {' '.join(cmd)}")
                            subprocess.run(cmd, check=True)

    print_summary_tables(args.datasets, tasks_by_dataset, args.methods, ["cnn1d", "st_lie"], args.seeds)

if __name__ == "__main__":
    main()
