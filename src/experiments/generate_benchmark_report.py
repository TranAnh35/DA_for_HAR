import os
import sys
import json
import statistics
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_postfix_benchmarks(root="results/benchmarks"):
    """Return grouped valid benchmark runs and the number of stale files.

    A file is valid only when the trainer explicitly recorded source-selected
    protocol metadata.  Pre-fix target-oracle files are counted as stale and
    never aggregated.
    """
    groups = defaultdict(list)
    stale = 0
    if not os.path.isdir(root):
        return groups, stale
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json"):
            continue
        data = load_json(os.path.join(root, name))
        if not isinstance(data, dict):
            continue
        if data.get("is_target_oracle") is not False:
            stale += 1
            continue
        selection = data.get("selection") or {}
        if selection.get("uses_target_labels") is not False:
            stale += 1
            continue
        args = data.get("args") or {}
        acc = data.get("best_target_accuracy")
        f1 = data.get("best_target_macro_f1")
        if not isinstance(acc, (int, float)) or not isinstance(f1, (int, float)):
            continue
        key = (
            str(args.get("dataset", "?")).lower(),
            str(args.get("direction", "?")).lower(),
            str(args.get("method", "?")).lower(),
            str(args.get("backbone", "?")).lower(),
        )
        groups[key].append((float(acc), float(f1)))
    return groups, stale


def history_version(history):
    """Classify a training-history file by protocol version.

    Legacy files have no ``is_best``/``source_val_*`` fields and were selected
    by target macro-F1.  They must never be presented as post-fix results.
    """
    if not history:
        return "missing"
    for row in history:
        if "is_best" in row or "source_val_accuracy" in row:
            return "post_fix"
    return "legacy"


def selected_entry(history):
    """Return the source-validation-selected epoch for a post-fix history.

    Never take ``max(target_accuracy)``: that is target-oracle selection and
    invalidates the benchmark.  Callers must gate on ``history_version`` so
    legacy pre-fix histories are not presented as results.
    """
    if not history:
        return None
    for row in history:
        if row.get("is_best"):
            return row
    return history[-1]


def main():
    print("=" * 80)
    print("📈 TỔNG HỢP KẾT QUẢ BENCHMARK & BÁO CÁO THỰC NGHIỆM KHOA HỌC")
    print("=" * 80)

    datasets = ["uci_har", "hhar", "pamap2"]

    # 1. Training History & Target Performance
    print("\n1. KẾT QUẢ THÍCH ỨNG MIỀN (chỉ số post-fix; legacy pre-fix bị loại):")
    print("-" * 80)
    print(f"{'Dataset':<15} | {'Source-Only Acc (%)':<20} | {'UDA Acc (%)':<15} | {'Adaptation Gain (Δ)':<20}")
    print("-" * 80)

    for ds in datasets:
        uda_file = f"results/uda_{ds}_history.json"
        if not os.path.exists(uda_file):
            if ds == "pamap2":
                uda_file = "results/uda_pamap2_chest_to_wrist_history.json"
            elif ds == "hhar":
                uda_file = "results/uda_hhar_s3_to_nexus4_history.json"

        uda_data = load_json(uda_file)
        uda_version = history_version(uda_data)
        uda_row = selected_entry(uda_data) if uda_version == "post_fix" else None
        best_uda_acc = uda_row["target_accuracy"] if uda_row else None
        best_uda_f1 = uda_row["target_macro_f1"] if uda_row else None

        src_file = f"results/source_only_{ds}_history.json"
        if not os.path.exists(src_file):
            if ds == "pamap2":
                src_file = "results/source_only_pamap2_chest_to_wrist_history.json"
            elif ds == "hhar":
                src_file = "results/source_only_hhar_s3_to_nexus4_history.json"

        src_data = load_json(src_file)
        src_version = history_version(src_data)
        src_row = selected_entry(src_data) if src_version == "post_fix" else None
        best_src_acc = src_row["target_accuracy"] if src_row else None
        best_src_f1 = src_row["target_macro_f1"] if src_row else None

        if src_version == "legacy":
            src_str = "Legacy pre-fix (bị loại)"
        elif src_version == "missing":
            src_str = "Chưa có dữ liệu"
        else:
            src_str = f"{best_src_acc:.2f}% (F1: {best_src_f1:.2f}%)" if best_src_acc is not None else "N/A"

        if uda_version == "legacy":
            uda_str = "Legacy pre-fix (bị loại)"
        elif uda_version == "missing":
            uda_str = "Chưa có dữ liệu"
        else:
            uda_str = f"{best_uda_acc:.2f}% (F1: {best_uda_f1:.2f}%)" if best_uda_acc is not None else "N/A"

        if best_src_acc is not None and best_uda_acc is not None:
            diff = best_uda_acc - best_src_acc
            diff_f1 = best_uda_f1 - best_src_f1
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f}% (ΔF1: {'+' if diff_f1 >= 0 else ''}{diff_f1:.2f}%)"
        else:
            diff_str = "N/A"

        print(f"{ds.upper():<15} | {src_str:<20} | {uda_str:<15} | {diff_str:<20}")

    # 1b. Valid post-fix benchmark JSONs (source-selected).
    bench_groups, stale_count = load_postfix_benchmarks()
    print("\n1b. BENCHMARK JSON POST-FIX (source-selected; target chỉ để báo cáo):")
    print("-" * 100)
    if not bench_groups:
        print(f"Chưa có run hợp lệ. {stale_count} file pre-fix/target-oracle bị loại.")
    else:
        print(f"{'Dataset':<10} | {'Direction':<12} | {'Method':<12} | {'Backbone':<8} | {'Accuracy (%)':<16} | {'Macro-F1 (%)':<16} | Seeds")
        print("-" * 100)
        for (ds, direction, method, backbone), vals in sorted(bench_groups.items()):
            accs = [v[0] for v in vals]
            f1s = [v[1] for v in vals]
            acc_std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
            f1_std = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0
            print(
                f"{ds.upper():<10} | {direction:<12} | {method:<12} | {backbone:<8} | "
                f"{statistics.mean(accs):6.2f} ± {acc_std:4.2f}     | "
                f"{statistics.mean(f1s):6.2f} ± {f1_std:4.2f}     | {len(vals)}"
            )
        print(f"\n* {stale_count} file pre-fix/target-oracle bị loại khỏi bảng này.")

    # 2. Rotation Robustness Stress-Test
    print("\n2. KIỂM CHỨNG TÍNH BẤT BIẾN QUAY HÌNH HỌC 3D (O_g(3) INVARIANCE):")
    print("-" * 80)
    print(f"{'Dataset':<10} | {'Angle':<8} | {'Proposed Acc (%)':<20} | {'Ablated Acc (%)':<20} | {'Invariance Margin':<15}")
    print("-" * 80)

    for ds in datasets:
        rot_file = f"results/rotation_stress_test_{ds}.json"
        rot_data = load_json(rot_file)
        if not rot_data:
            print(f"{ds.upper():<10} | Chưa có dữ liệu")
            continue

        # Legacy rotation files predate truthful ablation metadata.
        if "ablation_mode" not in rot_data:
            print(f"{ds.upper():<10} | Legacy pre-fix (bị loại; cùng checkpoint bypass)")
            continue

        mode = rot_data.get("ablation_mode", "unknown")
        independent = rot_data.get("independent_ablation", False)
        print(f"{ds.upper():<10} | ablation_mode={mode} | independent={independent}")

        angles = rot_data.get("angles", [])
        prop_acc = rot_data.get("proposed", {}).get("accuracy_mean", [])
        prop_std = rot_data.get("proposed", {}).get("accuracy_std", [])
        abl_acc = rot_data.get("ablated", {}).get("accuracy_mean", [])
        abl_std = rot_data.get("ablated", {}).get("accuracy_std", [])

        test_angles = [0.0, 45.0, 90.0, 180.0]
        for a in test_angles:
            if a in angles:
                idx = angles.index(a)
                p_m = prop_acc[idx]
                p_s = prop_std[idx]
                a_m = abl_acc[idx]
                a_s = abl_std[idx]
                margin = p_m - a_m
                sign = "+" if margin >= 0 else ""
                print(
                    f"{ds.upper():<10} | {a:^6.0f}°  | {p_m:6.2f} ± {p_s:4.2f}%       | "
                    f"{a_m:6.2f} ± {a_s:4.2f}%       | {sign}{margin:5.2f}%"
                )
        print("-" * 80)


if __name__ == "__main__":
    main()
