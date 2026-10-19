import os
import sys
import json
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    print("=" * 80)
    print("📈 TỔNG HỢP KẾT QUẢ BENCHMARK & BÁO CÁO THỰC NGHIỆM KHOA HỌC")
    print("=" * 80)

    datasets = ["uci_har", "hhar", "pamap2"]
    
    # 1. Training History & Target Performance
    print("\n1. KẾT QUẢ THÍCH ỨNG MIỀN (DOMAIN ADAPTATION PERFORMANCE):")
    print("-" * 80)
    print(f"{'Dataset':<15} | {'Source-Only Acc (%)':<20} | {'UDA Acc (%)':<15} | {'Adaptation Gain (Δ)':<20}")
    print("-" * 80)

    for ds in datasets:
        # Load UDA history
        uda_file = f"results/uda_{ds}_history.json"
        if not os.path.exists(uda_file):
            # check alternatives
            if ds == "pamap2":
                uda_file = "results/uda_pamap2_chest_to_wrist_history.json"
            elif ds == "hhar":
                uda_file = "results/uda_hhar_s3_to_nexus4_history.json"
        
        uda_data = load_json(uda_file)
        best_uda_acc = max([x["target_accuracy"] for x in uda_data]) if uda_data else None
        best_uda_f1 = max([x["target_macro_f1"] for x in uda_data]) if uda_data else None

        # Load Source-Only history
        src_file = f"results/source_only_{ds}_history.json"
        if not os.path.exists(src_file):
            if ds == "pamap2":
                src_file = "results/source_only_pamap2_chest_to_wrist_history.json"
            elif ds == "hhar":
                src_file = "results/source_only_hhar_s3_to_nexus4_history.json"

        src_data = load_json(src_file)
        best_src_acc = max([x["target_accuracy"] for x in src_data]) if src_data else None
        best_src_f1 = max([x["target_macro_f1"] for x in src_data]) if src_data else None

        src_str = f"{best_src_acc:.2f}% (F1: {best_src_f1:.2f}%)" if best_src_acc is not None else "Đang huấn luyện..."
        uda_str = f"{best_uda_acc:.2f}% (F1: {best_uda_f1:.2f}%)" if best_uda_acc is not None else "N/A"
        
        if best_src_acc is not None and best_uda_acc is not None:
            diff = best_uda_acc - best_src_acc
            diff_f1 = best_uda_f1 - best_src_f1
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f}% (ΔF1: {'+' if diff_f1 >= 0 else ''}{diff_f1:.2f}%)"
        else:
            diff_str = "N/A"

        print(f"{ds.upper():<15} | {src_str:<20} | {uda_str:<15} | {diff_str:<20}")

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

        angles = rot_data.get("angles", [])
        prop_acc = rot_data.get("proposed", {}).get("accuracy_mean", [])
        prop_std = rot_data.get("proposed", {}).get("accuracy_std", [])
        abl_acc = rot_data.get("ablated", {}).get("accuracy_mean", [])
        abl_std = rot_data.get("ablated", {}).get("accuracy_std", [])

        # Display at 0 deg, 45 deg, 90 deg, 180 deg
        test_angles = [0.0, 45.0, 90.0, 180.0]
        for a in test_angles:
            if a in angles:
                idx = angles.index(a)
                p_m = prop_acc[idx]
                p_s = prop_std[idx]
                a_m = abl_acc[idx]
                a_s = abl_std[idx]
                margin = p_m - a_m
                print(f"{ds.upper():<10} | {a:^6.0f}°  | {p_m:6.2f} ± {p_s:4.2f}%       | {a_m:6.2f} ± {a_s:4.2f}%       | +{margin:5.2f}%")
        print("-" * 80)

if __name__ == "__main__":
    main()
