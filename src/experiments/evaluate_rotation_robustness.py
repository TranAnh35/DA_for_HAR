import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("."))

from src.data.uci_har_loader import load_uci_har
from src.data.pamap2_loader import load_pamap2
from src.data.hhar_loader import load_hhar
from src.models.decoupled_backbone import DecoupledSTLieModel
from src.utils.metrics import compute_metrics

def generate_random_rotation_matrix(angle_degrees: float, device: torch.device) -> torch.Tensor:
    """
    Generates a 3D rotation matrix R in SO(3) rotating by angle_degrees
    around a randomly chosen unit axis n on the unit 2-sphere S^2.
    """
    if abs(angle_degrees) < 1e-6:
        return torch.eye(3, device=device, dtype=torch.float32)

    theta = math.radians(angle_degrees)

    # Random unit axis uniformly distributed on sphere
    z = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    n = torch.tensor([x, y, z], device=device, dtype=torch.float32)

    # Rodrigues rotation formula: R = I + sin(theta) * [n]_x + (1 - cos(theta)) * [n]_x^2
    # [n]_x is the skew-symmetric cross-product matrix
    K = torch.tensor([
        [0.0, -n[2], n[1]],
        [n[2], 0.0, -n[0]],
        [-n[1], n[0], 0.0]
    ], device=device, dtype=torch.float32)

    I = torch.eye(3, device=device, dtype=torch.float32)
    R = I + math.sin(theta) * K + (1.0 - math.cos(theta)) * torch.matmul(K, K)
    return R

def apply_rotation(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotates the 3D accelerometer channels [0:3] and 3D gyroscope channels [3:6]
    of tensor x: [B, C, T] by rotation matrix R: [3, 3].
    """
    B, C, T = x.shape
    acc = x[:, :3, :]   # [B, 3, T]
    gyro = x[:, 3:6, :] # [B, 3, T]

    # Matrix multiplication: [3, 3] x [B, 3, T] -> [B, 3, T]
    acc_rot = torch.einsum('ij,bjt->bit', R, acc)
    gyro_rot = torch.einsum('ij,bjt->bit', R, gyro)

    if C > 6:
        aux = x[:, 6:, :]
        return torch.cat([acc_rot, gyro_rot, aux], dim=1)
    return torch.cat([acc_rot, gyro_rot], dim=1)

def evaluate_on_rotated_target(
    model: nn.Module,
    loader: DataLoader,
    angle_degrees: float,
    num_repeats: int,
    device: torch.device
):
    """
    Evaluates model across num_repeats random rotation axes for the given angle.
    Returns:
        acc_mean, acc_std, f1_mean, f1_std
    """
    model.eval()
    repeats = 1 if abs(angle_degrees) < 1e-6 else num_repeats
    acc_list = []
    f1_list = []

    with torch.no_grad():
        for rep in range(repeats):
            R = generate_random_rotation_matrix(angle_degrees, device)
            all_preds = []
            all_targets = []

            for x, y, _, _ in loader:
                x = x.to(device, non_blocking=True)
                x_rot = apply_rotation(x, R)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    logits, _ = model(x_rot)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(y)

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = compute_metrics(all_targets, all_preds)
            acc_list.append(metrics["accuracy"])
            f1_list.append(metrics["macro_f1"])

    return (
        float(np.mean(acc_list)),
        float(np.std(acc_list)),
        float(np.mean(f1_list)),
        float(np.std(f1_list))
    )

def main():
    parser = argparse.ArgumentParser(description="Random 3D Rotation Stress-Test for Wearable HAR UDA")
    parser.add_argument("--dataset", type=str, default="uci_har", choices=["uci_har", "pamap2", "hhar"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint model cần test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_repeats", type=int, default=5, help="Số lần lấy mẫu trục quay ngẫu nhiên mỗi góc")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"🚀 Bắt đầu Random 3D Rotation Stress-Test trên: {args.dataset.upper()} | Thiết bị: {device}")

    # 1. Load Data
    if args.dataset == "uci_har":
        _, tgt_dataset = load_uci_har(
            data_dir="Data/UCI_HAR/UCI HAR Dataset",
            source_subjects=list(range(1, 21)),
            target_subjects=list(range(21, 31)),
            channels="6ch"
        )
        num_classes = 6
        in_channels = 6
        default_ckpt = "checkpoints/best_uda_uci_har.pt"
    elif args.dataset == "pamap2":
        _, tgt_dataset = load_pamap2(
            data_dir="Data/PAMAP2_Dataset",
            source_position="chest",
            target_position="wrist",
            window_size=128,
            step_size=64
        )
        num_classes = 12
        in_channels = 6
        default_ckpt = "checkpoints/best_uda_pamap2_chest_to_wrist.pt"
    elif args.dataset == "hhar":
        _, tgt_dataset = load_hhar(
            data_dir="Data/heterogeneity/Activity recognition exp",
            source_model="s3",
            target_model="nexus4",
            window_size=128,
            step_size=64
        )
        num_classes = 6
        in_channels = 6
        default_ckpt = "checkpoints/best_uda_hhar.pt"

    ckpt_path = args.checkpoint or default_ckpt
    assert os.path.exists(ckpt_path), f"Không tìm thấy checkpoint: {ckpt_path}"
    print(f"📦 Đang nạp checkpoint từ: {ckpt_path}")

    target_loader = DataLoader(
        tgt_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )
    print(f"📊 Tập kiểm thử Target: {len(tgt_dataset)} mẫu ({len(target_loader)} batches)")

    # 2. Build Models (Proposed with Canonical Projector vs Ablated Without Projector)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Model A: Proposed ST-LieOp (with O_g(3) Canonical Projector)
    model_proposed = DecoupledSTLieModel(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=ckpt.get("args", {}).get("d_model", 64) if isinstance(ckpt, dict) else 64,
        d_state=ckpt.get("args", {}).get("d_state", 32) if isinstance(ckpt, dict) else 32,
        num_layers=ckpt.get("args", {}).get("num_layers", 2) if isinstance(ckpt, dict) else 2,
        use_projector=True
    ).to(device)
    model_proposed.load_state_dict(state_dict, strict=False)

    # Model B: Ablation (WITHOUT Canonical Projector - Raw IMU input)
    model_ablated = DecoupledSTLieModel(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=ckpt.get("args", {}).get("d_model", 64) if isinstance(ckpt, dict) else 64,
        d_state=ckpt.get("args", {}).get("d_state", 32) if isinstance(ckpt, dict) else 32,
        num_layers=ckpt.get("args", {}).get("num_layers", 2) if isinstance(ckpt, dict) else 2,
        use_projector=False
    ).to(device)
    model_ablated.load_state_dict(state_dict, strict=False)

    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0]

    results = {
        "dataset": args.dataset,
        "checkpoint": ckpt_path,
        "angles": angles,
        "proposed": {"accuracy_mean": [], "accuracy_std": [], "macro_f1_mean": [], "macro_f1_std": []},
        "ablated": {"accuracy_mean": [], "accuracy_std": [], "macro_f1_mean": [], "macro_f1_std": []}
    }

    print("\n" + "=" * 90)
    print(f"{'Góc (độ)':^10} | {'Proposed Acc (%)':^20} | {'Proposed F1 (%)':^20} | {'Ablated Acc (%)':^18} | {'Ablated F1 (%)':^18}")
    print("=" * 90)

    for angle in tqdm(angles, desc="Đang quét các góc xoay 3D"):
        # Evaluate Proposed Model
        acc_p_m, acc_p_s, f1_p_m, f1_p_s = evaluate_on_rotated_target(
            model_proposed, target_loader, angle, args.num_repeats, device
        )
        results["proposed"]["accuracy_mean"].append(acc_p_m)
        results["proposed"]["accuracy_std"].append(acc_p_s)
        results["proposed"]["macro_f1_mean"].append(f1_p_m)
        results["proposed"]["macro_f1_std"].append(f1_p_s)

        # Evaluate Ablated Model (No Projector)
        acc_a_m, acc_a_s, f1_a_m, f1_a_s = evaluate_on_rotated_target(
            model_ablated, target_loader, angle, args.num_repeats, device
        )
        results["ablated"]["accuracy_mean"].append(acc_a_m)
        results["ablated"]["accuracy_std"].append(acc_a_s)
        results["ablated"]["macro_f1_mean"].append(f1_a_m)
        results["ablated"]["macro_f1_std"].append(f1_a_s)

        p_acc_str = f"{acc_p_m:.2f} ± {acc_p_s:.2f}"
        p_f1_str = f"{f1_p_m:.2f} ± {f1_p_s:.2f}"
        a_acc_str = f"{acc_a_m:.2f} ± {acc_a_s:.2f}"
        a_f1_str = f"{f1_a_m:.2f} ± {f1_a_s:.2f}"

        print(f"{angle:^10.1f} | {p_acc_str:^20} | {p_f1_str:^20} | {a_acc_str:^18} | {a_f1_str:^18}")

    print("=" * 90)

    # Save JSON results
    os.makedirs("results", exist_ok=True)
    json_path = f"results/rotation_stress_test_{args.dataset}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Kết quả chi tiết đã được lưu tại: {json_path}")

    # Plot Curves
    plt.figure(figsize=(10, 5), dpi=300)

    # Plot 1: Accuracy Degradation Curve
    plt.subplot(1, 2, 1)
    angles_arr = np.array(angles)
    
    # Proposed curve
    p_acc = np.array(results["proposed"]["accuracy_mean"])
    p_acc_std = np.array(results["proposed"]["accuracy_std"])
    plt.plot(angles_arr, p_acc, 'o-', color='#1f77b4', linewidth=2.5, label='ST-LieOp (Proposed)')
    plt.fill_between(angles_arr, p_acc - p_acc_std, p_acc + p_acc_std, color='#1f77b4', alpha=0.2)

    # Ablated curve
    a_acc = np.array(results["ablated"]["accuracy_mean"])
    a_acc_std = np.array(results["ablated"]["accuracy_std"])
    plt.plot(angles_arr, a_acc, 's--', color='#d62728', linewidth=2.0, label='Ablation (w/o Canonical Projector)')
    plt.fill_between(angles_arr, a_acc - a_acc_std, a_acc + a_acc_std, color='#d62728', alpha=0.2)

    plt.title(f"{args.dataset.upper()}: Accuracy under 3D Rotation", fontsize=12, fontweight='bold')
    plt.xlabel(r"Random 3D Rotation Angle $\theta$ (degrees)", fontsize=11)
    plt.ylabel("Target Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='best')

    # Plot 2: Macro-F1 Degradation Curve
    plt.subplot(1, 2, 2)
    p_f1 = np.array(results["proposed"]["macro_f1_mean"])
    p_f1_std = np.array(results["proposed"]["macro_f1_std"])
    plt.plot(angles_arr, p_f1, 'o-', color='#1f77b4', linewidth=2.5, label='ST-LieOp (Proposed)')
    plt.fill_between(angles_arr, p_f1 - p_f1_std, p_f1 + p_f1_std, color='#1f77b4', alpha=0.2)

    a_f1 = np.array(results["ablated"]["macro_f1_mean"])
    a_f1_std = np.array(results["ablated"]["macro_f1_std"])
    plt.plot(angles_arr, a_f1, 's--', color='#d62728', linewidth=2.0, label='Ablation (w/o Canonical Projector)')
    plt.fill_between(angles_arr, a_f1 - a_f1_std, a_f1 + a_f1_std, color='#d62728', alpha=0.2)

    plt.title(f"{args.dataset.upper()}: Macro-F1 under 3D Rotation", fontsize=12, fontweight='bold')
    plt.xlabel(r"Random 3D Rotation Angle $\theta$ (degrees)", fontsize=11)
    plt.ylabel("Target Macro-F1 (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='best')

    plt.tight_layout()
    plot_path = f"results/rotation_stress_test_{args.dataset}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Đồ thị kiểm chứng đã được lưu tại: {plot_path}")

if __name__ == "__main__":
    main()
