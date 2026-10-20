import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import json
import math
from typing import List, Optional

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


def generate_random_rotation_matrix(
    angle_degrees: float,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Generate a rotation in SO(3) by ``angle_degrees`` about a random axis."""
    if abs(angle_degrees) < 1e-6:
        return torch.eye(3, device=device, dtype=torch.float32)

    rng = np.random.default_rng() if rng is None else rng
    theta = math.radians(angle_degrees)

    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    n = torch.tensor([x, y, z], device=device, dtype=torch.float32)

    K = torch.tensor([
        [0.0, -n[2], n[1]],
        [n[2], 0.0, -n[0]],
        [-n[1], n[0], 0.0],
    ], device=device, dtype=torch.float32)

    I = torch.eye(3, device=device, dtype=torch.float32)
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * torch.matmul(K, K)


def apply_rotation(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate accelerometer (0:3) and gyroscope (3:6) channels by R."""
    B, C, T = x.shape
    acc = x[:, :3, :]
    gyro = x[:, 3:6, :]
    acc_rot = torch.einsum('ij,bjt->bit', R, acc)
    gyro_rot = torch.einsum('ij,bjt->bit', R, gyro)
    if C > 6:
        aux = x[:, 6:, :]
        return torch.cat([acc_rot, gyro_rot, aux], dim=1)
    return torch.cat([acc_rot, gyro_rot], dim=1)


def evaluate_on_rotated_target(
    model: nn.Module,
    loader: DataLoader,
    rotations: List[torch.Tensor],
    device: torch.device,
    delta_t: float,
    sampling_rate: float,
):
    """Evaluate a model under a fixed list of rotation matrices."""
    model.eval()
    acc_list = []
    f1_list = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for R in rotations:
            all_preds = []
            all_targets = []
            for x, y, _, _ in loader:
                x = x.to(device, non_blocking=True)
                x_rot = apply_rotation(x, R)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits, _ = model(x_rot, delta_t=delta_t, sampling_rate=sampling_rate)
                all_preds.append(torch.argmax(logits, dim=-1).cpu())
                all_targets.append(y)

            metrics = compute_metrics(torch.cat(all_targets), torch.cat(all_preds))
            acc_list.append(metrics["accuracy"])
            f1_list.append(metrics["macro_f1"])

    return (
        float(np.mean(acc_list)),
        float(np.std(acc_list)),
        float(np.mean(f1_list)),
        float(np.std(f1_list)),
    )


def _build_model(
    ckpt: dict,
    num_classes: int,
    in_channels: int,
    tgt_fs: float,
    use_projector: bool,
    device: torch.device,
) -> DecoupledSTLieModel:
    a = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = DecoupledSTLieModel(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=a.get("d_model", 64),
        d_state=a.get("d_state", 32),
        num_layers=a.get("num_layers", 2),
        dropout=a.get("dropout", 0.2),
        filter_window=a.get("filter_window", None),
        sampling_rate=tgt_fs,
        heading_mode=a.get("heading_mode", "pca"),
        use_projector=use_projector,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Random 3D Rotation Stress-Test for Wearable HAR UDA")
    parser.add_argument("--dataset", type=str, default="uci_har", choices=["uci_har", "pamap2", "hhar"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint của model proposed")
    parser.add_argument("--ablated-checkpoint", type=str, default=None,
                        help="Checkpoint được huấn luyện riêng KHÔNG có projector (ablation độc lập)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_repeats", type=int, default=5, help="Số trục quay ngẫu nhiên cho mỗi góc")
    parser.add_argument("--seed", type=int, default=42, help="Seed cho các trục quay (tái lập được)")
    parser.add_argument("--hhar_hz", type=float, default=100.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"🚀 Random 3D Rotation Stress-Test: {args.dataset.upper()} | Device: {device}")

    # 1. Load target data
    if args.dataset == "uci_har":
        _, tgt_dataset = load_uci_har(
            data_dir="Data/UCI_HAR/UCI HAR Dataset",
            source_subjects=list(range(1, 21)),
            target_subjects=list(range(21, 31)),
            channels="6ch",
        )
        num_classes, in_channels = 6, 6
        default_ckpt = "checkpoints/best_uda_uci_har.pt"
    elif args.dataset == "pamap2":
        _, tgt_dataset = load_pamap2(
            data_dir="Data/PAMAP2_Dataset",
            source_position="chest",
            target_position="wrist",
            window_size=128,
            step_size=64,
        )
        num_classes, in_channels = 12, 6
        default_ckpt = "checkpoints/best_uda_pamap2_chest_to_wrist.pt"
    else:  # hhar
        _, tgt_dataset = load_hhar(
            data_dir="Data/heterogeneity/Activity recognition exp",
            source_model="s3",
            target_model="nexus4",
            window_size=128,
            step_size=64,
            target_hz=args.hhar_hz,
        )
        num_classes, in_channels = 6, 6
        default_ckpt = "checkpoints/best_uda_hhar_s3_to_nexus4.pt"

    ckpt_path = args.checkpoint or default_ckpt
    assert os.path.exists(ckpt_path), f"Không tìm thấy checkpoint: {ckpt_path}"
    print(f"📦 Checkpoint proposed: {ckpt_path}")

    tgt_fs = float(tgt_dataset.sampling_rate)
    delta_t = 1.0 / tgt_fs
    target_loader = DataLoader(
        tgt_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda")
    )
    print(f"📊 Target: {len(tgt_dataset)} mẫu | fs={tgt_fs:g}Hz")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}

    checkpoint_protocol = ckpt.get("protocol_version", "legacy") if isinstance(ckpt, dict) else "legacy"
    checkpoint_is_oracle = ckpt.get("is_target_oracle") if isinstance(ckpt, dict) else None
    protocol_valid = checkpoint_protocol == "source_selected_v1" and checkpoint_is_oracle is False
    if not protocol_valid:
        print("⚠️ Checkpoint proposed không có protocol source_selected_v1/is_target_oracle=false; "
              "kết quả rotation sẽ được đánh dấu legacy, không dùng làm bằng chứng post-fix.")

    model_proposed = _build_model(ckpt, num_classes, in_channels, tgt_fs, True, device)
    missing, unexpected = model_proposed.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ Proposed checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    # 2. Ablated model
    independent = args.ablated_checkpoint is not None
    if independent:
        print(f"📦 Checkpoint ablated độc lập: {args.ablated_checkpoint}")
        abl_ckpt = torch.load(args.ablated_checkpoint, map_location=device, weights_only=False)
        abl_state = abl_ckpt["model_state_dict"] if isinstance(abl_ckpt, dict) and "model_state_dict" in abl_ckpt else abl_ckpt
        abl_state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in abl_state.items()}
        abl_protocol = abl_ckpt.get("protocol_version", "legacy") if isinstance(abl_ckpt, dict) else "legacy"
        abl_is_oracle = abl_ckpt.get("is_target_oracle") if isinstance(abl_ckpt, dict) else None
        protocol_valid = protocol_valid and (abl_protocol == "source_selected_v1" and abl_is_oracle is False)
        if not protocol_valid:
            print("⚠️ Checkpoint ablated không có protocol post-fix; cặp ablation sẽ bị đánh dấu legacy.")
        model_ablated = _build_model(abl_ckpt, num_classes, in_channels, tgt_fs, False, device)
        missing, unexpected = model_ablated.load_state_dict(abl_state, strict=False)
        if missing or unexpected:
            print(f"⚠️ Ablated checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
        ablation_mode = "independent_checkpoint"
        ablated_label = "Ablation (no projector, independently trained)"
    else:
        # Same weights, projector bypassed at inference.  This is NOT an
        # independent ablation; it isolates the contribution of the
        # parameter-free projector only.
        model_ablated = _build_model(ckpt, num_classes, in_channels, tgt_fs, False, device)
        model_ablated.load_state_dict(state_dict, strict=False)
        ablation_mode = "inference_bypass"
        ablated_label = "Inference bypass (same weights, no projector)"
        print("⚠️ Không có --ablated-checkpoint: 'ablated' chỉ là inference-time bypass, "
              "KHÔNG phải ablation độc lập.")

    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0]

    # Precompute rotations once so proposed and ablated see exactly the same R.
    rng = np.random.default_rng(args.seed)
    rotations_by_angle = {}
    for angle in angles:
        n = 1 if abs(angle) < 1e-6 else args.num_repeats
        rotations_by_angle[angle] = [
            generate_random_rotation_matrix(angle, device, rng) for _ in range(n)
        ]

    results = {
        "dataset": args.dataset,
        "protocol_version": "rotation_v2",
        "checkpoint": ckpt_path,
        "checkpoint_protocol": checkpoint_protocol,
        "checkpoint_is_target_oracle": checkpoint_is_oracle,
        "protocol_valid": bool(protocol_valid),
        "ablated_checkpoint": args.ablated_checkpoint,
        "ablation_mode": ablation_mode,
        "independent_ablation": bool(independent),
        "seed": args.seed,
        "target_sampling_rate": tgt_fs,
        "angles": angles,
        "rotation_protocol": "same random SO(3) rotations for proposed and ablated; acc+gyro rotated",
        "proposed": {"accuracy_mean": [], "accuracy_std": [], "macro_f1_mean": [], "macro_f1_std": []},
        "ablated": {"accuracy_mean": [], "accuracy_std": [], "macro_f1_mean": [], "macro_f1_std": []},
    }

    print("\n" + "=" * 100)
    print(f"{'Góc (độ)':^10} | {'Proposed Acc (%)':^20} | {'Proposed F1 (%)':^20} | {'Ablated Acc (%)':^18} | {'Ablated F1 (%)':^18}")
    print("=" * 100)

    for angle in tqdm(angles, desc="Đang quét các góc xoay 3D"):
        rots = rotations_by_angle[angle]
        acc_p_m, acc_p_s, f1_p_m, f1_p_s = evaluate_on_rotated_target(
            model_proposed, target_loader, rots, device, delta_t, tgt_fs
        )
        results["proposed"]["accuracy_mean"].append(acc_p_m)
        results["proposed"]["accuracy_std"].append(acc_p_s)
        results["proposed"]["macro_f1_mean"].append(f1_p_m)
        results["proposed"]["macro_f1_std"].append(f1_p_s)

        acc_a_m, acc_a_s, f1_a_m, f1_a_s = evaluate_on_rotated_target(
            model_ablated, target_loader, rots, device, delta_t, tgt_fs
        )
        results["ablated"]["accuracy_mean"].append(acc_a_m)
        results["ablated"]["accuracy_std"].append(acc_a_s)
        results["ablated"]["macro_f1_mean"].append(f1_a_m)
        results["ablated"]["macro_f1_std"].append(f1_a_s)

        print(
            f"{angle:^10.1f} | {acc_p_m:^10.2f} ± {acc_p_s:^6.2f} | {f1_p_m:^10.2f} ± {f1_p_s:^6.2f} | "
            f"{acc_a_m:^10.2f} ± {acc_a_s:^6.2f} | {f1_a_m:^10.2f} ± {f1_a_s:^6.2f}"
        )

    print("=" * 100)

    os.makedirs("results", exist_ok=True)
    json_path = f"results/rotation_stress_test_{args.dataset}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Kết quả chi tiết: {json_path}")

    plt.figure(figsize=(10, 5), dpi=300)
    angles_arr = np.array(angles)

    plt.subplot(1, 2, 1)
    p_acc = np.array(results["proposed"]["accuracy_mean"])
    p_acc_std = np.array(results["proposed"]["accuracy_std"])
    plt.plot(angles_arr, p_acc, 'o-', color='#1f77b4', linewidth=2.5, label='ST-LieOp (Proposed)')
    plt.fill_between(angles_arr, p_acc - p_acc_std, p_acc + p_acc_std, color='#1f77b4', alpha=0.2)

    a_acc = np.array(results["ablated"]["accuracy_mean"])
    a_acc_std = np.array(results["ablated"]["accuracy_std"])
    plt.plot(angles_arr, a_acc, 's--', color='#d62728', linewidth=2.0, label=ablated_label)
    plt.fill_between(angles_arr, a_acc - a_acc_std, a_acc + a_acc_std, color='#d62728', alpha=0.2)

    plt.title(f"{args.dataset.upper()}: Accuracy under 3D Rotation", fontsize=12, fontweight='bold')
    plt.xlabel(r"Random 3D Rotation Angle $\theta$ (degrees)", fontsize=11)
    plt.ylabel("Target Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='best', fontsize=8)

    plt.subplot(1, 2, 2)
    p_f1 = np.array(results["proposed"]["macro_f1_mean"])
    p_f1_std = np.array(results["proposed"]["macro_f1_std"])
    plt.plot(angles_arr, p_f1, 'o-', color='#1f77b4', linewidth=2.5, label='ST-LieOp (Proposed)')
    plt.fill_between(angles_arr, p_f1 - p_f1_std, p_f1 + p_f1_std, color='#1f77b4', alpha=0.2)

    a_f1 = np.array(results["ablated"]["macro_f1_mean"])
    a_f1_std = np.array(results["ablated"]["macro_f1_std"])
    plt.plot(angles_arr, a_f1, 's--', color='#d62728', linewidth=2.0, label=ablated_label)
    plt.fill_between(angles_arr, a_f1 - a_f1_std, a_f1 + a_f1_std, color='#d62728', alpha=0.2)

    plt.title(f"{args.dataset.upper()}: Macro-F1 under 3D Rotation", fontsize=12, fontweight='bold')
    plt.xlabel(r"Random 3D Rotation Angle $\theta$ (degrees)", fontsize=11)
    plt.ylabel("Target Macro-F1 (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='best', fontsize=8)

    plt.tight_layout()
    plot_path = f"results/rotation_stress_test_{args.dataset}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Đồ thị: {plot_path}")


if __name__ == "__main__":
    main()
