import sys
import os
import math

import torch

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src package is in path
sys.path.insert(0, os.path.abspath("."))

from src.data.uci_har_loader import load_uci_har
from src.models.canonical_projector import CanonicalOrientationProjector
from src.models.st_lie_operator import STLieOperatorBlock
from src.models.decoupled_backbone import DecoupledSTLieModel
from src.losses.gauge_invariant_loss import GaugeInvariantUDALoss
from src.utils.metrics import compute_metrics
from src.utils.selection import split_source_train_val


def run_tests():
    print("=" * 66)
    print("🚀 Bắt đầu Kiểm thử Toàn diện Pipeline src/ (da_har environment)")
    print("=" * 66)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Thiết bị thực thi: {device}")

    # 1. Data loader
    print("\n[1/7] Kiểm tra Data Loader: UCI_HAR (Data/UCI_HAR)...")
    src_ds, tgt_ds = load_uci_har(
        data_dir="Data/UCI_HAR/UCI HAR Dataset",
        source_subjects=list(range(1, 11)),
        target_subjects=list(range(11, 16)),
    )
    print(f"  ✅ Source Samples: {len(src_ds)}, Shape: {src_ds.data.shape}, fs={src_ds.sampling_rate:g}Hz")
    print(f"  ✅ Target Samples: {len(tgt_ds)}, Shape: {tgt_ds.data.shape}, fs={tgt_ds.sampling_rate:g}Hz")
    assert src_ds.data.shape[1] == 6, "Expected 6 channels"
    assert src_ds.data.shape[2] == 128, "Expected 128 time steps"

    # 2. Projector: shape, antipodal gravity, yaw invariance
    print("\n[2/7] Kiểm tra Module 01: Canonical Projector (antipodal + yaw)...")
    batch_x = src_ds.data[:16].to(device)
    projector = CanonicalOrientationProjector(filter_window=25, sampling_rate=50.0, heading_mode="pca").to(device)
    x_canon = projector(batch_x, sampling_rate=50.0)
    print(f"  ✅ Shape: {batch_x.shape} -> {x_canon.shape}")
    assert x_canon.shape == batch_x.shape

    # Device upside down: gravity = -e_z must be flipped to +e_z.
    antipodal = torch.zeros(2, 6, 128, device=device)
    antipodal[:, 2, :] = -9.81
    anti_out = projector(antipodal, sampling_rate=50.0)
    assert anti_out[:, 2, :].mean() > 9.0, "Antipodal gravity was not corrected"
    assert anti_out[:, 0, :].abs().max() < 1e-4
    assert anti_out[:, 1, :].abs().max() < 1e-4
    print("  ✅ Antipodal gravity (-e_z) được xoay đúng về +e_z")

    # Yaw-180 invariance on a deterministic motion signal.
    t = torch.arange(128, device=device, dtype=torch.float32) / 50.0
    base = torch.zeros(4, 6, 128, device=device)
    base[:, 0, :] = 2.0 * torch.sin(2 * math.pi * 1.3 * t) + 0.5
    base[:, 1, :] = 1.5 * torch.cos(2 * math.pi * 1.1 * t)
    base[:, 2, :] = 9.81
    base[:, 3, :] = 0.3 * torch.sin(2 * math.pi * 0.7 * t)
    base[:, 4, :] = 0.2 * torch.cos(2 * math.pi * 0.9 * t)
    base[:, 5, :] = 0.1 * torch.sin(2 * math.pi * 1.5 * t)

    yaw_pi = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    base_rot = base.clone()
    base_rot[:, :3, :] = torch.einsum('ij,bjt->bit', yaw_pi, base[:, :3, :])
    base_rot[:, 3:6, :] = torch.einsum('ij,bjt->bit', yaw_pi, base[:, 3:6, :])

    out_a = projector(base, sampling_rate=50.0)
    out_b = projector(base_rot, sampling_rate=50.0)
    yaw_diff = (out_a - out_b).abs().max().item()
    print(f"  ✅ Yaw-180 invariance max diff: {yaw_diff:.2e}")
    assert yaw_diff < 2e-4, f"Yaw canonicalization is not invariant: {yaw_diff}"

    # 3. ST-LieOp with physical delta_t and residual path
    print("\n[3/7] Kiểm tra Module 02: ST-LieOp (exact ZOH + residual + delta_t)...")
    st_block = STLieOperatorBlock(d_in=6, d_model=64, d_state=32).to(device)
    h_out = st_block(x_canon, delta_t=0.02)
    print(f"  ✅ Canonical Input: {x_canon.shape} -> ST-LieOp Output: {h_out.shape}")
    assert h_out.shape == (16, 64, 128)
    assert torch.isfinite(h_out).all(), "ST-LieOp produced NaN/Inf"

    # 4. End-to-end backbone
    print("\n[4/7] Kiểm tra Mô hình Toàn vẹn: DecoupledSTLieModel...")
    model = DecoupledSTLieModel(
        in_channels=6,
        num_classes=6,
        d_model=64,
        d_state=32,
        num_layers=2,
        sampling_rate=50.0,
        heading_mode="pca",
    ).to(device)

    batch_x_src = src_ds.data[:16].to(device)
    batch_y_src = src_ds.labels[:16].to(device)
    batch_x_tgt = tgt_ds.data[:16].to(device)

    logits_s, z_s = model(batch_x_src, delta_t=0.02, sampling_rate=50.0)
    logits_t, z_t = model(batch_x_tgt, delta_t=0.02, sampling_rate=50.0)

    print(f"  ✅ Logits Source: {logits_s.shape}, Latent z_s: {z_s.shape}")
    print(f"  ✅ Logits Target: {logits_t.shape}, Latent z_t: {z_t.shape}")
    assert logits_s.shape == (16, 6)
    assert z_s.shape == (16, 128)
    assert torch.isfinite(z_s).all()

    # 5. Class-conditional prototype loss
    print("\n[5/7] Kiểm tra Module 03: Prototype Alignment UDA Loss...")
    criterion = GaugeInvariantUDALoss(lambda_gauge=0.5, lambda_ent=0.1)
    loss, loss_dict = criterion(logits_s, batch_y_src, z_s, z_t, logits_t)
    print(f"  ✅ Total UDA Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"      - {k}: {v:.4f}")
    assert "loss_da" in loss_dict and "loss_proto" in loss_dict
    assert torch.isfinite(loss)

    loss.backward()
    print("  ✅ Backward pass thành công, gradient hữu hạn.")
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient at {name}"

    # Deterministic class overlap to exercise the class-conditional branch.
    z_s2 = torch.randn(12, 16, device=device)
    z_t2 = torch.randn(12, 16, device=device)
    y_s2 = torch.arange(12, device=device) % 4
    logits_s2 = torch.randn(12, 6, device=device)
    logits_t2 = torch.zeros(12, 6, device=device)
    logits_t2[torch.arange(12, device=device), torch.arange(12, device=device) % 4] = 6.0
    _, proto_dict = criterion(logits_s2, y_s2, z_s2, z_t2, logits_t2)
    print(f"  ✅ Class-conditional check: proto_classes={proto_dict['proto_classes']:.0f}, "
          f"loss_proto={proto_dict['loss_proto']:.4f}")
    assert proto_dict["proto_classes"] >= 4
    assert proto_dict["loss_da"] == proto_dict["loss_proto"]

    # 6. Subject-held-out source-validation split
    print("\n[6/7] Kiểm tra Source-Validation Split (không rò rỉ target)...")
    train_subset, val_subset, info = split_source_train_val(src_ds, val_fraction=0.2, seed=42)
    train_domains = set(src_ds.domain_ids[train_subset.indices].tolist())
    val_domains = set(src_ds.domain_ids[val_subset.indices].tolist())
    print(f"  ✅ Split: {info}")
    assert info["split"] == "subject_held_out"
    assert train_domains.isdisjoint(val_domains), "Subject leakage between train/val"
    assert len(train_subset) > 0 and len(val_subset) > 0

    # 7. Metrics
    preds_s = torch.argmax(logits_s, dim=-1)
    metrics = compute_metrics(batch_y_src, preds_s)
    print(f"\n[7/7] Metrics batch: Acc={metrics['accuracy']:.2f}%, Macro-F1={metrics['macro_f1']:.2f}%")

    print("\n" + "=" * 66)
    print("🎉 TẤT CẢ CÁC BƯỚC KIỂM THỬ ĐÃ THÀNH CÔNG 100%!")
    print("=" * 66)


if __name__ == "__main__":
    run_tests()
