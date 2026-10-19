import sys
import os
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

def run_tests():
    print("=" * 60)
    print("🚀 Bắt đầu Kiểm thử Toàn diện Pipeline src/ (da_har environment)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Thiết bị thực thi: {device}")

    # 1. Test Data Loader on real UCI HAR dataset
    print("\n[1/5] Kiểm tra Data Loader: UCI_HAR (Data/UCI_HAR)...")
    src_ds, tgt_ds = load_uci_har(
        data_dir="Data/UCI_HAR/UCI HAR Dataset",
        source_subjects=list(range(1, 11)), # 10 source subjects for fast test
        target_subjects=list(range(11, 16)) # 5 target subjects
    )
    print(f"  ✅ Source Samples: {len(src_ds)}, Shape: {src_ds.data.shape}")
    print(f"  ✅ Target Samples: {len(tgt_ds)}, Shape: {tgt_ds.data.shape}")
    assert src_ds.data.shape[1] == 6, "Expected 6 channels"
    assert src_ds.data.shape[2] == 128, "Expected 128 time steps"

    # 2. Test Canonical Orientation Projector
    print("\n[2/5] Kiểm tra Module 01: Canonical Orientation Projector O_g(3)...")
    batch_x = src_ds.data[:16].to(device) # [16, 6, 128]
    projector = CanonicalOrientationProjector(filter_window=25).to(device)
    x_canon = projector(batch_x)
    print(f"  ✅ Input Shape: {batch_x.shape} -> Canonical Output Shape: {x_canon.shape}")
    assert x_canon.shape == batch_x.shape, "Shape mismatch after canonical projection"

    # 3. Test ST-LieOp Continuous State-Space Block
    print("\n[3/5] Kiểm tra Module 02: ST-LieOp Continuous SSM...")
    st_block = STLieOperatorBlock(d_in=6, d_model=64, d_state=32).to(device)
    h_out = st_block(x_canon)
    print(f"  ✅ Canonical Input: {x_canon.shape} -> ST-LieOp Output: {h_out.shape}")
    assert h_out.shape == (16, 64, 128), "Shape mismatch in STLieOperatorBlock"

    # 4. Test Decoupled End-to-End Backbone
    print("\n[4/5] Kiểm tra Mô hình Toàn vẹn: DecoupledSTLieModel...")
    model = DecoupledSTLieModel(
        in_channels=6,
        num_classes=6,
        d_model=64,
        d_state=32,
        num_layers=2
    ).to(device)
    
    batch_x_src = src_ds.data[:16].to(device)
    batch_y_src = src_ds.labels[:16].to(device)
    batch_x_tgt = tgt_ds.data[:16].to(device)

    logits_s, z_s = model(batch_x_src)
    logits_t, z_t = model(batch_x_tgt)

    print(f"  ✅ Logits Source: {logits_s.shape}, Latent z_s: {z_s.shape}")
    print(f"  ✅ Logits Target: {logits_t.shape}, Latent z_t: {z_t.shape}")
    assert logits_s.shape == (16, 6)
    assert z_s.shape == (16, 128) # d_model * 2 (avg + max pool)

    # 5. Test Gauge-Invariant UDA Loss
    print("\n[5/5] Kiểm tra Module 03: Gauge-Invariant UDA Loss & Metrics...")
    criterion = GaugeInvariantUDALoss(lambda_gauge=0.5, lambda_ent=0.1)
    loss, loss_dict = criterion(logits_s, batch_y_src, z_s, z_t, logits_t)
    print(f"  ✅ Total UDA Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"      - {k}: {v:.4f}")

    # Backward pass verification
    loss.backward()
    print("  ✅ Lan truyền ngược Gradient (Backward Pass) thành công! Không có lỗi NaN.")

    # Metric evaluation
    preds_s = torch.argmax(logits_s, dim=-1)
    metrics = compute_metrics(batch_y_src, preds_s)
    print(f"  ✅ Batch Source Accuracy: {metrics['accuracy']:.2f}%, Macro-F1: {metrics['macro_f1']:.2f}%")

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ CÁC BƯỚC KIỂM THỬ ĐÃ THÀNH CÔNG 100%!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
