import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("."))

from src.data.uci_har_loader import load_uci_har
from src.data.pamap2_loader import load_pamap2
from src.data.hhar_loader import load_hhar
from src.models.decoupled_backbone import DecoupledSTLieModel

def extract_all_features(model, loader, device, max_samples=2000):
    model.eval()
    features = []
    labels = []
    total = 0

    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                z = model.extract_features(x)
            features.append(z.cpu().numpy())
            labels.append(y.numpy())
            total += len(x)
            if total >= max_samples:
                break

    features = np.concatenate(features, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    return features, labels

def main():
    parser = argparse.ArgumentParser(description="t-SNE Manifold Visualization for ST-LieOp + Gauge-DA")
    parser.add_argument("--dataset", type=str, default="uci_har", choices=["uci_har", "pamap2", "hhar"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_samples", type=int, default=1500, help="Số lượng mẫu tối đa mỗi miền để vẽ t-SNE")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"🎨 Đang trích xuất đặc trưng & vẽ t-SNE cho: {args.dataset.upper()} | Thiết bị: {device}")

    # 1. Load Data
    if args.dataset == "uci_har":
        src_ds, tgt_ds = load_uci_har(
            data_dir="Data/UCI_HAR/UCI HAR Dataset",
            source_subjects=list(range(1, 21)),
            target_subjects=list(range(21, 31)),
            channels="6ch"
        )
        num_classes = 6
        in_channels = 6
        default_ckpt = "checkpoints/best_uda_uci_har.pt"
        class_names = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
    elif args.dataset == "pamap2":
        src_ds, tgt_ds = load_pamap2(
            data_dir="Data/PAMAP2_Dataset",
            source_position="chest",
            target_position="wrist",
            window_size=128,
            step_size=64
        )
        num_classes = 12
        in_channels = 6
        default_ckpt = "checkpoints/best_uda_pamap2_chest_to_wrist.pt"
        class_names = [f"Act_{i}" for i in range(12)]
    elif args.dataset == "hhar":
        src_ds, tgt_ds = load_hhar(
            data_dir="Data/heterogeneity/Activity recognition exp",
            source_model="s3",
            target_model="nexus4",
            window_size=128,
            step_size=64
        )
        num_classes = 6
        in_channels = 6
        default_ckpt = "checkpoints/best_uda_hhar.pt"
        class_names = ["Biking", "Sitting", "Standing", "Walking", "StairsUp", "StairsDown"]

    ckpt_path = args.checkpoint or default_ckpt
    assert os.path.exists(ckpt_path), f"Không tìm thấy checkpoint: {ckpt_path}"
    print(f"📦 Đang nạp checkpoint từ: {ckpt_path}")

    src_loader = DataLoader(src_ds, batch_size=256, shuffle=True, pin_memory=(device.type == "cuda"))
    tgt_loader = DataLoader(tgt_ds, batch_size=256, shuffle=True, pin_memory=(device.type == "cuda"))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = DecoupledSTLieModel(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=ckpt.get("args", {}).get("d_model", 64) if isinstance(ckpt, dict) else 64,
        d_state=ckpt.get("args", {}).get("d_state", 32) if isinstance(ckpt, dict) else 32,
        num_layers=ckpt.get("args", {}).get("num_layers", 2) if isinstance(ckpt, dict) else 2
    ).to(device)
    model.load_state_dict(state_dict, strict=False)

    print("🔍 Đang trích xuất vector đặc trưng ẩn z...")
    z_s, y_s = extract_all_features(model, src_loader, device, max_samples=args.max_samples)
    z_t, y_t = extract_all_features(model, tgt_loader, device, max_samples=args.max_samples)

    all_features = np.concatenate([z_s, z_t], axis=0)
    domain_labels = np.array([0] * len(z_s) + [1] * len(z_t))
    all_classes = np.concatenate([y_s, y_t], axis=0)

    print(f"📊 Chạy t-SNE trên {len(all_features)} vector đặc trưng...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42)
    embedding_2d = tsne.fit_transform(all_features)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot 1: Domain Alignment (Source vs Target)
    ax1 = axes[0]
    ax1.scatter(embedding_2d[domain_labels == 0, 0], embedding_2d[domain_labels == 0, 1],
                c='#1f77b4', label='Source Domain', alpha=0.5, s=15, edgecolors='none')
    ax1.scatter(embedding_2d[domain_labels == 1, 0], embedding_2d[domain_labels == 1, 1],
                c='#d62728', label='Target Domain', alpha=0.5, s=15, edgecolors='none')
    ax1.set_title(f"{args.dataset.upper()}: Domain Alignment (Source vs Target)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Subplot 2: Activity Class Clustering
    ax2 = axes[1]
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')
    for c in range(num_classes):
        mask = (all_classes == c)
        c_name = class_names[c] if c < len(class_names) else f"Class_{c}"
        ax2.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    color=cmap(c % 20), label=c_name, alpha=0.6, s=15, edgecolors='none')
    ax2.set_title(f"{args.dataset.upper()}: Semantic Activity Clustering", fontsize=12, fontweight='bold')
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax2.legend(loc='best', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = f"results/tsne_{args.dataset}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"🎉 Biểu đồ t-SNE đã được lưu tại: {plot_path}")

if __name__ == "__main__":
    main()
