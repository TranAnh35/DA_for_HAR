import sys
import os
import argparse
import json
import time
from typing import Dict, Any

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src package is in path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.uci_har_loader import load_uci_har
from src.data.hhar_loader import load_hhar
from src.data.pamap2_loader import load_pamap2
from src.models.decoupled_backbone import DecoupledSTLieModel
from src.losses.gauge_invariant_loss import GaugeInvariantUDALoss
from src.utils.metrics import compute_metrics

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, _, _ in data_loader:
            x = x.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_targets, all_preds)
    return metrics

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"🔧 Đang chạy trên thiết bị: {device}")
    print(f"📋 Chế độ huấn luyện: {args.mode.upper()} | Dataset: {args.dataset.upper()} | Epochs: {args.epochs}")

    # 1. Load Data
    if args.dataset == "uci_har":
        # Source: Subjects 1..20, Target: Subjects 21..30
        src_dataset, tgt_dataset = load_uci_har(
            data_dir="Data/UCI_HAR/UCI HAR Dataset",
            source_subjects=list(range(1, 21)),
            target_subjects=list(range(21, 31)),
            channels="6ch"
        )
        num_classes = 6
        in_channels = 6
    elif args.dataset == "hhar":
        # Cross-Device: Samsung Galaxy S3 (Source) -> LG Nexus 4 (Target)
        src_dataset, tgt_dataset = load_hhar(
            data_dir="Data/heterogeneity/Activity recognition exp",
            source_model=getattr(args, "source_device", "s3"),
            target_model=getattr(args, "target_device", "nexus4"),
            window_size=128,
            step_size=64
        )
        num_classes = 6
    elif args.dataset == "pamap2":
        # Cross-Placement: Chest (Source) -> Wrist (Target)
        src_dataset, tgt_dataset = load_pamap2(
            data_dir="Data/PAMAP2_Dataset",
            source_position=getattr(args, "source_position", "chest"),
            target_position=getattr(args, "target_position", "wrist"),
            window_size=128,
            step_size=64
        )
        num_classes = 12
        in_channels = 6
    else:
        raise NotImplementedError(f"Dataset {args.dataset} chưa được hỗ trợ.")

    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tgt_eval_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"📊 Dữ liệu: Source={len(src_dataset)} mẫu ({len(src_loader)} batches) | Target={len(tgt_dataset)} mẫu ({len(tgt_eval_loader)} batches)")

    # 2. Build Model
    model = DecoupledSTLieModel(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    criterion = GaugeInvariantUDALoss(
        lambda_gauge=args.lambda_gauge if args.mode == "uda" else 0.0,
        lambda_ent=args.lambda_ent if args.mode == "uda" else 0.0
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    history = []
    best_target_f1 = 0.0

    print("\n" + "=" * 78)
    print(f"{'Epoch':^7} | {'Total Loss':^10} | {'Src CE':^8} | {'Gauge Loss':^10} | {'Tgt Acc (%)':^11} | {'Tgt F1 (%)':^11} | {'Time':^6}")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()

        total_loss_epoch = 0.0
        src_loss_epoch = 0.0
        gauge_loss_epoch = 0.0
        num_batches = 0

        tgt_iter = iter(tgt_loader)

        for src_x, src_y, _, _ in src_loader:
            try:
                tgt_x, _, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_x, _, _, _ = next(tgt_iter)

            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()

            # Forward passes
            logits_s, z_s = model(src_x)
            logits_t, z_t = model(tgt_x)

            loss, loss_dict = criterion(logits_s, src_y, z_s, z_t, logits_t)
            loss.backward()

            # Gradient clipping for stable dynamics
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += loss.item()
            src_loss_epoch += loss_dict["loss_src"]
            gauge_loss_epoch += loss_dict["loss_gauge"]
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - start_time

        avg_total_loss = total_loss_epoch / max(num_batches, 1)
        avg_src_loss = src_loss_epoch / max(num_batches, 1)
        avg_gauge_loss = gauge_loss_epoch / max(num_batches, 1)

        # Evaluate on Target Domain
        tgt_metrics = evaluate(model, tgt_eval_loader, device)
        tgt_acc = tgt_metrics["accuracy"]
        tgt_f1 = tgt_metrics["macro_f1"]

        is_best = tgt_f1 > best_target_f1
        if is_best:
            best_target_f1 = tgt_f1
            torch.save(model.state_dict(), f"checkpoints/best_{args.mode}_{args.dataset}.pt")

        history.append({
            "epoch": epoch,
            "total_loss": avg_total_loss,
            "src_loss": avg_src_loss,
            "gauge_loss": avg_gauge_loss,
            "target_accuracy": tgt_acc,
            "target_macro_f1": tgt_f1
        })

        best_mark = " ⭐" if is_best else ""
        print(f"{epoch:^7d} | {avg_total_loss:^10.4f} | {avg_src_loss:^8.4f} | {avg_gauge_loss:^10.4f} | {tgt_acc:^11.2f} | {tgt_f1:^11.2f}{best_mark} | {epoch_time:^5.1f}s")

    print("=" * 78)
    print(f"\n🎉 Huấn luyện hoàn tất! Best Target Macro-F1 ({args.mode.upper()}): {best_target_f1:.2f}%")

    # Save training history
    log_file = f"results/{args.mode}_{args.dataset}_history.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"📁 Lịch sử huấn luyện đã được lưu vào: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình ST-LieOp cho bài toán Domain Adaptation trên HAR")
    parser.add_argument("--mode", type=str, default="uda", choices=["source_only", "uda"], help="Chế độ: source_only hoặc uda")
    parser.add_argument("--dataset", type=str, default="uci_har", choices=["uci_har", "hhar", "pamap2"])
    parser.add_argument("--epochs", type=int, default=25, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64, help="Kích thước đặc trưng ẩn")
    parser.add_argument("--d_state", type=int, default=32, help="Kích thước trạng thái liên tục SSM")
    parser.add_argument("--num_layers", type=int, default=2, help="Số lớp ST-LieOp")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_gauge", type=float, default=0.5, help="Trọng số mất mát Gram Gauge")
    parser.add_argument("--lambda_ent", type=float, default=0.05, help="Trọng số mất mát Entropy đích")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--source_device", type=str, default="s3", help="Thiết bị nguồn HHAR (s3, s3mini, nexus4, etc.)")
    parser.add_argument("--target_device", type=str, default="nexus4", help="Thiết bị đích HHAR")
    parser.add_argument("--source_position", type=str, default="chest", help="Vị trí cảm biến nguồn PAMAP2 (chest, wrist, ankle)")
    parser.add_argument("--target_position", type=str, default="wrist", help="Vị trí cảm biến đích PAMAP2 (wrist, ankle, chest)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
