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

from tqdm import tqdm
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

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, use_amp: bool = True) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []

    pbar_eval = tqdm(data_loader, desc="🔍 Đang đánh giá Target", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            for x, y, _, _ in pbar_eval:
                x = x.to(device, non_blocking=True)
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
        in_channels = 6
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

    pin_mem = (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    eval_batch_size = getattr(args, "eval_batch_size", 256)
    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_mem)
    tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_mem)
    tgt_eval_loader = DataLoader(tgt_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=pin_mem)

    print(f"📊 Dữ liệu: Source={len(src_dataset)} mẫu ({len(src_loader)} batches) | Target={len(tgt_dataset)} mẫu ({len(tgt_eval_loader)} batches, eval_bs={eval_batch_size})")

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

    use_amp = (device.type == "cuda") and not getattr(args, "no_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("⚡ Đã kích hoạt Automatic Mixed Precision (AMP FP16) với Tensor Cores!")

    criterion = GaugeInvariantUDALoss(
        lambda_gauge=args.lambda_gauge if args.mode == "uda" else 0.0,
        lambda_ent=args.lambda_ent if args.mode == "uda" else 0.0
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    history = []
    best_target_f1 = 0.0
    start_epoch = 1

    # Checkpoint paths
    prefix = f"{args.mode}_{args.dataset}"
    if args.dataset == "pamap2":
        prefix = f"{prefix}_{getattr(args, 'source_position', 'chest')}_to_{getattr(args, 'target_position', 'wrist')}"
    elif args.dataset == "hhar":
        prefix = f"{prefix}_{getattr(args, 'source_device', 's3')}_to_{getattr(args, 'target_device', 'nexus4')}"

    last_ckpt_path = f"checkpoints/last_{prefix}.pt"
    best_ckpt_path = f"checkpoints/best_{prefix}.pt"
    log_file = f"results/{prefix}_history.json"

    # Resume logic
    resume_target = args.resume_path
    if not resume_target and args.resume:
        if os.path.exists(last_ckpt_path):
            resume_target = last_ckpt_path
        elif os.path.exists(best_ckpt_path):
            resume_target = best_ckpt_path

    if resume_target and os.path.exists(resume_target):
        print(f"🔄 Đang khôi phục quá trình huấn luyện từ: {resume_target} ...")
        checkpoint = torch.load(resume_target, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None and use_amp:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_target_f1 = checkpoint.get("best_target_f1", 0.0)
            history = checkpoint.get("history", [])
            print(f"✅ Khôi phục thành công! Tiếp tục từ Epoch {start_epoch} (Best Target F1 trước đó: {best_target_f1:.2f}%)")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Khôi phục model weights từ raw checkpoint! (Bắt đầu từ Epoch 1)")
    elif args.resume:
        print(f"⚠️ Không tìm thấy checkpoint tại '{last_ckpt_path}'. Bắt đầu huấn luyện mới từ Epoch 1.")

    if start_epoch > args.epochs:
        print(f"✅ Mô hình đã hoàn thành đủ {args.epochs} epochs trước đó. Không cần huấn luyện thêm.")
        return

    print("\n" + "=" * 78)
    print(f"{'Epoch':^7} | {'Total Loss':^10} | {'Src CE':^8} | {'Gauge Loss':^10} | {'Tgt Acc (%)':^11} | {'Tgt F1 (%)':^11} | {'Time':^6}")
    print("=" * 78)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()

        total_loss_epoch = 0.0
        src_loss_epoch = 0.0
        gauge_loss_epoch = 0.0
        num_batches = 0

        is_uda = (args.mode == "uda")
        tgt_iter = iter(tgt_loader) if is_uda else None

        pbar = tqdm(src_loader, desc=f"Epoch {epoch:2d}/{args.epochs}", dynamic_ncols=True, leave=False)
        for src_x, src_y, _, _ in pbar:
            src_x = src_x.to(device, non_blocking=True)
            src_y = src_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward passes with AMP
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits_s, z_s = model(src_x)
                if is_uda:
                    try:
                        tgt_x, _, _, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(tgt_loader)
                        tgt_x, _, _, _ = next(tgt_iter)
                    tgt_x = tgt_x.to(device, non_blocking=True)
                    logits_t, z_t = model(tgt_x)
                else:
                    logits_t, z_t = None, None

                loss, loss_dict = criterion(logits_s, src_y, z_s, z_t, logits_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping for stable dynamics
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            total_loss_epoch += loss_val
            src_loss_epoch += loss_dict["loss_src"]
            gauge_loss_epoch += loss_dict["loss_gauge"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "src": f"{loss_dict['loss_src']:.3f}",
                "gauge": f"{loss_dict['loss_gauge']:.3f}"
            })

        scheduler.step()
        epoch_time = time.time() - start_time

        avg_total_loss = total_loss_epoch / max(num_batches, 1)
        avg_src_loss = src_loss_epoch / max(num_batches, 1)
        avg_gauge_loss = gauge_loss_epoch / max(num_batches, 1)

        # Evaluate on Target Domain
        tgt_metrics = evaluate(model, tgt_eval_loader, device, use_amp=use_amp)
        tgt_acc = tgt_metrics["accuracy"]
        tgt_f1 = tgt_metrics["macro_f1"]

        is_best = tgt_f1 > best_target_f1
        if is_best:
            best_target_f1 = tgt_f1

        history.append({
            "epoch": epoch,
            "total_loss": avg_total_loss,
            "src_loss": avg_src_loss,
            "gauge_loss": avg_gauge_loss,
            "target_accuracy": tgt_acc,
            "target_macro_f1": tgt_f1
        })

        # Save checkpoint dictionary containing full training state
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_target_f1": best_target_f1,
            "history": history,
            "args": vars(args)
        }

        # 1. Always save latest checkpoint (for resume)
        torch.save(checkpoint_data, last_ckpt_path)

        # 2. Save best checkpoint when target F1 improves
        if is_best:
            torch.save(checkpoint_data, best_ckpt_path)

        # 3. Always update training log file every epoch
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        best_mark = " ⭐" if is_best else ""
        print(f"{epoch:^7d} | {avg_total_loss:^10.4f} | {avg_src_loss:^8.4f} | {avg_gauge_loss:^10.4f} | {tgt_acc:^11.2f} | {tgt_f1:^11.2f}{best_mark} | {epoch_time:^5.1f}s")

    print("=" * 78)
    print(f"\n🎉 Huấn luyện hoàn tất! Best Target Macro-F1 ({args.mode.upper()}): {best_target_f1:.2f}%")
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
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Kích thước batch khi đánh giá (mặc định: 256)")
    parser.add_argument("--no_amp", action="store_true", help="Tắt Automatic Mixed Precision (AMP FP16)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint gần nhất nếu có")
    parser.add_argument("--resume_path", type=str, default=None, help="Đường dẫn trực tiếp tới file checkpoint để khôi phục")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
