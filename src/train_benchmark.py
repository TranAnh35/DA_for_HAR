import os
import sys
import argparse
import json
import time
import math
from typing import Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
from src.models.baseline_backbones import Standard1DCNN
from src.losses.gauge_invariant_loss import GaugeInvariantUDALoss
from src.losses.uda_baselines import (
    CORALLoss,
    MMDLoss,
    DANNLoss,
    CDANLoss,
    DomainDiscriminator
)
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

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            for x, y, _, _ in data_loader:
                x = x.to(device, non_blocking=True)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return compute_metrics(all_targets, all_preds)

def train_benchmark(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    use_amp = (device.type == "cuda" and not args.no_amp)

    # 1. Dataset & Direction Setup
    if args.dataset == "uci_har":
        num_classes = 6
        in_channels = 6
        if args.direction == "forward":
            task_name = "sub1to20_to_sub21to30"
            src_ds, tgt_ds = load_uci_har(source_subjects=list(range(1, 21)), target_subjects=list(range(21, 31)))
        else:
            task_name = "sub21to30_to_sub1to20"
            src_ds, tgt_ds = load_uci_har(source_subjects=list(range(21, 31)), target_subjects=list(range(1, 21)))

    elif args.dataset == "hhar":
        num_classes = 6
        in_channels = 6
        if args.direction == "forward":
            task_name = "s3_to_nexus4"
            src_ds, tgt_ds = load_hhar(source_model="s3", target_model="nexus4", window_size=128, step_size=64)
        else:
            task_name = "nexus4_to_s3"
            src_ds, tgt_ds = load_hhar(source_model="nexus4", target_model="s3", window_size=128, step_size=64)

    elif args.dataset == "pamap2":
        num_classes = 12
        in_channels = 6
        if args.direction == "forward":
            task_name = "chest_to_wrist"
            src_ds, tgt_ds = load_pamap2(source_position="chest", target_position="wrist", window_size=128, step_size=64)
        else:
            task_name = "wrist_to_chest"
            src_ds, tgt_ds = load_pamap2(source_position="wrist", target_position="chest", window_size=128, step_size=64)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"\n🚀 [Benchmark Run] Dataset: {args.dataset.upper()} | Task: {task_name} | Method: {args.method.upper()} | Backbone: {args.backbone.upper()} | Seed: {args.seed}")
    print(f"📊 Data split: Source={len(src_ds)} | Target={len(tgt_ds)} | Device: {device}")

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == "cuda"))
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == "cuda"))
    eval_loader = DataLoader(tgt_ds, batch_size=args.eval_batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    # 2. Backbone Instantiation
    d_model = args.d_model
    if args.backbone == "cnn1d":
        model = Standard1DCNN(in_channels=in_channels, num_classes=num_classes, d_model=d_model, dropout=args.dropout).to(device)
    elif args.backbone == "st_lie":
        model = DecoupledSTLieModel(
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=d_model,
            d_state=args.d_state,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # 3. Loss & Discriminator Setup
    discriminator = None
    if args.method == "source_only":
        criterion = nn.CrossEntropyLoss()
    elif args.method == "coral":
        criterion = CORALLoss(lambda_coral=args.lambda_da, lambda_ent=args.lambda_ent)
    elif args.method == "mmd":
        criterion = MMDLoss(lambda_mmd=args.lambda_da, lambda_ent=args.lambda_ent)
    elif args.method == "dann":
        discriminator = DomainDiscriminator(in_features=d_model, hidden_dim=128).to(device)
        criterion = DANNLoss(discriminator=discriminator, lambda_dann=args.lambda_da, lambda_ent=args.lambda_ent)
    elif args.method == "cdan":
        discriminator = DomainDiscriminator(in_features=d_model * num_classes, hidden_dim=128).to(device)
        criterion = CDANLoss(discriminator=discriminator, lambda_cdan=args.lambda_da, lambda_ent=args.lambda_ent)
    elif args.method == "proposed":
        criterion = GaugeInvariantUDALoss(lambda_gauge=args.lambda_da, lambda_ent=args.lambda_ent)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Optimizer
    params = list(model.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs("results/benchmarks", exist_ok=True)
    os.makedirs("checkpoints/benchmarks", exist_ok=True)
    run_id = f"{args.dataset}_{task_name}_{args.method}_{args.backbone}_seed{args.seed}"
    log_file = f"results/benchmarks/{run_id}.json"
    best_ckpt_path = f"checkpoints/benchmarks/best_{run_id}.pt"

    history = []
    best_target_f1 = 0.0
    best_target_acc = 0.0

    num_steps_per_epoch = len(src_loader)

    print("-" * 80)
    print(f"{'Epoch':^7} | {'Total Loss':^10} | {'Src Loss':^10} | {'DA Loss':^10} | {'Target Acc':^11} | {'Target F1':^11} | {'Time':^6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if discriminator is not None:
            discriminator.train()

        tgt_iter = iter(tgt_loader)
        total_loss_sum = 0.0
        src_loss_sum = 0.0
        da_loss_sum = 0.0

        pbar = tqdm(enumerate(src_loader), total=num_steps_per_epoch, desc=f"Ep {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)

        for step, (x_s, y_s, _, _) in pbar:
            x_s, y_s = x_s.to(device, non_blocking=True), y_s.to(device, non_blocking=True)

            try:
                x_t, _, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_t, _, _, _ = next(tgt_iter)
            x_t = x_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits_s, z_s = model(x_s)

                if args.method == "source_only":
                    loss = criterion(logits_s, y_s)
                    loss_dict = {"loss_src": loss.item(), "loss_da": 0.0}
                else:
                    logits_t, z_t = model(x_t)
                    p = float(step + (epoch - 1) * num_steps_per_epoch) / (args.epochs * num_steps_per_epoch)
                    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

                    if args.method in ["dann", "cdan"]:
                        loss, loss_dict = criterion(logits_s, y_s, z_s, z_t, logits_t, alpha=alpha)
                    elif args.method == "proposed":
                        loss, loss_dict = criterion(logits_s, y_s, z_s, z_t, logits_t)
                    else: # coral, mmd
                        loss, loss_dict = criterion(logits_s, y_s, z_s, z_t, logits_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += loss.item()
            src_loss_sum += loss_dict.get("loss_src", loss.item())
            da_loss_sum += loss_dict.get("loss_da", 0.0)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "src": f"{loss_dict.get('loss_src', 0.0):.3f}",
                "da": f"{loss_dict.get('loss_da', 0.0):.3f}"
            })

        scheduler.step()
        epoch_time = time.time() - t0

        avg_total_loss = total_loss_sum / num_steps_per_epoch
        avg_src_loss = src_loss_sum / num_steps_per_epoch
        avg_da_loss = da_loss_sum / num_steps_per_epoch

        # Target Evaluation
        eval_metrics = evaluate(model, eval_loader, device, use_amp=use_amp)
        tgt_acc = eval_metrics["accuracy"]
        tgt_f1 = eval_metrics["macro_f1"]

        is_best = tgt_f1 > best_target_f1
        if is_best:
            best_target_f1 = tgt_f1
            best_target_acc = tgt_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_target_f1": best_target_f1,
                "best_target_acc": best_target_acc,
                "args": vars(args)
            }, best_ckpt_path)

        history.append({
            "epoch": epoch,
            "total_loss": avg_total_loss,
            "src_loss": avg_src_loss,
            "da_loss": avg_da_loss,
            "target_accuracy": tgt_acc,
            "target_macro_f1": tgt_f1
        })

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "args": vars(args),
                "best_target_accuracy": best_target_acc,
                "best_target_macro_f1": best_target_f1,
                "history": history
            }, f, indent=2)

        best_mark = " ⭐" if is_best else ""
        print(f"{epoch:^7d} | {avg_total_loss:^10.4f} | {avg_src_loss:^10.4f} | {avg_da_loss:^10.4f} | {tgt_acc:^11.2f} | {tgt_f1:^11.2f}{best_mark} | {epoch_time:^5.1f}s")

    print("-" * 80)
    print(f"🎉 Hoàn thành run: {run_id} | Best Acc: {best_target_acc:.2f}% | Best F1: {best_target_f1:.2f}%\n")
    return best_target_acc, best_target_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Benchmark Trainer for UDA on HAR")
    parser.add_argument("--method", type=str, default="proposed", choices=["source_only", "coral", "dann", "cdan", "mmd", "proposed"])
    parser.add_argument("--backbone", type=str, default="st_lie", choices=["cnn1d", "st_lie"])
    parser.add_argument("--dataset", type=str, default="uci_har", choices=["uci_har", "hhar", "pamap2"])
    parser.add_argument("--direction", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_da", type=float, default=0.5, help="Weight for UDA discrepancy / adversarial loss")
    parser.add_argument("--lambda_ent", type=float, default=0.05, help="Weight for target entropy loss")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_benchmark(args)
