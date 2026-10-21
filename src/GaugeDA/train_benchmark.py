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

from src.GaugeDA.data.uci_har_loader import load_uci_har
from src.GaugeDA.data.hhar_loader import load_hhar
from src.GaugeDA.data.pamap2_loader import load_pamap2
from src.GaugeDA.models.decoupled_backbone import DecoupledSTLieModel
from src.GaugeDA.models.baseline_backbones import Standard1DCNN
from src.GaugeDA.losses.gauge_invariant_loss import GaugeInvariantUDALoss
from src.GaugeDA.losses.uda_baselines import (
    CORALLoss,
    MMDLoss,
    DANNLoss,
    CDANLoss,
    DomainDiscriminator
)
from src.GaugeDA.utils.selection import (
    evaluate,
    forward_model,
    is_improvement,
    selection_score,
    split_source_train_val,
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            src_ds, tgt_ds = load_hhar(
                source_model="s3", target_model="nexus4",
                window_size=128, step_size=64, target_hz=args.hhar_hz,
            )
        else:
            task_name = "nexus4_to_s3"
            src_ds, tgt_ds = load_hhar(
                source_model="nexus4", target_model="s3",
                window_size=128, step_size=64, target_hz=args.hhar_hz,
            )

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

    pin_mem = (device.type == "cuda")
    src_fs = float(src_ds.sampling_rate)
    tgt_fs = float(tgt_ds.sampling_rate)
    src_dt = 1.0 / src_fs
    tgt_dt = 1.0 / tgt_fs

    # Label-free selection: never touch target labels.
    src_train_ds, src_val_ds, split_info = split_source_train_val(
        src_ds, val_fraction=args.val_fraction, seed=args.seed
    )
    src_loader = DataLoader(src_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_mem)
    src_val_loader = DataLoader(src_val_ds, batch_size=args.eval_batch_size, shuffle=False, pin_memory=pin_mem)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_mem)
    eval_loader = DataLoader(tgt_ds, batch_size=args.eval_batch_size, shuffle=False, pin_memory=pin_mem)

    if len(src_loader) == 0:
        raise ValueError("Source training loader is empty; reduce batch_size or provide more data")

    print(
        f"📊 Selection split: train={split_info['n_train']} | val={split_info['n_val']} "
        f"({split_info['split']}) | fs_source={src_fs:g}Hz | fs_target={tgt_fs:g}Hz"
    )

    # 2. Backbone Instantiation
    d_model = args.d_model
    if args.backbone == "cnn1d":
        model = Standard1DCNN(in_channels=in_channels, num_classes=num_classes, d_model=d_model, dropout=args.dropout).to(device)
        z_dim = d_model
    elif args.backbone == "st_lie":
        model = DecoupledSTLieModel(
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=d_model,
            d_state=args.d_state,
            num_layers=args.num_layers,
            dropout=args.dropout,
            filter_window=args.filter_window,
            use_projector=not getattr(args, "no_projector", False),
            sampling_rate=src_fs,
            heading_mode=args.heading_mode,
        ).to(device)
        # Pooled features are [avg_pool ; max_pool] -> 2 * d_model.
        z_dim = 2 * d_model
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
        # z_dim, not d_model: ST-LieOp returns 2*d_model features.
        discriminator = DomainDiscriminator(in_features=z_dim, hidden_dim=128).to(device)
        criterion = DANNLoss(discriminator=discriminator, lambda_dann=args.lambda_da, lambda_ent=args.lambda_ent)
    elif args.method == "cdan":
        # CDAN uses the outer product z (x) logits -> z_dim * num_classes.
        discriminator = DomainDiscriminator(in_features=z_dim * num_classes, hidden_dim=128).to(device)
        criterion = CDANLoss(discriminator=discriminator, lambda_cdan=args.lambda_da, lambda_ent=args.lambda_ent)
    elif args.method == "proposed":
        criterion = GaugeInvariantUDALoss(
            lambda_gauge=args.lambda_da,
            lambda_ent=args.lambda_ent,
            pseudo_threshold=args.pseudo_threshold,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

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
    last_ckpt_path = f"checkpoints/benchmarks/last_{run_id}.pt"

    history = []
    best_source_score = float("-inf")
    best_source_loss = float("inf")
    best_source_acc = 0.0
    best_source_f1 = 0.0
    best_target_acc = 0.0
    best_target_f1 = 0.0

    num_steps_per_epoch = len(src_loader)
    val_criterion = nn.CrossEntropyLoss()

    print("-" * 112)
    print(
        f"{'Epoch':^7} | {'Total':^9} | {'Src':^9} | {'DA':^9} | "
        f"{'SrcVal Acc':^11} | {'SrcVal F1':^10} | {'Tgt Acc*':^10} | {'Tgt F1*':^9} | {'Time':^6}"
    )
    print("-" * 112)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if discriminator is not None:
            discriminator.train()

        tgt_iter = iter(tgt_loader) if args.method != "source_only" else None
        total_loss_sum = 0.0
        src_loss_sum = 0.0
        da_loss_sum = 0.0

        pbar = tqdm(enumerate(src_loader), total=num_steps_per_epoch, desc=f"Ep {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)

        for step, (x_s, y_s, _, _) in pbar:
            x_s, y_s = x_s.to(device, non_blocking=True), y_s.to(device, non_blocking=True)

            if tgt_iter is not None:
                try:
                    x_t, _, _, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    x_t, _, _, _ = next(tgt_iter)
                x_t = x_t.to(device, non_blocking=True)
            else:
                x_t = None

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits_s, z_s = forward_model(
                    model, x_s, delta_t=src_dt, sampling_rate=src_fs
                )

                if args.method == "source_only":
                    loss = criterion(logits_s, y_s)
                    loss_dict = {"loss_src": loss.item(), "loss_da": 0.0}
                else:
                    logits_t, z_t = forward_model(
                        model, x_t, delta_t=tgt_dt, sampling_rate=tgt_fs
                    )
                    p = float(step + (epoch - 1) * num_steps_per_epoch) / (args.epochs * num_steps_per_epoch)
                    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

                    if args.method in ["dann", "cdan"]:
                        loss, loss_dict = criterion(logits_s, y_s, z_s, z_t, logits_t, alpha=alpha)
                    else:
                        loss, loss_dict = criterion(logits_s, y_s, z_s, z_t, logits_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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

        # Source-validation selection.
        src_val_metrics = evaluate(
            model,
            src_val_loader,
            device,
            use_amp=use_amp,
            delta_t=src_dt,
            sampling_rate=src_fs,
            criterion=val_criterion,
        )
        src_val_acc = src_val_metrics["accuracy"]
        src_val_f1 = src_val_metrics["macro_f1"]
        src_val_loss = src_val_metrics.get("loss", float("nan"))

        # Target metrics are reporting-only.
        eval_metrics = evaluate(
            model,
            eval_loader,
            device,
            use_amp=use_amp,
            delta_t=tgt_dt,
            sampling_rate=tgt_fs,
        )
        tgt_acc = eval_metrics["accuracy"]
        tgt_f1 = eval_metrics["macro_f1"]

        is_best = is_improvement(args.selection_metric, src_val_metrics, best_source_score)
        if is_best:
            best_source_score = selection_score(args.selection_metric, src_val_metrics)
            best_source_loss = src_val_loss
            best_source_acc = src_val_acc
            best_source_f1 = src_val_f1
            best_target_acc = tgt_acc
            best_target_f1 = tgt_f1
            torch.save({
                "epoch": epoch,
                "protocol_version": "source_selected_v1",
                "model_state_dict": model.state_dict(),
                "selection_metric": args.selection_metric,
                "best_source_score": best_source_score,
                "best_source_loss": best_source_loss,
                "best_source_accuracy": best_source_acc,
                "best_source_f1": best_source_f1,
                "best_target_accuracy": best_target_acc,
                "best_target_macro_f1": best_target_f1,
                "best_target_acc": best_target_acc,
                "best_target_f1": best_target_f1,
                "is_target_oracle": False,
                "args": vars(args),
            }, best_ckpt_path)

        history.append({
            "epoch": epoch,
            "total_loss": avg_total_loss,
            "src_loss": avg_src_loss,
            "da_loss": avg_da_loss,
            "source_val_accuracy": src_val_acc,
            "source_val_macro_f1": src_val_f1,
            "source_val_loss": src_val_loss,
            "target_accuracy": tgt_acc,
            "target_macro_f1": tgt_f1,
            "is_best": is_best,
        })

        oracle_target_acc = max(h["target_accuracy"] for h in history)
        oracle_target_f1 = max(h["target_macro_f1"] for h in history)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "protocol_version": "source_selected_v1",
                "args": vars(args),
                "selection": {
                    "metric": args.selection_metric,
                    "split": split_info,
                    "best_source_score": best_source_score,
                    "best_source_accuracy": best_source_acc,
                    "best_source_macro_f1": best_source_f1,
                    "best_source_loss": best_source_loss,
                    "uses_target_labels": False,
                },
                # Honest, source-selected target metrics.  Keys kept for
                # compatibility with the benchmark matrix / graph ingester.
                "best_target_accuracy": best_target_acc,
                "best_target_macro_f1": best_target_f1,
                "target_oracle_accuracy": oracle_target_acc,
                "target_oracle_macro_f1": oracle_target_f1,
                "is_target_oracle": False,
                "history": history,
            }, f, indent=2)

        # Always keep the latest checkpoint for recovery/debugging.
        torch.save({
            "epoch": epoch,
            "protocol_version": "source_selected_v1",
            "model_state_dict": model.state_dict(),
            "selection_metric": args.selection_metric,
            "best_source_score": best_source_score,
            "best_target_accuracy": best_target_acc,
            "best_target_macro_f1": best_target_f1,
            "args": vars(args),
        }, last_ckpt_path)

        best_mark = " ⭐" if is_best else ""
        print(
            f"{epoch:^7d} | {avg_total_loss:^9.4f} | {avg_src_loss:^9.4f} | {avg_da_loss:^9.4f} | "
            f"{src_val_acc:^11.2f} | {src_val_f1:^10.2f} | {tgt_acc:^10.2f} | {tgt_f1:^9.2f}{best_mark} | {epoch_time:^5.1f}s"
        )

    print("-" * 112)
    print(
        f"🎉 Hoàn thành run: {run_id} | selected by {args.selection_metric} "
        f"(source score={best_source_score:.4f}) | Target at selected ckpt: "
        f"Acc={best_target_acc:.2f}% | F1={best_target_f1:.2f}%"
    )
    print("   * Target metrics are reporting-only; checkpoint selection never used target labels.")
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
    parser.add_argument("--pseudo_threshold", type=float, default=0.0, help="Confidence threshold for target pseudo-labels")
    parser.add_argument("--selection_metric", type=str, default="source_macro_f1",
                        choices=["source_macro_f1", "source_loss", "source_accuracy"],
                        help="Source-validation metric for checkpoint selection (never target labels)")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of source domains/subjects held out")
    parser.add_argument("--heading_mode", type=str, default="pca", choices=["none", "pca"])
    parser.add_argument("--filter_window", type=int, default=None)
    parser.add_argument("--no_projector", action="store_true",
                        help="Train ST-LieOp without the canonical projector (independent ablation)")
    parser.add_argument("--hhar_hz", type=float, default=100.0, help="Common resampling rate for HHAR devices")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.no_projector and args.backbone != "st_lie":
        parser.error("--no_projector is only valid with --backbone st_lie")
    train_benchmark(args)
