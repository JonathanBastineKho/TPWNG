#!/usr/bin/env python3
"""
Train TPWNG on pre-extracted UCF-Crime CLIP features.
"""
import argparse
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

# Allow running from repository root with "python scripts/train.py"
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_anomaly, collate_normal
from src.data.dataset import UCFCrimeDataset
from src.models.tpwng import TPWNG


DEFAULT_CLASSES = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
    "Normal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TPWNG (CVPR 2024)")
    parser.add_argument("--experiment_id", type=str, default="main_ucfcrime")
    parser.add_argument("--dataset", type=str, default="ucfcrime")
    parser.add_argument("--data_root", type=str, default="ucfcrime")
    parser.add_argument("--run_suffix", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size_normal", type=int, default=32)
    parser.add_argument("--batch_size_anomaly", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--train_sample", type=str, default="uniform_avg",
                        choices=["uniform_avg", "random", "uniform", "truncate"])

    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_ctx", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.55)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--R", type=float, default=256.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--scheduler_milestones", type=int, nargs="+", default=[30, 40])
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("tpwng_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def maybe_plot_curves(metrics: List[Dict[str, float]], out_path: Path, logger: logging.Logger) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.info("matplotlib not found; skipping loss curve plot.")
        return

    epochs = [m["epoch"] for m in metrics]
    keys = ["loss", "L_n_rank", "L_a_rank", "L_dil", "L_sp", "L_sm", "L_cl"]
    plt.figure(figsize=(10, 6))
    for k in keys:
        plt.plot(epochs, [m[k] for m in metrics], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("TPWNG Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved loss curves to %s", out_path.as_posix())


def build_run_name(args: argparse.Namespace) -> str:
    pieces = [args.experiment_id, args.dataset]
    if args.run_suffix:
        pieces.append(args.run_suffix)
    return "_".join(pieces)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = build_run_name(args)
    run_dir = Path("runs") / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("Experiment: %s", run_name)
    logger.info("TensorBoard: view live charts with  tensorboard --logdir runs")
    logger.info("Using device: %s", device)
    logger.info("Log file: %s — To follow in another terminal: tail -f %s",
                (run_dir / "train.log").as_posix(), (run_dir / "train.log").as_posix())

    classes = DEFAULT_CLASSES
    normal_name = classes[-1]
    anomaly_classes = classes[:-1]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    train_root = Path(args.data_root) / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Training root not found: {train_root}")

    ds_normal = UCFCrimeDataset(str(train_root), normal=True)
    ds_anomaly = UCFCrimeDataset(str(train_root), normal=False)
    if len(ds_normal) == 0 or len(ds_anomaly) == 0:
        raise RuntimeError(
            f"Empty dataset split. normal={len(ds_normal)}, anomaly={len(ds_anomaly)} under {train_root}"
        )

    dl_normal = DataLoader(
        ds_normal,
        batch_size=args.batch_size_normal,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_normal(args.max_frames, args.train_sample),
    )
    dl_anomaly = DataLoader(
        ds_anomaly,
        batch_size=args.batch_size_anomaly,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_anomaly(args.max_frames, class_to_idx, args.train_sample),
    )

    model = TPWNG(
        class_names=classes,
        model_name="ViT-B/16",
        ctx_length=args.n_ctx,
        feature_dim=args.feature_dim,
        alpha=args.alpha,
        threshold=args.theta,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=str(device),
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

    # Keep config contract close to prior runs.
    config = {
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "alpha": args.alpha,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "n_ctx": args.n_ctx,
        "R": args.R,
        "feature_dim": args.feature_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size_normal": args.batch_size_normal,
        "batch_size_anomaly": args.batch_size_anomaly,
        "max_frames": args.max_frames,
        "scheduler_milestones": args.scheduler_milestones,
        "scheduler_gamma": args.scheduler_gamma,
        "theta": args.theta,
        "classes": classes,
        "anomaly_classes": anomaly_classes,
        "normal_name": normal_name,
        "metric": "AUC",
        "use_nvp": True,
        "nvp_mode": "as",
        "use_normality_guidance": True,
        "temporal_module": "tcsal",
        "use_Ln_rank": True,
        "use_La_rank": True,
        "use_Ldil": True,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Starting training for %d epochs (train_sample=%s)", args.epochs, args.train_sample)
    metrics: List[Dict[str, float]] = []

    # Iterate both loaders for max length by cycling the shorter one.
    n_steps = max(len(dl_normal), len(dl_anomaly))
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        normal_iter = iter(dl_normal)
        anomaly_iter = iter(dl_anomaly)

        running = {
            "loss": 0.0,
            "L_n_rank": 0.0,
            "L_a_rank": 0.0,
            "L_dil": 0.0,
            "L_sp": 0.0,
            "L_sm": 0.0,
            "L_cl": 0.0,
        }

        for _ in range(n_steps):
            try:
                x_normal = next(normal_iter)
            except StopIteration:
                normal_iter = iter(dl_normal)
                x_normal = next(normal_iter)
            try:
                x_abnormal, tau = next(anomaly_iter)
            except StopIteration:
                anomaly_iter = iter(dl_anomaly)
                x_abnormal, tau = next(anomaly_iter)

            x_normal = x_normal.to(device, non_blocking=True)
            x_abnormal = x_abnormal.to(device, non_blocking=True)
            tau = tau.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x_normal, x_abnormal, tau)
            out["total_loss"].backward()
            optimizer.step()

            running["loss"] += out["total_loss"].item()
            running["L_n_rank"] += out["L_n_rank"].item()
            running["L_a_rank"] += out["L_a_rank"].item()
            running["L_dil"] += out["L_dil"].item()
            running["L_sp"] += out["L_sp"].item()
            running["L_sm"] += out["L_sm"].item()
            running["L_cl"] += out["L_cl"].item()

        scheduler.step()
        for k in running:
            running[k] /= float(n_steps)

        metrics.append({
            "epoch": epoch,
            "loss": running["loss"],
            "L_n_rank": running["L_n_rank"],
            "L_a_rank": running["L_a_rank"],
            "L_dil": running["L_dil"],
            "L_sp": running["L_sp"],
            "L_sm": running["L_sm"],
            "L_cl": running["L_cl"],
        })
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if epoch % args.log_every == 0:
            logger.info(
                "Epoch %d | loss=%.4f L_n_rank=%.4f L_a_rank=%.4f L_dil=%.4f L_sp=%.4f L_sm=%.4f L_cl=%.4f",
                epoch, running["loss"], running["L_n_rank"], running["L_a_rank"],
                running["L_dil"], running["L_sp"], running["L_sm"], running["L_cl"]
            )
            elapsed_epoch = time.time() - epoch_start
            remain = elapsed_epoch * (args.epochs - epoch)
            eta = datetime.fromtimestamp(time.time() + remain).strftime("%Y-%m-%d %H:%M")
            logger.info("  -> ~%s left | ETA %s", f"{int(math.ceil(remain))}s" if remain < 120 else f"{int(round(remain / 60.0))} min", eta)

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path.as_posix())

    maybe_plot_curves(metrics, Path("plots") / "loss_curves.png", logger)
    logger.info("Training finished. Run evaluate.py for AUC/AP.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
