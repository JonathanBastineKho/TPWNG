#!/usr/bin/env python3
"""
Train TPWNG for WSVAD. Replicates paper experiments via --experiment_id and --dataset.
Logs, checkpoints, and metrics go to runs/<experiment_name>/.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from itertools import cycle

import torch
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.experiments import get_experiment_config, DATASET_CONFIG
from src.data.dataset import UCFCrimeDataset
from src.data.collate import collate_normal, collate_anomaly
from src.models.tpwng import TPWNG
from src.losses import NormalRankingLoss, AnomalyRankingLoss, DistributionalInconsistencyLoss, SparsityLoss, SmoothnessLoss
from src.utils.experiment_logging import ExperimentTracker
from src.utils.evaluation import evaluate_model


def parse_args():
    p = argparse.ArgumentParser(description="Train TPWNG (WSVAD)")
    p.add_argument("--experiment_id", type=str, default="main_ucfcrime", help="e.g. main_ucfcrime, abl_nvp_none")
    p.add_argument("--dataset", type=str, default="ucfcrime", choices=["ucfcrime", "xdviolence"])
    p.add_argument("--data_root", type=str, default="data/ucfcrime", help="Root containing train/ with class folders")
    p.add_argument("--max_frames", type=int, default=256, help="Max frames per video (VadCLIP uses 256); pad/sample to this")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--batch_size_normal", type=int, default=None)
    p.add_argument("--batch_size_anomaly", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="runs", help="Base dir for runs/<experiment_name>")
    p.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard (enabled by default for live charts)")
    p.add_argument("--eval-every", type=int, default=100, metavar="N", help="Run test-set evaluation every N epochs (0 = disable; eval logged to TensorBoard under eval/)")
    p.add_argument("--test-root", type=str, default=None, help="Test data path (default: <data_root>/test)")
    p.add_argument("--train-sample", type=str, default="uniform_avg", choices=["truncate", "random", "uniform", "uniform_avg"],
                   help="VadCLIP-style: uniform_avg (default), random; or truncate, uniform")
    p.add_argument("--resume", type=str, default=None, metavar="PATH", help="Resume from checkpoint (e.g. runs/.../checkpoints/checkpoint_epoch10.pt)")
    p.add_argument("--extra_epochs", type=int, default=None, metavar="N", help="When resuming: train N more epochs (overrides --epochs; e.g. --extra_epochs 300)")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (0 = main thread only; 2–4 can speed up data loading)")
    p.add_argument("--run_suffix", type=str, default=None, help="Suffix for run dir (e.g. timestamp) so TensorBoard/logs get unique names")
    return p.parse_args()


def main():
    args = parse_args()
    config = get_experiment_config(args.experiment_id, args.dataset)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size_normal is not None:
        config["batch_size_normal"] = args.batch_size_normal
    if args.batch_size_anomaly is not None:
        config["batch_size_anomaly"] = args.batch_size_anomaly

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_root = Path(args.data_root) / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")

    experiment_name = f"{args.experiment_id}_{args.dataset}"
    if args.run_suffix:
        experiment_name = f"{experiment_name}_{args.run_suffix}"
    run_dir = Path(args.output_dir) / experiment_name
    tracker = ExperimentTracker(run_dir, experiment_name, config, use_tensorboard=not args.no_tensorboard)
    log_path = run_dir / "train.log"
    tracker.logger.info("Using device: %s", device)
    tracker.logger.info(
        "Log file: %s — To follow in another terminal: tail -f %s",
        log_path,
        log_path,
    )

    # Class index for anomaly loader (anomaly class name -> 0..K-2)
    anomaly_classes = config["anomaly_classes"]
    class_to_idx = {c: i for i, c in enumerate(anomaly_classes)}
    max_frames = args.max_frames or config.get("max_frames", 320)

    normal_ds = UCFCrimeDataset(str(train_root), normal=True)
    anomaly_ds = UCFCrimeDataset(str(train_root), normal=False)
    if len(normal_ds) == 0 or len(anomaly_ds) == 0:
        tracker.logger.warning("Empty normal or anomaly set; check data_root and folder layout.")

    sample_mode = args.train_sample
    num_workers = args.num_workers
    normal_loader = DataLoader(
        normal_ds,
        batch_size=config["batch_size_normal"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_normal(max_frames, sample_mode=sample_mode),
        pin_memory=(device.type == "cuda"),
    )
    anomaly_loader = DataLoader(
        anomaly_ds,
        batch_size=config["batch_size_anomaly"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_anomaly(max_frames, class_to_idx, sample_mode=sample_mode),
        pin_memory=(device.type == "cuda"),
    )

    model = TPWNG(config, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.get("scheduler_milestones", [30, 40]),
        gamma=config.get("scheduler_gamma", 0.1),
    )
    best_eval_metric = -1.0
    start_epoch = 1

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if "metrics" in ckpt and config.get("metric"):
            best_eval_metric = float(ckpt["metrics"].get(f"best_{config['metric']}", best_eval_metric))
        # Recreate scheduler so LR schedule aligns with global epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get("scheduler_milestones", [30, 40]),
            gamma=config.get("scheduler_gamma", 0.1),
            last_epoch=start_epoch - 1,
        )
        tracker.logger.info("Resumed from %s (epoch %d → start epoch %d)", ckpt_path, start_epoch - 1, start_epoch)
        if args.extra_epochs is not None:
            num_epochs = start_epoch + args.extra_epochs - 1
            config["epochs"] = num_epochs
            tracker.logger.info("Training %d more epochs (until epoch %d)", args.extra_epochs, num_epochs)

    loss_fn_n_rank = NormalRankingLoss()
    loss_fn_a_rank = AnomalyRankingLoss()
    loss_fn_dil = DistributionalInconsistencyLoss()
    loss_fn_sp = SparsityLoss()
    loss_fn_sm = SmoothnessLoss()

    num_epochs = config["epochs"]
    training_start = time.time()
    tracker.logger.info("Starting training for %d epochs (train_sample=%s)%s", num_epochs, sample_mode, f", resuming from epoch {start_epoch}" if start_epoch > 1 else "")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        losses_breakdown = {}
        num_steps = 0

        # Iterate over both loaders; cycle the shorter one
        if len(normal_loader) >= len(anomaly_loader):
            loader_pair = zip(cycle(anomaly_loader), normal_loader)
            steps = len(normal_loader)
        else:
            loader_pair = zip(anomaly_loader, cycle(normal_loader))
            steps = len(anomaly_loader)

        for (anomaly_feats, anomaly_idx), normal_feats in loader_pair:
            normal_feats = normal_feats.to(device)
            anomaly_feats = anomaly_feats.to(device)
            anomaly_idx = anomaly_idx.to(device)

            out = model(normal_feats, anomaly_feats, anomaly_idx)
            S_nn = out["S_nn"]
            phi_na = out["phi_na"]
            S_an = out["S_an"]
            S_aa_true = out["S_aa_true"]
            phi_aa_other = out["phi_aa_other"]
            S_aa_norm = out["S_aa_norm"]
            pseudo_labels = out["pseudo_labels"].float()
            scores = out["scores"]

            loss = torch.tensor(0.0, device=device)
            if config.get("use_Ln_rank", True):
                Ln = loss_fn_n_rank(S_nn, phi_na)
                loss = loss + Ln
                losses_breakdown["L_n_rank"] = losses_breakdown.get("L_n_rank", 0.0) + Ln.item()
            if config.get("use_La_rank", True):
                La = loss_fn_a_rank(S_aa_true, phi_aa_other, S_an)
                loss = loss + La
                losses_breakdown["L_a_rank"] = losses_breakdown.get("L_a_rank", 0.0) + La.item()
            if config.get("use_Ldil", True):
                Ldil = loss_fn_dil(S_aa_true, S_an)
                loss = loss + Ldil
                losses_breakdown["L_dil"] = losses_breakdown.get("L_dil", 0.0) + Ldil.item()

            L_sp = loss_fn_sp(S_aa_norm)
            L_sm = loss_fn_sm(S_aa_norm)
            loss = loss + config["lambda1"] * L_sp + config["lambda2"] * L_sm
            losses_breakdown["L_sp"] = losses_breakdown.get("L_sp", 0.0) + L_sp.item()
            losses_breakdown["L_sm"] = losses_breakdown.get("L_sm", 0.0) + L_sm.item()

            L_cl = torch.nn.functional.binary_cross_entropy(scores, pseudo_labels)
            loss = loss + L_cl
            losses_breakdown["L_cl"] = losses_breakdown.get("L_cl", 0.0) + L_cl.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        avg_breakdown = {k: v / max(num_steps, 1) for k, v in losses_breakdown.items()}
        metrics = {"loss": avg_loss, **avg_breakdown}
        tracker.log_metrics(epoch, metrics)

        if epoch % args.save_every == 0 or epoch == num_epochs:
            tracker.save_checkpoint(epoch, model, optimizer, metrics)

        # Optional: periodic evaluation on test set (AUC/AP); save best checkpoint
        if args.eval_every and epoch % args.eval_every == 0:
            test_root = Path(args.test_root or str(Path(args.data_root) / "test"))
            if test_root.exists():
                try:
                    tracker.logger.info("  Running evaluation on test set (may take a few minutes)...")
                    eval_result = evaluate_model(
                        model, test_root, config, args.dataset, device,
                        chunk_size=args.max_frames, return_arrays=False,
                    )
                    tracker.log_eval_metrics(epoch, eval_result)
                    metric_name = config.get("metric", "AUC")
                    metric_val = eval_result.get(metric_name, -1.0)
                    if metric_val > best_eval_metric:
                        best_eval_metric = metric_val
                        tracker.save_checkpoint(epoch, model, optimizer, {**metrics, f"best_{metric_name}": metric_val}, suffix="best")
                        tracker.logger.info("  → New best %s=%.4f, saved checkpoint_best.pt", metric_name, metric_val)
                except Exception as e:
                    tracker.logger.warning("Eval failed: %s", e)
            else:
                tracker.logger.warning("test_root not found, skipping eval: %s", test_root)

        scheduler.step()

        # Time estimation: elapsed so far, avg per epoch, remaining, ETA
        elapsed = time.time() - training_start
        avg_sec_per_epoch = elapsed / epoch
        remaining_epochs = num_epochs - epoch
        eta_sec = avg_sec_per_epoch * remaining_epochs
        if eta_sec >= 3600:
            time_left_str = "~%dh %dm left" % (int(eta_sec // 3600), int((eta_sec % 3600) // 60))
        elif eta_sec >= 60:
            time_left_str = "~%d min left" % int(eta_sec // 60)
        else:
            time_left_str = "~%ds left" % int(eta_sec)
        eta_dt = datetime.now() + timedelta(seconds=eta_sec)
        eta_str = eta_dt.strftime("%Y-%m-%d %H:%M")
        tracker.logger.info("  → %s | ETA %s", time_left_str, eta_str)

    tracker.plot_loss_curves()
    tracker.close()
    tracker.logger.info("Training finished. Run evaluate.py for AUC/AP.")


if __name__ == "__main__":
    main()
