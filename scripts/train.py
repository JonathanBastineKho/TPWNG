# Minimal training script for quick TPWNG validation using pre-extracted .pt features.
# This is not the full video training pipeline.
# Please ensure file paths are correctly configured before running.

#this is just a sample training script to test the tpwng.py - YU

import argparse
import logging
import random
from itertools import cycle
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import UCFCrimeDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_feature_batch(batch: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    """
    Batch features with shape safety.
    If temporal lengths differ, truncate to min length to keep (B,T,D).
    """
    features, labels = zip(*batch)
    if len(features) == 0:
        raise ValueError("Received an empty batch.")

    feature_dim = features[0].shape[-1]
    for feat in features:
        if feat.dim() != 2:
            raise ValueError(f"Each feature must be (T,D), got {tuple(feat.shape)}")
        if feat.shape[-1] != feature_dim:
            raise ValueError("Inconsistent feature dimension found in a batch.")

    min_t = min(feat.shape[0] for feat in features)
    if min_t <= 0:
        raise ValueError("Encountered an empty temporal feature sequence.")

    clipped = [feat[:min_t].float() for feat in features]
    return torch.stack(clipped, dim=0), list(labels)


def create_loader(
    root: str,
    normal: bool,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = UCFCrimeDataset(root=root, normal=normal)
    if len(dataset) == 0:
        mode = "normal" if normal else "abnormal"
        raise ValueError(f"No samples found for {mode} dataset under: {root}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_feature_batch,
    )


def resolve_data_root(root: str) -> Path:
    """
    Resolve dataset root robustly.
    Priority:
    1) As-provided path (absolute or relative to current working directory)
    2) Relative to project root (directory containing src/, scripts/, ...)
    """
    candidate = Path(root)
    if candidate.exists():
        return candidate.resolve()

    project_root = Path(__file__).resolve().parents[1]
    candidate2 = project_root / root
    if candidate2.exists():
        return candidate2.resolve()

    raise FileNotFoundError(
        f"Dataset path not found.\n"
        f"- Provided: {Path(root).resolve()}\n"
        f"- Also tried: {(project_root / root).resolve()}\n"
        f"Please set --train_root explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TPWNG training script")
    parser.add_argument("--train_root", type=str, default="ucfcrime-256/ucfcrime/train")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size_normal", type=int, default=4)
    parser.add_argument("--batch_size_abnormal", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--lambda_rank_n", type=float, default=1.0)
    parser.add_argument("--lambda_rank_a", type=float, default=1.0)
    parser.add_argument("--lambda_dil", type=float, default=1.0)
    parser.add_argument("--lambda_cl", type=float, default=1.0)
    parser.add_argument("--lambda_sp", type=float, default=1.0)
    parser.add_argument("--lambda_sm", type=float, default=1.0)

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--return_intermediates", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    train_root = resolve_data_root(args.train_root)
    logger.info("Loading datasets from: %s", train_root)
    normal_loader = create_loader(
        root=str(train_root),
        normal=True,
        batch_size=args.batch_size_normal,
        num_workers=args.num_workers,
        shuffle=True,
    )
    abnormal_loader = create_loader(
        root=str(train_root),
        normal=False,
        batch_size=args.batch_size_abnormal,
        num_workers=args.num_workers,
        shuffle=True,
    )

    logger.info("Normal videos: %d", len(normal_loader.dataset))
    logger.info("Abnormal videos: %d", len(abnormal_loader.dataset))

    steps_per_epoch = max(len(normal_loader), len(abnormal_loader))
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    logger.info("Steps per epoch: %d", steps_per_epoch)

    try:
        from src.models import TPWNG
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import TPWNG dependencies. Please run: pip install -r requirements.txt"
        ) from exc

    model = TPWNG(
        device=args.device,
        lambda_rank_n=args.lambda_rank_n,
        lambda_rank_a=args.lambda_rank_a,
        lambda_dil=args.lambda_dil,
        lambda_cl=args.lambda_cl,
        lambda_sp=args.lambda_sp,
        lambda_sm=args.lambda_sm,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Start training")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        normal_iter = cycle(normal_loader)
        abnormal_iter = cycle(abnormal_loader)

        running = {
            "loss_total": 0.0,
            "loss_rank_n": 0.0,
            "loss_rank_a": 0.0,
            "loss_dil": 0.0,
            "loss_cl": 0.0,
            "loss_sp": 0.0,
            "loss_sm": 0.0,
        }

        for step in range(1, steps_per_epoch + 1):
            normal_feats, _ = next(normal_iter)
            abnormal_feats, abnormal_labels = next(abnormal_iter)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                normal_features=normal_feats,
                abnormal_features=abnormal_feats,
                abnormal_labels=abnormal_labels,
                return_intermediates=args.return_intermediates,
            )

            loss = outputs["loss_total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}, step {step}: {loss.item()}")

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for key in running:
                running[key] += float(outputs[key].detach().item())

            if step == 1:
                logger.info(
                    "Epoch %d first-batch shapes: normal=%s abnormal=%s",
                    epoch,
                    tuple(normal_feats.shape),
                    tuple(abnormal_feats.shape),
                )

            if step % args.log_every == 0 or step == steps_per_epoch:
                denom = float(step)
                logger.info(
                    "Epoch [%d/%d] Step [%d/%d] "
                    "total=%.4f rank_n=%.4f rank_a=%.4f dil=%.4f cl=%.4f sp=%.4f sm=%.4f",
                    epoch,
                    args.epochs,
                    step,
                    steps_per_epoch,
                    running["loss_total"] / denom,
                    running["loss_rank_n"] / denom,
                    running["loss_rank_a"] / denom,
                    running["loss_dil"] / denom,
                    running["loss_cl"] / denom,
                    running["loss_sp"] / denom,
                    running["loss_sm"] / denom,
                )

        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"tpwng_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

    logger.info("Training completed.")


if __name__ == "__main__":
    main(parse_args())
