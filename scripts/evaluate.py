#!/usr/bin/env python3
"""
Evaluate TPWNG on test set: compute AUC (UCF-Crime) or AP (XD-Violence), save ROC/PR curves.
Long videos are processed in chunks (like VadCLIP) to avoid OOM; scores are concatenated.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import roc_curve, precision_recall_curve
from configs.experiments import get_experiment_config
from src.models.tpwng import TPWNG
from src.utils.experiment_logging import ExperimentTracker
from src.utils.evaluation import evaluate_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--experiment_id", type=str, default="main_ucfcrime")
    p.add_argument("--dataset", type=str, default="ucfcrime", choices=["ucfcrime", "xdviolence"])
    p.add_argument("--test_root", type=str, default=None, help="e.g. data/ucfcrime/test (default: data/<dataset>/test)")
    p.add_argument("--output_dir", type=str, default=None, help="Where to save results/plots (default: same as run dir)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=256, help="Max frames per chunk for long videos (VadCLIP uses 256); avoids OOM")
    return p.parse_args()


def main():
    args = parse_args()
    config = get_experiment_config(args.experiment_id, args.dataset)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    test_root = Path(args.test_root or f"data/{args.dataset}/test")

    model = TPWNG(config, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    result, scores_concat, labels_concat = evaluate_model(
        model, test_root, config, args.dataset, device,
        chunk_size=args.max_frames, return_arrays=True,
    )
    metric_name = list(result.keys())[0]
    metric_val = result[metric_name]

    experiment_name = f"{args.experiment_id}_{args.dataset}"
    output_dir = Path(args.output_dir or f"runs/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    tracker = ExperimentTracker(output_dir, experiment_name, config)
    tracker.save_final_results(result)
    tracker.logger.info("Result: %s", result)

    if metric_name == "AUC":
        fpr, tpr, _ = roc_curve(labels_concat, scores_concat)
        tracker.plot_roc_curve(fpr.tolist(), tpr.tolist(), metric_val)
    else:
        precision, recall, _ = precision_recall_curve(labels_concat, scores_concat)
        tracker.plot_pr_curve(precision.tolist(), recall.tolist(), metric_val)

    return result


if __name__ == "__main__":
    main()
