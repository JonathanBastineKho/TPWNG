#!/usr/bin/env python3
"""
Run multiple TPWNG experiments (train + evaluate) and collect results into a summary table.
Use --experiments to run a subset, or --all to run all paper experiments.
Outputs: runs/<exp_name>/ (per run) and runs/summary_<timestamp>.json + .md
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.experiments import PAPER_EXPERIMENTS, get_experiment_config, DATASET_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Run TPWNG experiments (train then eval)")
    p.add_argument("--experiments", nargs="+", default=None, help="e.g. main_ucfcrime abl_nvp_none (default: main_ucfcrime)")
    p.add_argument("--all", action="store_true", help="Run all PAPER_EXPERIMENTS")
    p.add_argument("--train_only", action="store_true", help="Skip evaluation")
    p.add_argument("--eval_only", action="store_true", help="Only run evaluation (requires existing checkpoints)")
    p.add_argument("--data_root", type=str, default="data/ucfcrime")
    p.add_argument("--test_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="runs")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def run_cmd(cmd: list, cwd: Path) -> bool:
    ret = subprocess.run(cmd, cwd=cwd)
    return ret.returncode == 0


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        runs = list(PAPER_EXPERIMENTS)
    elif args.experiments:
        runs = []
        for e in args.experiments:
            dataset = "xdviolence" if e.endswith("_xdviolence") else "ucfcrime"
            runs.append((e, dataset))
    else:
        runs = [("main_ucfcrime", "ucfcrime")]

    results = []
    for experiment_id, dataset in runs:
        exp_name = f"{experiment_id}_{dataset}"
        data_root = args.data_root if "ucf" in dataset or dataset == "ucfcrime" else f"data/{dataset}"
        test_root = args.test_root or f"data/{dataset}/test"
        config = get_experiment_config(experiment_id, dataset)

        if args.eval_only:
            # Eval-only: use latest existing timestamped run for this experiment
            run_dirs = sorted(output_dir.glob(f"{exp_name}_*"), key=lambda p: p.name, reverse=True)
            run_path = run_dirs[0] if run_dirs else output_dir / exp_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_path = output_dir / f"{exp_name}_{ts}"
            # Log this experiment's parameters to run directory (unique per run via timestamp)
            params_log = {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "exp_name": exp_name,
                "timestamp": ts,
                "data_root": data_root,
                "test_root": test_root,
                "epochs": args.epochs if args.epochs is not None else config.get("epochs"),
                "device": args.device,
                "config": config,
            }
            try:
                run_path.mkdir(parents=True, exist_ok=True)
                with open(run_path / "experiment_params.json", "w") as f:
                    json.dump(params_log, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: could not write experiment_params.json: {e}")

        if not args.eval_only:
            cmd = [
                sys.executable, "scripts/train.py",
                "--experiment_id", experiment_id,
                "--dataset", dataset,
                "--data_root", data_root,
                "--output_dir", str(output_dir),
                "--run_suffix", ts,
            ]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.device:
                cmd += ["--device", args.device]
            ok = run_cmd(cmd, root)
            if not ok:
                results.append({"experiment": exp_name, "run_path": str(run_path), "train_ok": False, "metric": None})
                continue

        if args.train_only:
            results.append({"experiment": exp_name, "run_path": str(run_path), "train_ok": True, "metric": None})
            continue

        # Find latest checkpoint (run_path is timestamped)
        ckpt_dir = run_path / "checkpoints"
        if not ckpt_dir.exists():
            results.append({"experiment": exp_name, "run_path": str(run_path), "train_ok": True, "eval_ok": False, "metric": None})
            continue
        checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"), key=lambda p: int(p.stem.split("epoch")[1]))
        if not checkpoints:
            results.append({"experiment": exp_name, "run_path": str(run_path), "eval_ok": False, "metric": None})
            continue
        last_ckpt = str(checkpoints[-1])

        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--checkpoint", last_ckpt,
            "--experiment_id", experiment_id,
            "--dataset", dataset,
            "--test_root", test_root,
            "--output_dir", str(run_path),
        ]
        if args.device:
            cmd += ["--device", args.device]
        ok = run_cmd(cmd, root)
        res_file = run_path / "results.json"
        metric_val = None
        if res_file.exists():
            with open(res_file) as f:
                data = json.load(f)
                metric_val = data.get("AUC") or data.get("AP")
        results.append({
            "experiment": exp_name,
            "run_path": str(run_path),
            "eval_ok": ok,
            "metric_name": DATASET_CONFIG.get(dataset, {}).get("metric", "AUC"),
            "metric": metric_val,
        })

    # Summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    md_path = output_dir / f"summary_{ts}.md"
    with open(md_path, "w") as f:
        f.write("| Experiment | Metric | Value |\n")
        f.write("|------------|--------|-------|\n")
        for r in results:
            val = r.get("metric")
            val_str = f"{val:.4f}" if val is not None else "-"
            f.write(f"| {r['experiment']} | {r.get('metric_name', 'AUC/AP')} | {val_str} |\n")
    print(f"Summary: {summary_path}")
    print(f"Table:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
