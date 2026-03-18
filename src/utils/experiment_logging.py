"""
Experiment logging: metrics, checkpoints, loss curves, and result tables.
Saves to runs/<experiment_name>/ with config, logs, plots, and metrics.
Uses TensorBoard for live charts (tensorboard --logdir runs).
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    _HAS_TENSORBOARD = False

# Optional matplotlib for local plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class UnbufferedFileHandler(logging.FileHandler):
    """FileHandler that flushes after each emit so tail -f shows logs in real time."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """
    Create logger that writes to console (stdout) and to log_dir/train.log.
    File is flushed after every line so you can view continuously with:
      tail -f runs/<experiment_name>/train.log
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    fh = UnbufferedFileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)

    return logger


class ExperimentTracker:
    """
    Tracks metrics per epoch, saves config, checkpoints, and plots.
    Output dir: runs/<experiment_name>/
      - config.json
      - train.log
      - metrics.json (per-epoch metrics)
      - checkpoints/checkpoint_epoch{N}.pt
      - plots/loss_curves.png, roc_curve.png, etc.
    """

    def __init__(self, run_dir: Path, experiment_name: str, config: dict, use_tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.experiment_name = experiment_name
        self.config = config
        self.use_tensorboard = use_tensorboard and _HAS_TENSORBOARD
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

        self.logger = setup_logging(self.run_dir, experiment_name)
        self.logger.info("Experiment: %s", experiment_name)
        self._save_config()
        self._writer = None
        if self.use_tensorboard:
            self._writer = SummaryWriter(log_dir=str(self.run_dir))
            # Log config as text so parameters appear in TensorBoard
            try:
                cfg_text = json.dumps(self.config, indent=2, default=str)
                self._writer.add_text("experiment/config", cfg_text, global_step=0)
            except Exception:
                pass
            self.logger.info("TensorBoard: view live charts with  tensorboard --logdir %s", self.run_dir.parent)

        self.history: List[Dict[str, float]] = []

    def _save_config(self) -> None:
        """Save config as JSON (serializable only)."""
        cfg = {k: v for k, v in self.config.items() if _json_serializable(v)}
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Append epoch metrics to history and metrics.json."""
        record = {"epoch": epoch, **metrics}
        self.history.append(record)
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
        if self._writer is not None:
            # Log total loss and each term under train/ so TensorBoard shows one group "train" with all curves
            for k, v in metrics.items():
                self._writer.add_scalar(f"train/{k}", v, global_step=epoch)
            self._writer.flush()
        msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info("Epoch %d | %s", epoch, msg)

    def log_eval_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log validation/test metrics (e.g. AUC, AP) to TensorBoard and log file."""
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(f"eval/{k}", v, global_step=epoch)
            self._writer.flush()
        msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info("Eval epoch %d | %s", epoch, msg)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[Dict[str, float]] = None,
        suffix: Optional[str] = None,
    ) -> Path:
        """Save checkpoint to runs/<exp>/checkpoints/checkpoint_epoch{N}.pt or checkpoint_{suffix}.pt."""
        if suffix:
            path = self.run_dir / "checkpoints" / f"checkpoint_{suffix}.pt"
        else:
            path = self.run_dir / "checkpoints" / f"checkpoint_epoch{epoch}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if metrics:
            state["metrics"] = metrics
        torch.save(state, path)
        self.logger.info("Saved checkpoint: %s", path)
        return path

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def save_final_results(self, results: Dict[str, Any]) -> None:
        """Save final eval results (e.g. AUC, AP) to results.json."""
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info("Final results: %s", results)

    def plot_loss_curves(self, metrics_keys: Optional[List[str]] = None) -> None:
        """Plot loss/metric curves from history and save to plots/loss_curves.png."""
        if not _HAS_MPL or not self.history:
            return
        keys = metrics_keys or [k for k in self.history[0] if k != "epoch"]
        keys = [k for k in keys if k in self.history[0]]
        if not keys:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = [h["epoch"] for h in self.history]
        for k in keys:
            ax.plot(epochs, [h[k] for h in self.history], label=k)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.run_dir / "plots" / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved loss curves to plots/loss_curves.png")

    def plot_roc_curve(self, fpr: List[float], tpr: List[float], auc: float) -> None:
        """Save ROC curve to plots/roc_curve.png."""
        if not _HAS_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.run_dir / "plots" / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_pr_curve(self, precision: List[float], recall: List[float], ap: float) -> None:
        """Save PR curve to plots/pr_curve.png."""
        if not _HAS_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.run_dir / "plots" / "pr_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _json_serializable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False
