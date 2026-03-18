"""
Evaluation helpers: chunked inference and metric computation.
Used by scripts/evaluate.py (standalone) and scripts/train.py (periodic eval).
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.experiments import DATASET_CONFIG
from src.data.test_dataset import WSVADTestDataset

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = average_precision_score = None


def run_inference_chunked(model, feat: torch.Tensor, chunk_size: int, device: torch.device) -> np.ndarray:
    """
    Run forward_inference on long videos in fixed-size chunks (VadCLIP-style) to avoid OOM.
    feat: (T, D) or (1, T, D). Returns (T,) scores.
    """
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    T = feat.shape[1]
    if T <= chunk_size:
        feat = feat.to(device)
        scores = model.forward_inference(feat).cpu().squeeze(0).detach().numpy()
        return scores[:T]
    chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = feat[:, start:end]
        if chunk.shape[1] < chunk_size:
            pad = torch.zeros(1, chunk_size - chunk.shape[1], feat.shape[2], dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=1)
        chunks.append(chunk)
    chunk_tensor = torch.cat(chunks, dim=0).to(device)
    with torch.no_grad():
        out = model.forward_inference(chunk_tensor).cpu().detach().numpy()
    scores_list = []
    for i, start in enumerate(range(0, T, chunk_size)):
        end = min(start + chunk_size, T)
        n = end - start
        scores_list.append(out[i, :n])
    return np.concatenate(scores_list, axis=0)


def evaluate_model(
    model: torch.nn.Module,
    test_root: Path,
    config: dict,
    dataset: str,
    device: torch.device,
    chunk_size: int = 256,
    return_arrays: bool = False,
):
    """
    Run evaluation on the test set.
    Returns {"AUC": float} or {"AP": float}. If return_arrays=True, returns
    (result, scores_concat, labels_concat) for plotting.
    """
    if roc_auc_score is None:
        raise ImportError("sklearn is required. pip install scikit-learn")
    model.eval()
    test_root = Path(test_root)
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    gt_npy_path = test_root / "gt.npy"
    if gt_npy_path.exists():
        pt_paths = sorted(test_root.rglob("*.pt"))
        if not pt_paths:
            raise RuntimeError(f"No .pt files under {test_root}")
        labels_concat = np.load(gt_npy_path).ravel()
        all_scores = []
        for pt_path in pt_paths:
            feat = torch.load(pt_path, map_location="cpu", weights_only=True)
            scores = run_inference_chunked(model, feat, chunk_size, device)
            all_scores.append(scores)
        scores_concat = np.concatenate(all_scores, axis=0)
        n = min(len(scores_concat), len(labels_concat))
        scores_concat = scores_concat[:n]
        labels_concat = labels_concat[:n]
    else:
        test_ds = WSVADTestDataset(str(test_root))
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        all_scores, all_labels = [], []
        for feat, labels in loader:
            feat = feat.squeeze(0)
            scores = run_inference_chunked(model, feat, chunk_size, device)
            labels = labels.squeeze(0).numpy()
            F = min(len(scores), len(labels))
            all_scores.append(scores[:F])
            all_labels.append(labels[:F])
        scores_concat = np.concatenate(all_scores, axis=0)
        labels_concat = np.concatenate(all_labels, axis=0)

    metric_name = DATASET_CONFIG[dataset]["metric"]
    if metric_name == "AUC":
        result = {"AUC": float(roc_auc_score(labels_concat, scores_concat))}
    else:
        result = {"AP": float(average_precision_score(labels_concat, scores_concat))}
    if return_arrays:
        return result, scores_concat, labels_concat
    return result
