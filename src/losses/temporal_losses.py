import torch
from torch import nn

class SparsityLoss(nn.Module):
    """
    L_sp: sum(A_i) / T
    Anomalies should happen rarely
    """
    def __init__(self):
        super().__init__()

    def forward(self, anomaly_score: torch.Tensor):
        """
        Args:
            anomaly_scores: (B, T) Anomaly score per frame
        """
        return anomaly_score.mean()
    

class SmoothnessLoss(nn.Module):
    """
    L_sm: sum((A_i - A_{i+1})^2) / (T-1)
    Adjacent frames should have similar scores
    """
    def __init__(self):
        super().__init__()

    def forward(self, anomaly_score: torch.Tensor):
        """
        Args:
            anomaly_scores: (B, T) Anomaly score per frame
        """
        differences = anomaly_score[:, 1:] - anomaly_score[:, :-1]
        return (differences**2).mean()