import torch
from torch import nn

class SparsityLoss(nn.Module):
    """
    L_sp: sum((A_i - A_{i+1})^2) / (T-1)
    Anomalies should happen rarely
    """
    def forward(self, anomaly_score: torch.Tensor):
        differences = anomaly_score[:, 1:] - anomaly_score[:, :-1]
        return (differences**2).mean()


class SmoothnessLoss(nn.Module):
    """
    L_sm: sum(A_i) / T
    Adjacent frames should have similar scores
    """
    def forward(self, anomaly_score: torch.Tensor):
        return anomaly_score.mean()