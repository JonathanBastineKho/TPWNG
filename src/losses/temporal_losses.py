import torch
from torch import nn

class SparsityLoss(nn.Module):
    """
    Paper L_sp: temporal smoothing on similarity vector.
    sum_j (S_aa_j - S_aa_{j+1})^2. Apply to normalized S_aa (similarity to true
    anomaly class), not classifier output. Named 'Sparsity' in code for legacy;
    implements smoothing (adjacent frames similar).
    """
    def forward(self, similarity_or_score: torch.Tensor):
        # (B, T) -> squared diff along T
        differences = similarity_or_score[:, 1:] - similarity_or_score[:, :-1]
        return (differences**2).mean()


class SmoothnessLoss(nn.Module):
    """
    Paper L_sm: sparsity on similarity vector.
    sum_j S_aa_j. Apply to normalized S_aa. Pushes similarity magnitudes down.
    """
    def forward(self, similarity_or_score: torch.Tensor):
        return similarity_or_score.mean()