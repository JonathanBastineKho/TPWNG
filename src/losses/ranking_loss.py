import torch
from torch import nn

class NormalRankingLoss(nn.Module):
    """
    L_rank(n): max(0, 1 - max(S_nn) + max(max(phi_na))) (Eq 10)
    Normal text should match better than any anomaly text
    """
    def __init__(self):
        super().__init__()

    def forward(self, S_nn: torch.Tensor, phi_na: torch.Tensor):
        """
        Args:
            S_nn: (B, T) Similarity of normal frames to normal text
            phi_na: (B, T, K) Similarity of normal frames to K anomaly texts
        """
        max_S_nn = S_nn.max(dim=1)[0]
        max_phi_na = phi_na.max(dim=2)[0].max(dim=1)[0]
        return torch.clamp(1 - max_S_nn + max_phi_na, min=0).mean()
    

class AnomalyRankingLoss(nn.Module):
    """
    L_rank(a): (Eq 11)
    Term 1: max(0, 1 - max(S_aa_true) + max(max(phi_aa_other)))
    Term 2: max(0, 1 - max(S_an) + max(max(phi_aa_other)))
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                S_aa_true: torch.Tensor,
                phi_aa_other: torch.Tensor,
                S_an: torch.Tensor):
        """
        Args:
            S_aa_true: (B, T) Similarity to true anomaly text
            phi_aa_other: (B, T, K-1) Similarities to other anomaly texts
            S_an: (B, T) Similarity to normal text (NVP-enhanced)
        """
        
        # Term 1: True class should score higher than any other class
        max_S_aa_true = S_aa_true.max(dim=1)[0]
        max_phi_aa_other = phi_aa_other.max(dim=2)[0].max(dim=1)[0]
        loss1 = torch.clamp(1.0 - max_S_aa_true + max_phi_aa_other, min=0.0)

        # Term 2: Normal text should rank high (for normal frames)
        max_S_an = S_an.max(dim=1)[0]
        loss2 = torch.clamp(1.0 - max_S_an + max_phi_aa_other, min=0.0)

        return (loss1 + loss2).mean()