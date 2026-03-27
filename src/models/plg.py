import torch
from torch import nn

class PseudoLabelGenerator(nn.Module):
    def __init__(self,
                 alpha: float = 0.2,
                 threshold: float = 0.55):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def normalize(self, x, dim=-1):
        """Min-max normalization"""
        return (x - x.min(dim=dim, keepdim=True)[0]) / (x.max(dim=dim, keepdim=True)[0] - x.min(dim=dim, keepdim=True)[0] + 1e-8)

    def forward(self, 
                S_aa: torch.Tensor,
                S_an: torch.Tensor):
        """
        Args:
            S_aa: (F,) For anomaly videos: similarity to TRUE class
                        For normal videos: MAX similarity to all anomaly classes
            S_an: (F,) Similarity to NVP-enhanced normal text
        """
        # Eq (5)
        # psi = alpha * ~S_an + (1-alpha) * (1 - ~S_aa)
        S_aa_norm = self.normalize(S_aa)
        S_an_norm = self.normalize(S_an)
        psi = self.alpha * S_an_norm + (1 - self.alpha) * (1 - S_aa_norm)

        psi_norm = self.normalize(psi)

        # Eq (6) Threshold low psi = anomaly
        return (psi_norm >= self.threshold).long()


