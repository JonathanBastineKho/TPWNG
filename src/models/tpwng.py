import torch
import torch.nn.functional as F
from torch import nn

from .clip import TextPromptEncoder
from .nvp import NormalityVisualPrompt
from .plg import PseudoLabelGenerator
from .tcsal import TCSAL
from ..losses.ranking_loss import NormalRankingLoss, AnomalyRankingLoss
from ..losses.temporal_losses import SparsityLoss, SmoothnessLoss
from ..losses.dil import DistributionalInconsistencyLoss


UCF_CRIME_CLASSES = [
    'abuse', 'arrest', 'arson', 'assault', 'roadaccidents',
    'burglary', 'explosion', 'fighting', 'robbery', 'shooting',
    'stealing', 'shoplifting', 'vandalism', 'normal'
]


def minmax_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Min-max normalization along a dimension (the tilde ~ in the paper)."""
    x_min = x.min(dim=dim, keepdim=True)[0]
    x_max = x.max(dim=dim, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + 1e-8)


class TPWNG(nn.Module):
    """
    Text Prompt with Normality Guidance (CVPR 2024).

    Expects pre-extracted CLIP image features as input.
    Only the text side of CLIP runs during training
    (text_projection + learnable prompt vectors).
    """

    def __init__(self,
                 class_names: list[str] = None,
                 model_name: str = 'ViT-B/16',
                 ctx_length: int = 8,
                 feature_dim: int = 512,
                 alpha: float = 0.2,
                 threshold: float = 0.55,
                 lambda1: float = 0.1,
                 lambda2: float = 0.01,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.class_names = class_names or UCF_CRIME_CLASSES
        self.normal_idx = len(self.class_names) - 1  # "normal" is always last
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # --- Fix #1: use TextPromptEncoder, not CLIPModel ---
        self.text_encoder = TextPromptEncoder(
            self.class_names, model_name, ctx_length, device
        )
        self.nvp   = NormalityVisualPrompt(feature_dim)
        self.plg   = PseudoLabelGenerator(alpha, threshold)
        self.tcsal = TCSAL(feature_dim)

        self.normal_ranking_loss  = NormalRankingLoss()
        self.anomaly_ranking_loss = AnomalyRankingLoss()
        self.dil                  = DistributionalInconsistencyLoss()
        self.sparsity_loss        = SparsityLoss()
        self.smoothness_loss      = SmoothnessLoss()

    def forward(self,
                x_normal: torch.Tensor,
                x_abnormal: torch.Tensor,
                tau: torch.Tensor) -> dict:
        """
        Args:
            x_normal:   (B_n, T, D) — pre-extracted CLIP features, normal videos
            x_abnormal: (B_a, T, D) — pre-extracted CLIP features, anomaly videos
            tau:        (B_a,) int  — per-sample anomaly class index

        Returns:
            dict with total_loss and intermediate tensors for debugging
        """
        B_n, T, D = x_normal.shape
        B_a = x_abnormal.shape[0]

        # ---- Text embeddings (C, D) ----
        text_emb = self.text_encoder()                      # (C, D)
        T_n = text_emb[self.normal_idx]                     # (D,)

        # ---- NVP: enhance normal text with visual grounding ----
        T_n_enhanced = self.nvp(x_normal, T_n)              # (D,)

        # ---- Normal video similarities ----
        # S_nn: how well normal frames match "normal" text
        S_nn = torch.einsum('btd,d->bt', x_normal, T_n)    # (B_n, T)

        # phi_na: how well normal frames match each anomaly text
        # We need all anomaly class embeddings (everything except normal)
        anomaly_idx = [i for i in range(len(self.class_names))
                       if i != self.normal_idx]
        T_all_anomaly = text_emb[anomaly_idx]               # (K-1, D)
        S_na = torch.einsum('btd,kd->btk', x_normal, T_all_anomaly)  # (B_n, T, K-1)

        # ---- Fix #2: Per-sample anomaly similarities ----
        # Each abnormal video has its own tau, so we gather per-sample
        T_a_per_sample = text_emb[tau]                      # (B_a, D)

        # S_aa: each video matched against its OWN anomaly class text
        S_aa = torch.einsum('btd,bd->bt', x_abnormal, T_a_per_sample)  # (B_a, T)

        # S_an: anomaly video frames matched against enhanced normal text
        S_an = torch.einsum('btd,d->bt', x_abnormal, T_n_enhanced)     # (B_a, T)

        # S_aa_other: similarity to all anomaly classes EXCEPT each sample's true class
        # Full similarity to all anomaly classes: (B_a, T, K-1)
        S_aa_all = torch.einsum('btd,kd->btk', x_abnormal, T_all_anomaly)

        # Build per-sample "other" mask: for each sample, exclude its true class
        # anomaly_idx maps position k -> class index. We need to find which k
        # corresponds to each sample's tau.
        anomaly_idx_t = torch.tensor(anomaly_idx, device=x_abnormal.device)  # (K-1,)
        # tau_in_anomaly[b] = position of tau[b] within anomaly_idx
        tau_in_anomaly = (anomaly_idx_t.unsqueeze(0) == tau.unsqueeze(1))  # (B_a, K-1)
        # other_mask[b, k] = True if class k is NOT the true class for sample b
        other_mask = ~tau_in_anomaly                        # (B_a, K-1)

        # For ranking loss, we need max over "other" classes per sample
        # Set true-class positions to -inf so they don't affect the max
        S_aa_other_masked = S_aa_all.clone()
        S_aa_other_masked[tau_in_anomaly.unsqueeze(1).expand_as(S_aa_all)] = float('-inf')
        # Shape: (B_a, T, K-1), with one class per sample masked out

        # ---- Pseudo labels ----
        pseudo_labels_abnormal = self.plg(S_aa, S_an).float()       # (B_a, T)

        # ---- Fix #3: Normal videos get all-zero pseudo labels ----
        pseudo_labels_normal = torch.zeros(B_n, T, device=x_normal.device, dtype=torch.float32)

        # ---- Classifier predictions for BOTH normal and abnormal ----
        pred_normal  = self.tcsal(x_normal)                 # (B_n, T)
        pred_abnormal = self.tcsal(x_abnormal)              # (B_a, T)

        # ---- Fix #4 & #5: Min-max normalize S_aa for DIL and temporal losses ----
        S_aa_norm = minmax_norm(S_aa, dim=-1)               # (B_a, T)
        S_an_norm = minmax_norm(S_an, dim=-1)               # (B_a, T)

        # ---- Losses ----
        # Ranking losses (Eq 10, 11)
        L_n_rank = self.normal_ranking_loss(S_nn, S_na)
        L_a_rank = self.anomaly_ranking_loss(S_aa, S_aa_other_masked, S_an)

        # DIL on normalized similarities (Eq 12)
        L_dil = self.dil(S_aa_norm, S_an_norm)

        L_cl_normal = F.binary_cross_entropy(
            pred_normal.clamp(1e-7, 1 - 1e-7),
            pseudo_labels_normal
        )

        # Abnormal videos: reversed formulation
        L_cl_abnormal = -(
            pred_abnormal * torch.log(pseudo_labels_abnormal.clamp(1e-7, 1 - 1e-7)) +
            (1 - pred_abnormal) * torch.log((1 - pseudo_labels_abnormal).clamp(1e-7, 1 - 1e-7))
        ).mean()

        L_cl = (L_cl_normal + L_cl_abnormal) / 2

        # Temporal losses on normalized S_aa (Eq: Lsp, Lsm)
        L_sp = self.sparsity_loss(S_aa_norm)
        L_sm = self.smoothness_loss(S_aa_norm)

        total_loss = (L_n_rank + L_a_rank + L_dil + L_cl
                      + self.lambda1 * L_sp + self.lambda2 * L_sm)

        return {
            'total_loss':    total_loss,
            'L_n_rank':      L_n_rank,
            'L_a_rank':      L_a_rank,
            'L_dil':         L_dil,
            'L_cl':          L_cl,
            'L_sp':          L_sp,
            'L_sm':          L_sm,
            'pseudo_labels': pseudo_labels_abnormal,
            'pred_abnormal': pred_abnormal,
            'pred_normal':   pred_normal,
            'S_aa':          S_aa,
            'S_an':          S_an,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Test-time inference. Only TCSAL + classifier needed.

        Args:
            x: (B, T, D) — pre-extracted CLIP features
        Returns:
            scores: (B, T) — frame-level anomaly scores in (0, 1)
        """
        return self.tcsal(x)