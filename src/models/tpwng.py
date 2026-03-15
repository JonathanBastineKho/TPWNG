# TPWNG wrapper that combines CLIP text encoder, NVP, PLG, TCSAL and training losses.
# It organizes the overall training pipeline of the TPWNG model.
# The implementation will be refined and updated during development.




from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import (
    AnomalyRankingLoss,
    DistributionalInconsistencyLoss,
    NormalRankingLoss,
    SmoothnessLoss,
    SparsityLoss,
)
from src.models.clip import PromptTextEncoder
from src.models.nvp import NormalityVisualPrompt
from src.models.plg import PseudoLabelGenerator
from src.models.tcsal import TCSAL


# Default class vocabulary used in UCF-Crime:
# 1 normal class + 13 anomaly classes.
DEFAULT_CLASSNAMES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


class TPWNG(nn.Module):
    """
    End-to-end TPWNG wrapper.

    Main responsibilities:
    1) Call model blocks in paper order:
       PromptTextEncoder -> NVP -> similarity tensors -> PLG -> TCSAL.
    2) Aggregate losses from src/losses for training.
    3) Return a unified output dictionary for train.py.
    """

    def __init__(
        self,
        classnames: Optional[Sequence[str]] = None,
        normal_class_name: str = "Normal",
        feature_dim: int = 512,
        n_ctx: int = 8,
        nvp_hidden_dim: int = 2048,
        nvp_dropout: float = 0.1,
        plg_alpha: float = 0.2,
        plg_threshold: float = 0.55,
        tcsal_layers: int = 4,
        tcsal_heads: int = 4,
        tcsal_R: float = 256.0,
        tcsal_dropout: float = 0.1,
        lambda_rank_n: float = 1.0,
        lambda_rank_a: float = 1.0,
        lambda_dil: float = 1.0,
        lambda_cl: float = 1.0,
        lambda_sp: float = 1.0,
        lambda_sm: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        # Class index bookkeeping used by text selection and label conversion.
        self.classnames = list(classnames) if classnames is not None else list(DEFAULT_CLASSNAMES)
        self.class_to_idx = {c.lower(): i for i, c in enumerate(self.classnames)}

        normal_key = normal_class_name.lower()
        if normal_key not in self.class_to_idx:
            raise ValueError(f"normal_class_name '{normal_class_name}' is not in classnames")

        self.normal_idx = self.class_to_idx[normal_key]
        self.anomaly_global_indices = [i for i in range(len(self.classnames)) if i != self.normal_idx]
        self.global_to_anomaly_idx = {
            global_idx: anom_idx for anom_idx, global_idx in enumerate(self.anomaly_global_indices)
        }

        # Model blocks (paper pipeline).
        self.text_encoder = PromptTextEncoder(self.classnames, n_ctx=n_ctx)
        self.nvp = NormalityVisualPrompt(
            feature_dim=feature_dim,
            ffn_hidden_dim=nvp_hidden_dim,
            dropout=nvp_dropout,
        )
        self.plg = PseudoLabelGenerator(alpha=plg_alpha, threshold=plg_threshold)
        self.tcsal = TCSAL(
            d_model=feature_dim,
            n_layers=tcsal_layers,
            n_heads=tcsal_heads,
            R=tcsal_R,
            dropout=tcsal_dropout,
        )

        # Loss blocks imported from src/losses.
        self.normal_rank_loss = NormalRankingLoss()
        self.anomaly_rank_loss = AnomalyRankingLoss()
        self.dil_loss = DistributionalInconsistencyLoss()
        self.sparsity_loss = SparsityLoss()
        self.smoothness_loss = SmoothnessLoss()
        self.bce_loss = nn.BCELoss()

        # Loss weights for Eq(14)-style aggregation.
        self.lambda_rank_n = lambda_rank_n
        self.lambda_rank_a = lambda_rank_a
        self.lambda_dil = lambda_dil
        self.lambda_cl = lambda_cl
        self.lambda_sp = lambda_sp
        self.lambda_sm = lambda_sm

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    # Forward-pass helpers.
    # Execution structure:
    # 1) prepare features
    # 2) encode text prompts
    # 3) enhance normal text via NVP
    # 4) build similarity tensors
    # 5) generate pseudo labels via PLG
    # 6) predict frame scores via TCSAL
    # 7) compose losses
    def _prepare_video_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected features with shape (B,T,D) or (T,D), got {tuple(x.shape)}")
        x = x.to(self.device).float()
        return F.normalize(x, p=2, dim=-1)

    def _similarity_to_text(self, video_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        video_feats: (B,T,D), normalized
        text_feats : (D,) or (K,D), normalized
        returns:
          (B,T)   if text_feats is (D,)
          (B,T,K) if text_feats is (K,D)
        """
        text_feats = F.normalize(text_feats.float(), p=2, dim=-1)
        if text_feats.dim() == 1:
            return torch.matmul(video_feats, text_feats)
        return torch.einsum("btd,kd->btk", video_feats, text_feats)

    def _abnormal_labels_to_indices(
        self, abnormal_labels: Optional[Union[Sequence[str], torch.Tensor]], batch_size: int
    ) -> torch.Tensor:
        """
        Convert abnormal labels to anomaly-class indices in [0, K_anomaly-1].
        If labels are missing, fallback to zeros for shape completeness.
        """
        if abnormal_labels is None:
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)

        if isinstance(abnormal_labels, torch.Tensor):
            raw = abnormal_labels.to(self.device).long().view(-1)
            if raw.numel() != batch_size:
                raise ValueError("abnormal_labels tensor size does not match batch size")
            global_indices = raw.tolist()
        else:
            if len(abnormal_labels) != batch_size:
                raise ValueError("abnormal_labels length does not match batch size")
            global_indices = []
            for label in abnormal_labels:
                key = str(label).lower()
                if key not in self.class_to_idx:
                    raise ValueError(f"Unknown class label: {label}")
                global_indices.append(self.class_to_idx[key])

        anomaly_indices = []
        for global_idx in global_indices:
            if global_idx == self.normal_idx:
                raise ValueError("abnormal_labels contains the Normal class")
            if global_idx not in self.global_to_anomaly_idx:
                raise ValueError(f"Invalid class index: {global_idx}")
            anomaly_indices.append(self.global_to_anomaly_idx[global_idx])
        return torch.tensor(anomaly_indices, dtype=torch.long, device=self.device)

    def _split_true_vs_other(
        self, anomaly_sims: torch.Tensor, true_anomaly_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        anomaly_sims : (B,T,K_anomaly)
        true_anomaly_idx : (B,)
        returns:
          S_aa_true    : (B,T)
          phi_aa_other : (B,T,K_anomaly-1)
        """
        B, T, K = anomaly_sims.shape
        gather_idx = true_anomaly_idx.view(B, 1, 1).expand(B, T, 1)
        s_aa_true = anomaly_sims.gather(dim=2, index=gather_idx).squeeze(-1)

        true_mask = F.one_hot(true_anomaly_idx, num_classes=K).bool()
        other_mask = ~true_mask[:, None, :].expand(B, T, K)
        phi_aa_other = anomaly_sims[other_mask].view(B, T, K - 1)
        return s_aa_true, phi_aa_other

    def forward(
        self,
        normal_features: torch.Tensor,
        abnormal_features: torch.Tensor,
        abnormal_labels: Optional[Union[Sequence[str], torch.Tensor]] = None,
        return_intermediates: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            normal_features:   (Bn,T,D) or (T,D)
            abnormal_features: (Ba,T,D) or (T,D)
            abnormal_labels:   class names or global class indices for abnormal videos
        Returns:
            dict with total loss, each sub-loss, scores, and optional intermediates.
        """
        normal_features = self._prepare_video_features(normal_features)
        abnormal_features = self._prepare_video_features(abnormal_features)

        # 1) PromptTextEncoder
        text_embeddings = self.text_encoder()  # (N_class,D)
        normal_text = text_embeddings[self.normal_idx]  # (D,)
        anomaly_text = text_embeddings[self.anomaly_global_indices]  # (K,D)

        # 2) NVP
        enhanced_normal_text = self.nvp(normal_features, normal_text)
        enhanced_normal_text = F.normalize(enhanced_normal_text, p=2, dim=-1)

        # 3) Similarity tensors
        s_nn = self._similarity_to_text(normal_features, enhanced_normal_text)  # (Bn,T)
        phi_na = self._similarity_to_text(normal_features, anomaly_text)  # (Bn,T,K)
        s_an = self._similarity_to_text(abnormal_features, enhanced_normal_text)  # (Ba,T)
        anomaly_sims = self._similarity_to_text(abnormal_features, anomaly_text)  # (Ba,T,K)

        true_anomaly_idx = self._abnormal_labels_to_indices(abnormal_labels, batch_size=abnormal_features.size(0))
        s_aa_true, phi_aa_other = self._split_true_vs_other(anomaly_sims, true_anomaly_idx)

        # 4) PLG (pseudo labels for abnormal videos)
        pseudo_labels = self.plg(s_aa_true.detach(), s_an.detach()).float()

        # 5) TCSAL scores
        normal_scores = self.tcsal(normal_features)
        abnormal_scores = self.tcsal(abnormal_features)

        # 6) Loss composition:
        # l_total =
        #   lambda_rank_n * l_rank_n   (NormalRankingLoss)
        # + lambda_rank_a * l_rank_a   (AnomalyRankingLoss)
        # + lambda_dil    * l_dil      (DistributionalInconsistencyLoss)
        # + lambda_cl     * l_cl       (BCE: abnormal pseudo labels + normal zeros)
        # + lambda_sp     * l_sp       (SparsityLoss in temporal_losses.py)
        # + lambda_sm     * l_sm       (SmoothnessLoss in temporal_losses.py)
        l_rank_n = self.normal_rank_loss(s_nn, phi_na)
        l_rank_a = self.anomaly_rank_loss(s_aa_true, phi_aa_other, s_an)
        l_dil = self.dil_loss(s_aa_true, s_an)

        zeros = torch.zeros_like(normal_scores)
        l_cl = self.bce_loss(abnormal_scores, pseudo_labels) + self.bce_loss(normal_scores, zeros)

        l_sp = self.sparsity_loss(abnormal_scores)
        l_sm = self.smoothness_loss(abnormal_scores)

        l_total = (
            self.lambda_rank_n * l_rank_n
            + self.lambda_rank_a * l_rank_a
            + self.lambda_dil * l_dil
            + self.lambda_cl * l_cl
            + self.lambda_sp * l_sp
            + self.lambda_sm * l_sm
        )

        output: Dict[str, torch.Tensor] = {
            "loss_total": l_total,
            "loss_rank_n": l_rank_n,
            "loss_rank_a": l_rank_a,
            "loss_dil": l_dil,
            "loss_cl": l_cl,
            "loss_sp": l_sp,
            "loss_sm": l_sm,
            "scores_normal": normal_scores,
            "scores_abnormal": abnormal_scores,
            "pseudo_labels": pseudo_labels,
        }

        if return_intermediates:
            output.update(
                {
                    "text_embeddings": text_embeddings,
                    "enhanced_normal_text": enhanced_normal_text,
                    "S_nn": s_nn,
                    "phi_na": phi_na,
                    "S_an": s_an,
                    "S_aa_true": s_aa_true,
                    "phi_aa_other": phi_aa_other,
                }
            )
        return output

    @torch.no_grad()
    def predict_scores(self, video_features: torch.Tensor) -> torch.Tensor:
        """
        Inference helper used by evaluate.py.
        Args:
            video_features: (B,T,D) or (T,D)
        Returns:
            frame scores: (B,T) or (1,T)
        """
        video_features = self._prepare_video_features(video_features)
        return self.tcsal(video_features)
