"""
Full TPWNG model: CLIP text prompt + NVP + PLG + temporal module (TCSAL or TF-encoder).
Wires all components for training and inference.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .clip_encoder import PromptTextEncoder, load_clip_for_tpwng
from .nvp import NormalityVisualPrompt
from .plg import PseudoLabelGenerator
from .tcsal import TCSAL, TFEncoder


def _normalize_similarity(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Min-max normalize along dim."""
    xmin = x.min(dim=dim, keepdim=True)[0]
    xmax = x.max(dim=dim, keepdim=True)[0]
    return (x - xmin) / (xmax - xmin + 1e-8)


class TPWNG(nn.Module):
    """
    Text Prompt with Normality Guidance for WSVAD.
    Config supports ablations: NVP mode, normality guidance, temporal module, loss terms.
    """

    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        D = config.get("feature_dim", 512)
        classes: List[str] = config["classes"]
        self.num_classes = K = len(classes)
        self.normal_idx = K - 1  # last is Normal
        self.anomaly_indices = list(range(K - 1))

        # CLIP + learnable prompt
        clip_model = load_clip_for_tpwng(device=self.device)
        self.clip_model = clip_model
        self.text_encoder = PromptTextEncoder(
            classnames=classes,
            clip_model=clip_model,
            n_ctx=config.get("n_ctx", 8),
            device=self.device,
        )

        # NVP (optional; mode: "as" | "fa" | "none")
        self.use_nvp = config.get("use_nvp", True)
        self.nvp_mode = config.get("nvp_mode", "as")
        self.nvp = NormalityVisualPrompt(feature_dim=D) if self.use_nvp else None

        # PLG
        self.use_ng = config.get("use_normality_guidance", True)
        self.plg = PseudoLabelGenerator(
            alpha=config.get("alpha", 0.2),
            threshold=config.get("theta", 0.55),
        )

        # Temporal module
        tm = config.get("temporal_module", "tcsal").lower()
        if tm == "tcsal":
            self.temporal = TCSAL(
                d_model=D,
                n_layers=4,
                n_heads=4,
                R=config.get("R", 256.0),
                dropout=0.1,
            )
        elif tm in ("tf_encoder", "tf"):
            self.temporal = TFEncoder(d_model=D, n_layers=4, n_heads=4, dropout=0.1)
        else:
            # MTN / GL-MHSA not implemented; fallback to TCSAL
            self.temporal = TCSAL(d_model=D, n_layers=4, n_heads=4, R=config.get("R", 256.0), dropout=0.1)

        self.to(self.device)

    def get_text_embeddings(self) -> torch.Tensor:
        """(K, D) L2-normalized."""
        return self.text_encoder()

    def _compute_nvp(self, normal_feats: torch.Tensor, T_n: torch.Tensor) -> torch.Tensor:
        """normal_feats (B_n, F, D), T_n (D,) -> enhanced T_n (D,)"""
        if not self.use_nvp or self.nvp is None:
            return T_n
        if self.nvp_mode == "fa":
            # NVP-FA: frame average (paper Table 2)
            nvp_vec = normal_feats.reshape(-1, normal_feats.shape[-1]).mean(dim=0)
            concat = torch.cat([T_n, nvp_vec], dim=0)
            return self.nvp.ffn(concat) + T_n
        # NVP-AS: aggregate by similarity
        return self.nvp(normal_feats, T_n)

    def forward(
        self,
        normal_feats: torch.Tensor,
        anomaly_feats: torch.Tensor,
        anomaly_class_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        normal_feats: (B_n, F_n, D)
        anomaly_feats: (B_a, F_a, D)
        anomaly_class_idx: (B_a,) int in [0, K-2] (index of anomaly class per video)
        """
        B_n, F_n, _ = normal_feats.shape
        B_a, F_a, _ = anomaly_feats.shape
        T_emb = self.get_text_embeddings()  # (K, D)
        T_anomaly = T_emb[: self.normal_idx]   # (K-1, D)
        T_n = T_emb[self.normal_idx : self.normal_idx + 1].squeeze(0)  # (D,)

        # NVP: enhance normal text using normal video batch
        T_n_enhanced = self._compute_nvp(normal_feats, T_n)  # (D,)

        # Similarities for normal batch
        # S_nn: (B_n, F_n) = X_n @ T_n_enhanced
        S_nn = torch.matmul(normal_feats, T_n_enhanced)  # (B_n, F_n)
        # phi_na: (B_n, F_n, K-1) normal frames x anomaly texts
        phi_na = torch.matmul(normal_feats, T_anomaly.T)  # (B_n, F_n, K-1)

        # Similarities for anomaly batch
        S_an = torch.matmul(anomaly_feats, T_n_enhanced)  # (B_a, F_a)
        # S_aa for true class: (B_a, F_a); index by tau
        tau = anomaly_class_idx  # (B_a,)
        S_aa_true = torch.zeros(B_a, F_a, device=anomaly_feats.device, dtype=anomaly_feats.dtype)
        for b in range(B_a):
            S_aa_true[b] = torch.matmul(anomaly_feats[b], T_anomaly[tau[b]])
        # phi_aa_other: for each sample, other (K-2) anomaly classes
        phi_aa_other_list = []
        for b in range(B_a):
            other_idx = [i for i in range(self.normal_idx) if i != tau[b].item()]
            if not other_idx:
                phi_aa_other_list.append(torch.zeros(F_a, 1, device=anomaly_feats.device))
            else:
                T_other = T_anomaly[other_idx]  # (K-2, D)
                phi_aa_other_list.append(torch.matmul(anomaly_feats[b], T_other.T))
        max_other = max(p.shape[1] for p in phi_aa_other_list)
        phi_aa_other = torch.zeros(B_a, F_a, max_other, device=anomaly_feats.device, dtype=anomaly_feats.dtype)
        for b, p in enumerate(phi_aa_other_list):
            phi_aa_other[b, :, : p.shape[1]] = p

        # Pseudo-labels (with or without normality guidance)
        if self.use_ng:
            plg_S_aa = S_aa_true
            plg_S_an = S_an
        else:
            plg_S_aa = S_aa_true
            plg_S_an = torch.zeros_like(S_an)  # no guidance: effectively only S_aa
        pseudo_labels = self.plg(plg_S_aa, plg_S_an)  # (B_a, F_a)

        # Classifier scores (TCSAL / TF)
        scores = self.temporal(anomaly_feats)  # (B_a, F_a)

        # Normalized S_aa for temporal losses (L_sp, L_sm)
        S_aa_norm = _normalize_similarity(S_aa_true, dim=1)

        return {
            "S_nn": S_nn,
            "phi_na": phi_na,
            "S_an": S_an,
            "S_aa_true": S_aa_true,
            "phi_aa_other": phi_aa_other,
            "S_aa_norm": S_aa_norm,
            "pseudo_labels": pseudo_labels,
            "scores": scores,
            "T_n_enhanced": T_n_enhanced,
        }

    def forward_inference(self, feats: torch.Tensor) -> torch.Tensor:
        """For evaluation: (B, F, D) -> (B, F) frame-level anomaly scores."""
        return self.temporal(feats)
