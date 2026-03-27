import torch
from torch import nn
import math

class NormalityVisualPrompt(nn.Module):
    """
    Normality Visual Prompt (Section 3.2, Eq 3-4).
    
    Uses normal video frames to ground the abstract "normal" text embedding
    in actual visual appearance, producing an enhanced text embedding that
    better matches normal frames in anomalous videos.
    """

    def __init__(self,
                 feature_dim: int = 512,
                 ffn_hidden_dim: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim * 2, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, feature_dim),
        )

    def forward(self,
                normal_video_features: torch.Tensor,
                normal_text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normal_video_features: (B, F, D) — frame features from normal videos
            normal_text_embedding: (D,)      — "normal" class text embedding

        Returns:
            enhanced_normal_text:  (D,)      — visually-grounded normal embedding
        """
        # ---- Eq (3): per-video similarity-weighted aggregation ----
        # S_nn_i = X_n_i @ T_n^T  →  (B, F)
        S_nn = torch.matmul(
            normal_video_features,              # (B, F, D)
            normal_text_embedding.unsqueeze(-1)  # (D, 1)
        ).squeeze(-1)                            # (B, F)

        # weights_i = softmax(S_nn_i)  →  (B, F)
        weights = torch.softmax(S_nn / math.sqrt(normal_video_features.shape[-1]), dim=1)

        # Q_i = weights_i^T @ X_n_i  →  (B, D)
        Q = torch.einsum('bf,bfd->bd', weights, normal_video_features)  # (B, D)

        # Average across batch to get a single NVP prototype
        Q_mean = Q.mean(dim=0)  # (D,)

        # ---- Eq (4): FFN with skip connection ----
        # T_n_enhanced = FFN([T_n; Q]) + T_n
        concatenated = torch.cat([normal_text_embedding, Q_mean], dim=0)  # (2D,)
        return self.ffn(concatenated) + normal_text_embedding              # (D,)