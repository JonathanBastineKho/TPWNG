import torch
from torch import nn

class NormalityVisualPrompt(nn.Module):
    """
    Docstring for NormalityVisualPrompt
    """

    def __init__(self, 
                 feature_dim: int = 512,
                 ffn_hidden_dim: int = 2048, 
                 dropout: int = 0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim * 2, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, feature_dim)
        )

    def forward(self, 
                normal_video_features: torch.Tensor, 
                normal_text_embedding: torch.Tensor):
        """
        Args:
            normal_video_features: (B, F, D) - Features from normal videos
            normal_text_embedding: (D,) - "Normal" text embedding
        
        Returns:
            enhanced_normal_text: (D,) - Enhanced normal text embedding
        """
        # Eq (3): Compute similarities and aggregate
        similarities = torch.matmul(
            normal_video_features,
            normal_text_embedding.unsqueeze(-1)
        ).squeeze(-1)

        weights = torch.softmax(similarities, dim=1)
        nvp = torch.matmul(
            weights.unsqueeze(1),
            normal_video_features
        ).squeeze(1).mean(dim=0)

        # Eq (4): FFN with residual
        concatenated = torch.cat([normal_text_embedding, nvp], dim=0)
        return self.ffn(concatenated) + normal_text_embedding