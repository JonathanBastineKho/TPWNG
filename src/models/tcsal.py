import math
import torch
from torch import nn


class AdaptiveSpanAttention(nn.Module):
    """
    Bidirectional self-attention where each head learns its own
    attention span via a soft mask (Eq 7-9).
    """

    def __init__(self, d_model: int, n_heads: int, R: float = 256.0,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.R = R

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        # Eq 8: predicts one span scalar per head from pooled input
        self.span_net = nn.Linear(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

    def _soft_mask(self, dist: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        χ_z(h) = clamp((1/R)(R + z - h), 0, 1)  — Eq 7

        Args:
            dist: (T, T)    — absolute distance between every pair of positions
            z:    (B, H)    — learned span per head
        Returns:
            mask: (B, H, T, T)
        """
        # z[:, :, None, None] broadcasts to (B, H, 1, 1)
        # dist[None, None, :, :] broadcasts to (1, 1, T, T)
        return ((self.R + z[:, :, None, None] - dist[None, None]) / self.R).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        z = T * torch.sigmoid(self.span_net(x.mean(dim=1)))  # (B, H)

        # CAUSAL distance: h = t - r (only past positions, r < t)
        idx = torch.arange(T, device=x.device, dtype=torch.float32)
        dist = idx.unsqueeze(0) - idx.unsqueeze(1)  # (T, T), dist[t,r] = t - r

        # Causal mask: frame t can only attend to frames r <= t
        causal_mask = (dist >= 0).float()  # (T, T), upper-left triangle + diagonal

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask BEFORE softmax (mask out future positions)
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf')
        )

        # Soft span mask on causal distances (only non-negative distances)
        dist_causal = dist.clamp(min=0)  # (T, T)
        span_mask = self._soft_mask(dist_causal, z)  # (B, H, T, T)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * span_mask
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
    
class TCSALLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, R: float = 256.0, dropout: float = 0.1):
        super().__init__()
        self.attn = AdaptiveSpanAttention(d_model, n_heads, R, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
    
class TCSAL(nn.Module):
    """
    Temporal Context Self-Adaptive Learning (Section 3.4).
    4 transformer layers, 4 heads, each head learns its own attention span.
    Classifier: LayerNorm -> Linear -> Sigmoid (supplementary §1).
    """

    def __init__(self,
                 d_model: int = 512,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 R: float = 256.0,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(*[
            TCSALLayer(d_model, n_heads, R, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - per-frame CLIP features
        Returns:
            scores: (B, T) - frame-level anomaly scores in (0, 1)
        """
        x = self.layers(x)
        return self.classifier(x).squeeze(-1)