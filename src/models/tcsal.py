import math
import torch
from torch import nn


class AdaptiveSpanAttention(nn.Module):
    """
    Self-attention where each head learns its own attention span (Eq 7-9).
    The span z is computed from the input, so short events get short spans
    and long events get long spans automatically.
    """

    def __init__(self, d_model: int, n_heads: int, R: float = 256.0, dropout: float = 0.1):
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

        # Predicts one span scalar per head from pooled input (Eq 8)
        self.span_net = nn.Linear(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

    def _soft_mask(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """χz(h) = clamp((1/R) * (R + z - h), 0, 1)  — Eq 7"""
        return ((1.0 / self.R) * (self.R + z[:, :, None, None] - h)).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D)
        """
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Learned span per head: z in (0, T)  — Eq 8
        z = T * torch.sigmoid(self.span_net(x.mean(dim=1)))  # (B, H)

        # Distance grid: dist[t, r] = t - r
        idx = torch.arange(T, device=x.device).float()
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0)  # (T, T), causal

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Mask out future positions
        future = (idx.unsqueeze(0) > idx.unsqueeze(1))  # [t, r]: r > t
        scores = scores.masked_fill(future.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = attn * self._soft_mask(dist.unsqueeze(0).unsqueeze(0), z)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                              # (B, H, T, d)
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


class TFEncoder(nn.Module):
    """
    Standard Transformer encoder (global self-attention, causal mask).
    Used for Table 4 ablation (w TF-encoder).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> scores (B, T). Causal mask so position t only sees 0..t."""
        B, T, D = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask)
        return self.classifier(x).squeeze(-1)