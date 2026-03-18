"""
CLIP text encoder with learnable prompt (CoOp-style). No global state; safe for training.
"""
import torch
import torch.nn as nn
from typing import List

try:
    import clip
except ImportError:
    clip = None


class PromptTextEncoder(nn.Module):
    """
    Learnable text prompt + CLIP text encoder. Outputs L2-normalized text embeddings (K, D).
    """

    def __init__(
        self,
        classnames: List[str],
        clip_model: nn.Module,
        n_ctx: int = 8,
        device: torch.device = None,
    ):
        super().__init__()
        if clip is None:
            raise ImportError("openai-clip is required. pip install openai-clip")
        self.clip_model = clip_model
        self.classnames = list(classnames)
        self.n_ctx = n_ctx
        self.device = device or next(clip_model.parameters()).device
        dtype = clip_model.dtype
        ctx_dim = clip_model.text_projection.shape[0]  # D

        self.ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim, dtype=dtype) * 0.02)
        # Tokenize each class; CLIP tokenize returns (1, 77), we cat to (K, 77)
        tokenized = torch.cat([clip.tokenize(c) for c in self.classnames]).to(self.device)
        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            text_features: (K, D) L2-normalized embeddings for all classes.
        """
        tokenized = self.tokenized_prompts  # (K, 77)
        x = self.clip_model.token_embedding(tokenized).type(self.clip_model.dtype)
        ctx = self.ctx.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x[:, :1, :], ctx, x[:, 1:, :]], dim=1)
        x = x[:, :77, :]
        x = x + self.clip_model.positional_embedding[: x.shape[1]]
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        eot_pos = tokenized.argmax(dim=-1)
        sentence_feature = x[torch.arange(x.shape[0], device=x.device), eot_pos]
        text_features = sentence_feature @ self.clip_model.text_projection
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


def load_clip_for_tpwng(version: str = "ViT-B/16", device: torch.device = None) -> nn.Module:
    """Load CLIP, freeze all except text_projection. Returns the model (float)."""
    if clip is None:
        raise ImportError("openai-clip is required")
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model, _ = clip.load(version, device=device)
    model = model.float()
    for p in model.parameters():
        p.requires_grad = False
    model.text_projection.requires_grad = True
    return model
