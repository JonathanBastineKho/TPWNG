import torch
import clip
from typing import Optional
from torchcodec.decoders import VideoDecoder
from PIL import Image

class CLIPImageFeatureExtractor:
    def __init__(self, model_name: str = 'ViT-B/16', device: str = 'cuda', chunk_size: int = 2048):
        self.device = device
        self.chunk_size = chunk_size
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def extract_video(self, video_path: str) -> torch.Tensor:
        decoder = VideoDecoder(str(video_path), device='cpu', dimension_order='NHWC')
        n_frames = len(decoder)

        all_features = []
        for start in range(0, n_frames, self.chunk_size):
            frames = decoder[start:start + self.chunk_size].data

            chunk = torch.stack([
                self.preprocess(Image.fromarray(f.numpy()))
                for f in frames
            ]).to(self.device)

            feats = self.model.encode_image(chunk).float().cpu()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)

            torch.cuda.empty_cache()

        return torch.cat(all_features, dim=0)