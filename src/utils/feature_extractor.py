import torch
import clip
from typing import Optional
from torchvision.io import read_video
from PIL import Image

class CLIPImageFeatureExtractor:
    def __init__(self, model_name='ViT-B/16', device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def extract_video(self, video_path: str, fps: Optional[int] = None):
        video, _, info = read_video(str(video_path), pts_unit='sec')
        video_fps = info['video_fps']

        if fps is None:
            sampled_frames = video  # All frames
        else:
            frame_interval = int(video_fps / fps)
            sampled_frames = video[::frame_interval]  # (T, H, W, C)

        frames_preprocessed = torch.stack([
            self.preprocess(Image.fromarray(frame.numpy()))
            for frame in sampled_frames
        ]).to(self.device)
        
        return self.model.encode_image(frames_preprocessed)

        