import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

from src.utils.feature_extractor import CLIPImageFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    extractor = CLIPImageFeatureExtractor(
        model_name=args.model,
        device=args.device
    )
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_paths = list(video_dir.glob('**/*.mp4'))
    
    for video_path in tqdm(video_paths, desc='Extracting features'):
        features = extractor.extract_video(video_path, fps=args.fps)
        
        relative_path = video_path.relative_to(video_dir)
        output_path = output_dir / relative_path.with_suffix('.pt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(features, output_path)
    
    logger.info(f"Extracted features for {len(video_paths)} videos")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='data/raw/')
    parser.add_argument('--output_dir', type=str, default='data/features')
    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    main(args)
