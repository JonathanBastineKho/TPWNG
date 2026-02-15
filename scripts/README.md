# Scripts

Utility scripts for data preprocessing and model training/evaluation.

## Feature Extraction

### `extract_features.py`

Extracts CLIP visual features from video frames for efficient training.

**Usage:**

```bash
# Basic usage - extract features from videos
python scripts/extract_features.py \
    --video_dir data/ucfcrime/train \
    --output_dir data/features/train \
    --model ViT-B/16 \
    --device cuda

# Extract test set features
python scripts/extract_features.py \
    --video_dir data/ucfcrime/test \
    --output_dir data/features/test \
    --model ViT-B/16 \
    --device cuda

# Sample at specific FPS (optional)
python scripts/extract_features.py \
    --video_dir data/ucfcrime/train \
    --output_dir data/features/train \
    --fps 1 \
    --device cuda
```
### `train.py`
Coming soon

### `evaluate.py`
coming soon