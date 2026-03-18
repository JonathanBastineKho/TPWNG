# TPWNG

Implementation of **Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection** (CVPR 2024)

## Paper

- **Main Paper**: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.pdf)
- **Supplementary Material**: [CVPR 2024 Supplemental](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Text_Prompt_with_CVPR_2024_supplemental.pdf)

---

## Setup

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```
### Dataset Download

Download the UCF-Crime dataset from:
- [Official UCFCrime Link](https://www.crcv.ucf.edu/projects/real-world/)
- [Dropbox Link](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?dl=0&e=2&rlkey=5bg7mxxbq46t7aujfch46dlvz)

### Expected Dataset Structure

After downloading, organize the data as follows:

```bash
data/
└── ucfcrime/
    ├── train/
    │   ├── Normal/
    │   │   ├── Normal_Videos001.mp4
    │   │   ├── Normal_Videos002.mp4
    │   │   └── ...
    │   ├── Abuse/
    │   │   ├── Abuse001_x264.mp4
    │   │   └── ...
    │   ├── Arrest/
    │   ├── Arson/
    │   ├── Assault/
    │   ├── Burglary/
    │   ├── Explosion/
    │   ├── Fighting/
    │   ├── RoadAccidents/
    │   ├── Robbery/
    │   ├── Shooting/
    │   ├── Shoplifting/
    │   ├── Stealing/
    │   └── Vandalism/
    │
    └── test/
        ├── Normal/
        │   ├── Normal_Videos950.mp4
        │   ├── Normal_Videos950.txt  ← Temporal annotations
        │   └── ...
        ├── Abuse/
        │   ├── Abuse028_x264.mp4
        │   ├── Abuse028_x264.txt  ← Frame-level labels
        │   └── ...
        └── ... (same 13 anomaly classes)
```

**Note**: 
- Training videos have no annotations
- Test videos include `.txt` files with frame-level anomaly intervals
- 14 classes total: 1 Normal + 13 Anomaly types

---

## Experiment running & tracking

Training and evaluation are set up so you can replicate the paper’s experiments and get logs, metrics, and plots in one place.

- **Output directory:** each run writes to `runs/<experiment_id>_<dataset>/`:
  - `config.json`, `train.log`, `metrics.json`
  - `checkpoints/checkpoint_epoch{N}.pt`
  - `plots/loss_curves.png`, `roc_curve.png` / `pr_curve.png`
  - `results.json` (AUC or AP after evaluation)
- **Single run:**  
  `python scripts/train.py --experiment_id main_ucfcrime --dataset ucfcrime --data_root data/ucfcrime`  
  then  
  `python scripts/evaluate.py --checkpoint runs/main_ucfcrime_ucfcrime/checkpoints/checkpoint_epoch50.pt --experiment_id main_ucfcrime --dataset ucfcrime --test_root data/ucfcrime/test`
- **Batch runs:**  
  `python scripts/run_experiments.py --experiments main_ucfcrime abl_nvp_none` or `--all` for all paper experiments.  
  Summary table: `runs/summary_<timestamp>.md`.

See **scripts/README.md** for full usage and experiment IDs (ablations for NVP, normality guidance, temporal module, and loss terms).

---

## Replication status & next steps

See **[NEXT_STEPS.md](NEXT_STEPS.md)** for a short paper summary, implementation audit (what matches the paper and what’s missing), and a concrete checklist to finish the replication (TPWNG model, training, evaluation).

---

## TODO

### Core Components

**Models**
- [x] NVP model (`src/models/nvp.py`)
- [x] PLG model (`src/models/plg.py`)
- [ ] CLIP encoders (`src/models/clip.py`)
- [x] TCSAL module (`src/models/tcsal.py`)
- [ ] Complete TPWNG model (`src/models/tpwng.py`)

**Losses**
- [x] Ranking losses (`src/losses/ranking_loss.py`)
- [x] Temporal losses (`src/losses/temporal_losses.py`)
- [x] DIL loss implementation (`src/losses/dil.py`)

**Data Pipeline**
- [x] UCF-Crime Dataset class (`data/dataset.py`)

**Utilities**
- [x] Experiment logging, metrics, and plots (`src/utils/experiment_logging.py`); TensorBoard for live charts

### Scripts

- [x] Pre-video CLIP extraction (`scripts/extract_features.py`)
- [x] Training script (`scripts/train.py`) with logging and checkpoints
- [x] Evaluation script (`scripts/evaluate.py`) with AUC/AP and ROC/PR plots
- [x] Experiment runner (`scripts/run_experiments.py`) for paper replication

---

