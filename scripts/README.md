# Scripts

Utility scripts for data preprocessing, training, evaluation, and running all paper experiments.

## Feature extraction

### `extract_features.py`

Extracts CLIP (ViT-B/16) visual features from video frames. Save outputs as `.pt` per video so that train/eval use precomputed features.

```bash
python scripts/extract_features.py \
    --video_dir data/ucfcrime/train \
    --output_dir data/ucfcrime/train \
    --model ViT-B/16 \
    --device cuda
```

Repeat for `data/ucfcrime/test` (and for XD-Violence if used). Folder layout under `--output_dir` should mirror class folders (Normal/, Abuse/, …) with one `.pt` file per video.

---

## Training

### `train.py`

Trains TPWNG with a given experiment config. All outputs go to `runs/<experiment_name>/`:

- `config.json` – experiment config
- `train.log` – training log
- `metrics.json` – per-epoch metrics
- `checkpoints/checkpoint_epoch{N}.pt` – saved checkpoints
- `plots/loss_curves.png` – loss curves (after training)

```bash
# Full model on UCF-Crime (Table 1); input processed VadCLIP-style (uniform_avg, 256 frames)
python scripts/train.py --experiment_id main_ucfcrime --dataset ucfcrime --data_root data/ucfcrime

# NVP ablation (Table 2): no NVP
python scripts/train.py --experiment_id abl_nvp_none --dataset ucfcrime --data_root data/ucfcrime

# Normality guidance off (Table 3)
python scripts/train.py --experiment_id abl_ng_none --dataset ucfcrime

# Loss ablation (Table 5): baseline (L_cl + L_sp + L_sm only)
python scripts/train.py --experiment_id abl_loss_bs_only --dataset ucfcrime

```

**Input processing (VadCLIP-style):** Training uses the same scheme as VadCLIP. Long videos are reduced to `max_frames` (default **256**) via **uniform_avg**: the video is split into 256 segments and each output frame is the mean of that segment; short videos are zero-padded. Use `--train-sample uniform_avg` (default), `random` (one random 256-frame segment), `uniform` (evenly spaced indices), or `truncate` (first 256). Use `--max_frames N` to change length (VadCLIP uses 256).

Override hyperparameters: `--epochs`, `--lr`, `--batch_size_normal`, `--batch_size_anomaly`, `--output_dir`, `--save_every`, `--device`.

**View performance charts in real time:** TensorBoard is on by default. In a separate terminal run:
```bash
tensorboard --logdir runs
```
Then open http://localhost:6006 in your browser. You’ll see scalars (loss, L_n_rank, L_a_rank, L_cl, etc.) update each epoch. Use `--no-tensorboard` to disable.

**Continuous evaluation:** Use `--eval-every N` (e.g. `--eval-every 5`) to run test-set evaluation every N epochs; the metric is logged to TensorBoard under `eval/` and to the log file. Default is 0 (no periodic eval; run `evaluate.py` after training for a single end result).

---

## Evaluation

### `evaluate.py`

Loads a checkpoint and runs the TCSAL classifier on the test set. Computes AUC (UCF-Crime) or AP (XD-Violence) and saves ROC/PR curves under the same run directory.

```bash
python scripts/evaluate.py \
    --checkpoint runs/main_ucfcrime_ucfcrime/checkpoints/checkpoint_epoch50.pt \
    --experiment_id main_ucfcrime \
    --dataset ucfcrime \
    --test_root data/ucfcrime/test
```

Writes `runs/<experiment_name>/results.json` and `plots/roc_curve.png` (or `pr_curve.png` for XD-Violence).

Long videos are processed in **chunks** (same idea as VadCLIP) to avoid GPU OOM: use `--max_frames 256` (default) to cap frames per chunk. Smaller values use less memory; larger may be faster but can OOM on very long videos.

---

## Running all paper experiments

### `run_experiments.py`

Runs train + evaluate for a set of experiments and writes a summary table.

```bash
# Single default run (main_ucfcrime)
python scripts/run_experiments.py

# Specific experiments
python scripts/run_experiments.py --experiments main_ucfcrime main_xdviolence abl_nvp_none

# All paper experiments (Tables 1–5)
python scripts/run_experiments.py --all

# Train only (no evaluation)
python scripts/run_experiments.py --experiments main_ucfcrime --train_only

# Shorter run for debugging
python scripts/run_experiments.py --experiments main_ucfcrime --epochs 2
```

Outputs:

- Per run: `runs/<experiment_id>_<dataset>/` (config, logs, checkpoints, metrics, plots, results.json).
- Summary: `runs/summary_<timestamp>.json` and `runs/summary_<timestamp>.md` (result table).

Experiment IDs match the configs in `configs/experiments.py` (e.g. `main_ucfcrime`, `abl_nvp_none`, `abl_ng_none`, `abl_temporal_tf`, `abl_loss_bs_only`, …).
