# TPWNG Implementation – Paper Summary & Next Steps

## Paper Summary (CVPR 2024)

**Goal:** Weakly supervised video anomaly detection (WSVAD) using only video-level labels. The method has two stages: (1) generate frame-level pseudo-labels by aligning event text and frames with CLIP, (2) train a classifier on those pseudo-labels.

**Main components:**

1. **CLIP domain adaptation**  
   Fine-tune CLIP (ViT-B/16) so text–frame alignment works for anomaly vs normal. Only the text projection is trainable. Uses:
   - **L_n_rank (Eq 10):** Normal text should match normal frames better than any anomaly text.
   - **L_a_rank (Eq 11):** For anomaly videos, true anomaly text and normal text should match the right frames better than other anomaly texts.
   - **L_dil (Eq 12):** Distribution inconsistency – similarity to anomaly text should be inconsistent with similarity to normal text (cosine similarity between the two score vectors over frames).

2. **Learnable text prompt**  
   CoOp-style: \( l = 8 \) learnable context vectors before the class token (e.g. `[ctx_1..ctx_8, "Fighting"]`). One shared context for all classes.

3. **Normality visual prompt (NVP)**  
   From normal videos: similarity-weighted aggregation of normal frames → one visual vector \( Q \). FFN(concat(\( T^n_k \), \( Q \))) + \( T^n_k \) gives enhanced normal text embedding \( \dot{T}^n_k \). Helps normal text match normal frames inside anomaly videos.

4. **Pseudo-label generation (PLG)**  
   For each anomaly video:  
   \( \psi = \alpha \tilde{S}_{an} + (1-\alpha)(1 - \tilde{S}_{aa}) \), then normalize \( \tilde{\psi} \).  
   Frame is **anomaly (1)** if \( \tilde{\psi} < \theta \), else normal (0).  
   \( S_{aa} \): similarity to true anomaly text; \( S_{an} \): to NVP-enhanced normal text.  
   Paper: \( \theta = 0.55 \) (UCF-Crime), \( 0.35 \) (XD-Violence); \( \alpha = 0.2 \).

5. **Temporal context self-adaptive learning (TCSAL)**  
   Transformer encoder where each head has a **learned attention span** \( z \) (Eq 7–9). Soft mask \( \chi_z(h) \) so attention decays by distance. \( R = 256 \). Output goes through a small classifier (LayerNorm → Linear → Sigmoid) for frame-level scores.

6. **Extra losses**  
   Applied to the **similarity vector** \( \tilde{S}_{aa} \) (not classifier output):  
   - **L_sp (smoothing):** \( \sum_j (\tilde{S}_{aa,j} - \tilde{S}_{aa,j+1})^2 \).  
   - **L_sm (sparsity):** \( \sum_j \tilde{S}_{aa,j} \).  
   Weights: \( \lambda_1 = 0.1 \), \( \lambda_2 = 0.01 \).

7. **Classification loss**  
   BCE between classifier frame scores and pseudo-labels: \( L_{cl} \) (Eq 13).

**Datasets:** UCF-Crime (AUC), XD-Violence (AP). Train on video-level labels; test with frame-level labels.

---

## Implementation Audit

### What’s done and aligned with the paper

| Component | Status | Notes |
|----------|--------|--------|
| **NVP** (`nvp.py`) | OK | Eq 3–4: similarity weights → aggregate → FFN + skip. Batch of normal videos is averaged to one NVP (reasonable). |
| **PLG** (`plg.py`) | OK | Eq 5–6, \( \alpha \), threshold. Anomaly when \( \psi_{\text{norm}} < \theta \). |
| **TCSAL** (`tcsal.py`) | OK | Adaptive span (Eq 7–8), causal attention, classifier head. \( R=256 \). |
| **Ranking losses** | OK | Normal and anomaly ranking losses match Eq 10–11. |
| **DIL** | OK | Cosine similarity between \( S_{aa} \) and \( S_{an} \) over frames (dim=1), then mean over batch. |
| **Temporal losses** | OK | Sparsity = smoothing in time (squared diff); Smoothness = mean (sparsity). **Important:** apply these to **\( \tilde{S}_{aa} \)** (similarity to true anomaly class), not classifier output. |
| **Dataset** | OK | Loads pre-extracted `.pt` features and class name; supports normal vs anomaly filter. |
| **Feature extraction** | OK | CLIP ViT-B/16, per-frame, normalized; script saves `.pt` per video. |

### Gaps and fixes

1. **CLIP / text encoder (`clip.py`)**  
   - **Issue:** Global `clip_model` and hardcoded device; not a proper nn.Module for the full pipeline.  
   - **Needed:** A single TPWNG model that owns CLIP, prompt, NVP, PLG, TCSAL and exposes image encoder + text encoder (with learnable prompt) for similarities and losses.

2. **PromptTextEncoder**  
   - Forward currently takes no inputs (uses registered buffers). Need to support batch of class indices or one embedding per class.  
   - Return shape: one vector per class (e.g. `(K, D)` for K classes).  
   - Class set: 13 anomaly + 1 normal (e.g. UCF-Crime). Map folder name → index; for each anomaly video, \( \tau \) = index of its class.

3. **NVP input in training**  
   - NVP expects `(B, F, D)` for normal videos and one `(D,)` normal text embedding. Your current `.mean(dim=0)` gives one global NVP for the batch. Paper defines \( Q_i \) per normal video; using one NVP per batch is a reasonable implementation choice. No change strictly required.

4. **Temporal losses – what to pass**  
   - Paper applies L_sp and L_sm to **\( \tilde{S}_{aa} \)** (normalized similarity to the **true** anomaly class for that video).  
   - In the training loop, for each anomaly video compute \( S_{aa} \) for the true class \( \tau \), normalize to get \( \tilde{S}_{aa} \), then compute `SparsityLoss(˜S_aa)` (smoothing) and `SmoothnessLoss(˜S_aa)` (sparsity).  
   - Naming: in your code, `SparsityLoss` is the temporal **smoothing** term, `SmoothnessLoss` is the **sparsity** term. Just use them with \( \tilde{S}_{aa} \) and the right \( \lambda_1, \lambda_2 \).

5. **No full TPWNG model**  
   - There is no `tpwng.py` that wires:  
     - Image features (from disk or from CLIP in memory).  
     - Text embeddings for all classes (learnable prompt + CLIP text encoder).  
     - NVP from normal batch → enhanced \( \dot{T}^n_k \).  
     - Similarities \( S_{nn}, S_{aa}, S_{an}, \phi_{na}, \phi_{aa} \) for losses and PLG.  
     - PLG → pseudo-labels.  
     - TCSAL + classifier → frame scores → \( L_{cl} \).  
   - This is the main missing piece.

6. **No training script**  
   - Need `scripts/train.py` that:  
     - Loads UCF-Crime (and optionally XD-Violence) from pre-extracted `.pt` + class from path.  
     - Builds batches (e.g. mixed normal + anomaly, or paired normal + anomaly).  
     - For each batch: compute all similarities, NVP, PLG, TCSAL scores, then all losses (ranking, DIL, L_sp, L_sm, L_cl).  
     - Backprop and optimizer step (prompt + text_projection + NVP + TCSAL).

7. **No evaluation script**  
   - Load test set with frame-level annotations (e.g. from `.txt`), run TCSAL classifier on test features, compute AUC (UCF-Crime) and AP (XD-Violence).

8. **DIL shape**  
   - `F.cosine_similarity(S_aa, S_an, dim=1)` with `(B, F)` gives `(B,)`. Paper Eq 12 has sum over \( i,j \) then divide by MF; your mean over batch is equivalent to (1/M)*mean over videos of per-video cosine. Paper uses 1/(MF); if you prefer exact scaling, you can multiply by (F/M) or average over (B,F) after expanding. Current choice is acceptable.

9. **XD-Violence**  
   - Dataset class is UCF-centric. Add XD-Violence folder layout and class list, and optionally a single dataset that supports both (e.g. by `dataset_name` or path layout).

---

## Suggested Next Steps (in order)

### 1. Implement full TPWNG model (`src/models/tpwng.py`)

- **Inputs:**  
  - Batch of normal video features `(B_n, F_n, D)`, batch of anomaly video features `(B_a, F_a, D)`, and for each anomaly video its true class index \( \tau \in \{0..K-2\} \) (K-1 anomaly classes; normal is last or separate).
- **Internals:**  
  - Load CLIP once; freeze image encoder (or use pre-extracted features and only train text side + NVP + TCSAL).  
  - Learnable prompt: 8 context vectors + class tokens for K classes → text encoder → \( E = \{ T^a_1..T^a_{K-1}, T^n_K \}\).  
  - NVP: from normal batch → \( Q \), then \( \dot{T}^n_k = \text{FFN}([T^n_k; Q]) + T^n_k \).  
  - Similarities:  
    - Normal batch: \( S_{nn} = X_n (\dot{T}^n_k)^T \) (or \( T^n_k \) before NVP for the part that computes NVP).  
    - Anomaly batch: \( S_{an} = X_a (\dot{T}^n_k)^T \), \( S_{aa}^{(\tau)} = X_a (T^a_\tau)^T \), and \( \phi_{na}, \phi_{aa} \) for other classes.  
  - PLG: from \( S_{aa}^{(\tau)} \) and \( S_{an} \) → pseudo-labels.  
  - TCSAL: from \( X_a \) (and optionally \( X_n \)) → frame scores.  
- **Outputs:**  
  - All similarity tensors and \( \dot{T}^n_k \) for losses; pseudo-labels; frame scores for \( L_{cl} \).

### 2. Refactor CLIP usage in `src/models/clip.py`

- Wrap in an nn.Module that:  
  - Takes class names and prompt length.  
  - Exposes `encode_text(class_indices)` or `get_text_embeddings()` → `(K, D)`.  
  - Optionally exposes `encode_image(frames)` for ablation or when not using pre-extracted features.  
- No global `clip_model`; device passed in or inferred from first tensor.

### 3. Add `scripts/train.py`

- Parse args: data root, feature dir, dataset (ucf / xd), batch sizes, epochs, lr, \( \alpha \), \( \theta \), \( \lambda_1 \), \( \lambda_2 \), checkpoint dir.  
- DataLoader: e.g. one dataset for normal, one for anomaly; or a combined sampler that yields (normal_batch, anomaly_batch) with anomaly class indices.  
- Forward: call TPWNG model to get similarities, enhanced normal text, pseudo-labels, scores.  
- Losses:  
  - From normal batch: \( L_n rank \) (using \( S_{nn} \), \( \phi_{na} \)).  
  - From anomaly batch: \( L_a rank \), \( L_{dil} \), L_sp and L_sm on \( \tilde{S}_{aa} \), and \( L_{cl} \) (scores vs pseudo-labels).  
- Optimizer: AdamW for prompt, text_projection, NVP, TCSAL.  
- Logging and saving checkpoints.

### 4. Add `scripts/evaluate.py`

- Load test split: list of (video feature path, frame-level label path or list).  
- Run TCSAL classifier (and optionally full model) on test features.  
- Compute frame-level ROC AUC for UCF-Crime and average precision for XD-Violence; report.

### 5. Data and config

- Document exact dir layout for UCF-Crime and XD-Violence (train/test, .pt and .txt).  
- Add a small config or constants for: class names, \( \theta \) per dataset (0.55 / 0.35), and feature length \( F \) handling (pad/truncate if needed).

### 6. Optional

- Weights & Biases (or similar) for metrics and curves.  
- Ablations: w/o NVP, w/o normality guidance, w/o TCSAL (replace with mean pool + MLP), and different temporal modules (as in paper Table 4).

---

## Quick reference: hyperparameters (paper)

| Param | UCF-Crime | XD-Violence |
|-------|-----------|-------------|
| \( \theta \) | 0.55 | 0.35 |
| \( \alpha \) | 0.2 | 0.2 |
| \( \lambda_1 \) (L_sp) | 0.1 | 0.1 |
| \( \lambda_2 \) (L_sm) | 0.01 | 0.01 |
| R (TCSAL) | 256 | 256 |
| l (prompt length) | 8 | 8 |

---

## Summary

- **Done:** NVP, PLG, TCSAL, ranking losses, DIL, temporal losses, dataset (UCF), feature extraction script.  
- **Critical next:** (1) Full TPWNG model in `tpwng.py` wiring CLIP/prompt, NVP, PLG, TCSAL and all losses; (2) refactor CLIP/prompt into a clean module; (3) training script; (4) evaluation script.  
- **Then:** Run on UCF-Crime (and XD-Violence), tune if needed, and add ablations to match paper tables.
