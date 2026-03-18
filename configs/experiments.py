"""
Experiment configurations to replicate all tables in the TPWNG paper (CVPR 2024).

- Table 1: Main results (full model on UCF-Crime and XD-Violence)
- Table 2: NVP ablation (w/o NVP, NVP-FA, NVP-AS)
- Table 3: Normality guidance ablation (w/o NG, w NG)
- Table 4: Temporal module ablation (TF-encoder, MTN, GL-MHSA, TCSAL)
- Table 5: Loss terms ablation (bs, +L_n_rank, +L_a_rank, +L_dil, all)
"""

from typing import Any

# Dataset class names and hyperparameters (paper §4.2)
UCF_CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing",
    "Vandalism", "Normal"
]
# For text encoder: anomaly first (indices 0..12), normal last (13)
UCF_CRIME_ANOMALY_CLASSES = UCF_CRIME_CLASSES[:-1]  # 13 anomaly
UCF_CRIME_NORMAL_NAME = "Normal"

XD_VIOLENCE_CLASSES = [
    "Abuse", "Arrest", "Explosion", "Fighting", "RoadAccidents", "Shooting", "Normal"
]
XD_VIOLENCE_ANOMALY_CLASSES = XD_VIOLENCE_CLASSES[:-1]
XD_VIOLENCE_NORMAL_NAME = "Normal"

DATASET_CONFIG = {
    "ucfcrime": {
        "classes": UCF_CRIME_CLASSES,
        "anomaly_classes": UCF_CRIME_ANOMALY_CLASSES,
        "normal_name": UCF_CRIME_NORMAL_NAME,
        "theta": 0.55,
        "metric": "AUC",
    },
    "xdviolence": {
        "classes": XD_VIOLENCE_CLASSES,
        "anomaly_classes": XD_VIOLENCE_ANOMALY_CLASSES,
        "normal_name": XD_VIOLENCE_NORMAL_NAME,
        "theta": 0.35,
        "metric": "AP",
    },
}

# Default training hyperparameters (paper §4.2 + author supp: arXiv-2404.08531v1 sec/X_suppl.tex)
# Supp: Adam, weight_decay=0.005, batch 64 (32 normal + 32 anomaly), UCF lr=0.001 50ep, XD lr=0.0001 20ep
DEFAULT_HP = {
    "alpha": 0.2,
    "lambda1": 0.1,   # L_sp (smoothing)
    "lambda2": 0.01,  # L_sm (sparsity)
    "n_ctx": 8,
    "R": 256.0,
    "feature_dim": 512,
    "lr": 1e-3,       # author supp: 0.001 for UCF-Crime
    "weight_decay": 0.005,  # author supp
    "epochs": 50,
    "batch_size_normal": 32,   # author supp: 32 normal + 32 abnormal = 64
    "batch_size_anomaly": 32,
    "max_frames": 256,  # VadCLIP visual_length; cap/pad for batching
    "scheduler_milestones": [30, 40],
    "scheduler_gamma": 0.1,
}


def get_experiment_config(experiment_id: str, dataset: str = "ucfcrime") -> dict[str, Any]:
    """
    Return full config for an experiment. Used by train.py and run_experiments.py.
    """
    ds_cfg = DATASET_CONFIG.get(dataset, DATASET_CONFIG["ucfcrime"])
    theta = ds_cfg["theta"]
    cfg = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        **DEFAULT_HP,
        "theta": theta,
        "classes": ds_cfg["classes"],
        "anomaly_classes": ds_cfg["anomaly_classes"],
        "normal_name": ds_cfg["normal_name"],
        "metric": ds_cfg["metric"],
        # Ablation flags (all False = full model)
        "use_nvp": True,
        "nvp_mode": "as",  # "as" (aggregate by similarity), "fa" (frame average), or "none"
        "use_normality_guidance": True,
        "temporal_module": "tcsal", # "tcsal" | "tf_encoder" | "mtn" | "gl_mhsa"
        "use_Ln_rank": True,
        "use_La_rank": True,
        "use_Ldil": True,
    }
    # Dataset-specific training (author supp: XD-Violence lr=0.0001, 20 epochs)
    if dataset == "xdviolence":
        cfg["lr"] = 1e-4
        cfg["epochs"] = 20

    # Apply experiment-specific overrides
    if experiment_id.startswith("main_"):
        pass  # full model already set
    elif experiment_id == "abl_nvp_none":
        cfg["use_nvp"] = False
        cfg["nvp_mode"] = "none"
    elif experiment_id == "abl_nvp_fa":
        cfg["use_nvp"] = True
        cfg["nvp_mode"] = "fa"
    elif experiment_id == "abl_nvp_as":
        cfg["use_nvp"] = True
        cfg["nvp_mode"] = "as"
    elif experiment_id == "abl_ng_none":
        cfg["use_normality_guidance"] = False
    elif experiment_id == "abl_ng_full":
        cfg["use_normality_guidance"] = True
    elif experiment_id == "abl_temporal_tf":
        cfg["temporal_module"] = "tf_encoder"
    elif experiment_id == "abl_temporal_mtn":
        cfg["temporal_module"] = "mtn"
    elif experiment_id == "abl_temporal_gl":
        cfg["temporal_module"] = "gl_mhsa"
    elif experiment_id == "abl_temporal_tcsal":
        cfg["temporal_module"] = "tcsal"
    elif experiment_id == "abl_loss_bs_only":
        cfg["use_Ln_rank"] = False
        cfg["use_La_rank"] = False
        cfg["use_Ldil"] = False
    elif experiment_id == "abl_loss_plus_Ln":
        cfg["use_La_rank"] = False
        cfg["use_Ldil"] = False
    elif experiment_id == "abl_loss_plus_La":
        cfg["use_Ln_rank"] = False
        cfg["use_Ldil"] = False
    elif experiment_id == "abl_loss_plus_Ldil":
        cfg["use_Ln_rank"] = False
        cfg["use_La_rank"] = False
    elif experiment_id == "abl_loss_all":
        pass  # all True
    return cfg


# Registry of all paper experiments for run_experiments.py
PAPER_EXPERIMENTS = [
    # Table 1: main (one per dataset)
    ("main_ucfcrime", "ucfcrime"),
    ("main_xdviolence", "xdviolence"),
    # Table 2: NVP
    ("abl_nvp_none", "ucfcrime"),
    ("abl_nvp_none", "xdviolence"),
    ("abl_nvp_fa", "ucfcrime"),
    ("abl_nvp_fa", "xdviolence"),
    ("abl_nvp_as", "ucfcrime"),
    ("abl_nvp_as", "xdviolence"),
    # Table 3: Normality guidance
    ("abl_ng_none", "ucfcrime"),
    ("abl_ng_none", "xdviolence"),
    ("abl_ng_full", "ucfcrime"),
    ("abl_ng_full", "xdviolence"),
    # Table 4: Temporal module
    ("abl_temporal_tf", "ucfcrime"),
    ("abl_temporal_tf", "xdviolence"),
    ("abl_temporal_mtn", "ucfcrime"),
    ("abl_temporal_mtn", "xdviolence"),
    ("abl_temporal_gl", "ucfcrime"),
    ("abl_temporal_gl", "xdviolence"),
    ("abl_temporal_tcsal", "ucfcrime"),
    ("abl_temporal_tcsal", "xdviolence"),
    # Table 5: Loss terms
    ("abl_loss_bs_only", "ucfcrime"),
    ("abl_loss_bs_only", "xdviolence"),
    ("abl_loss_plus_Ln", "ucfcrime"),
    ("abl_loss_plus_Ln", "xdviolence"),
    ("abl_loss_plus_La", "ucfcrime"),
    ("abl_loss_plus_La", "xdviolence"),
    ("abl_loss_plus_Ldil", "ucfcrime"),
    ("abl_loss_plus_Ldil", "xdviolence"),
    ("abl_loss_all", "ucfcrime"),
    ("abl_loss_all", "xdviolence"),
]
