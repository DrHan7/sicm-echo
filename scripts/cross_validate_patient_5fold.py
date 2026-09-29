
"""
SICM one-click pipeline:
raw A4C NPY -> LV segmentation -> ED-to-ED cycles -> patient-level 5-fold CV

label.csv must contain:
    id,label

One id = one patient.
Every fold starts from the original EchoJEPA checkpoint, not the previously
trained SICM final_model.pt.
"""

import copy
import gc
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, confusion_matrix,
    f1_score, precision_score, brier_score_loss,
)
from tqdm import tqdm


# ============================================================
# 1. PATHS
# ============================================================

def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return Path(value).expanduser()


RAW_NPY_ROOT = required_env_path("SICM_VIDEO_ROOT")
LABEL_CSV = required_env_path("SICM_LABEL_CSV")
CHECKPOINT_DIR = Path(os.environ.get("SICM_CHECKPOINT_DIR", "checkpoints")).expanduser()
ECHOJEPA_REPO_DIR = Path(
    os.environ.get("ECHOJEPA_REPO_DIR", CHECKPOINT_DIR / "EchoJEPA")
).expanduser()
ECHOJEPA_CHECKPOINT = Path(
    os.environ.get(
        "ECHOJEPA_CHECKPOINT",
        CHECKPOINT_DIR / "vitl-vmix22m-pt220-c55.pt",
    )
).expanduser()
LV_SEGMENTATION_CHECKPOINT = Path(
    os.environ.get(
        "LV_SEGMENTATION_CHECKPOINT",
        CHECKPOINT_DIR / "deeplabv3_resnet50_random.pt",
    )
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("SICM_CV_OUTPUT_DIR", "outputs/patient_level_5fold")
).expanduser()
PREPROCESS_DIR = OUTPUT_ROOT / "preprocessing"
CV_DIR = OUTPUT_ROOT / "cross_validation"

for path, description in [
    (RAW_NPY_ROOT, "video directory"),
    (LABEL_CSV, "label CSV"),
    (ECHOJEPA_REPO_DIR, "EchoJEPA repository"),
    (ECHOJEPA_CHECKPOINT, "EchoJEPA checkpoint"),
    (LV_SEGMENTATION_CHECKPOINT, "LV segmentation checkpoint"),
]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {description}:\n{path}")

for output_path in [OUTPUT_ROOT, PREPROCESS_DIR, CV_DIR]:
    output_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. RUN MODE
# ============================================================

# First completely fresh run from raw NPY:
FORCE_REPROCESS_CYCLES = True

# Quick test:
#   SSL=1, Stage A=2, Stage B=3
# Formal:
#   SSL=10, Stage A=10, Stage B=20
QUICK_TEST = False

# Formal 5-fold: None
# Single Fold 1: 1
RUN_ONLY_FOLD = None

N_SPLITS = 5

if QUICK_TEST:
    SSL_EPOCHS, STAGE_A_EPOCHS, STAGE_B_EPOCHS = 1, 2, 3
else:
    SSL_EPOCHS, STAGE_A_EPOCHS, STAGE_B_EPOCHS = 10, 10, 20

TOTAL_CLS_EPOCHS = STAGE_A_EPOCHS + STAGE_B_EPOCHS


# ============================================================
# 3. GENERAL SETTINGS
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type
AMP_ENABLED = DEVICE_TYPE == "cuda"
NUM_WORKERS = 0

SEG_SIZE = 112
SEG_BATCH_SIZE = 64
NORMALIZATION_SAMPLE_VIDEOS = 64
NORMALIZATION_FRAMES_PER_VIDEO = 16

SMOOTH_SIGMA = 1.2
MIN_ED_DISTANCE = 6
ED_PROMINENCE_FRACTION = 0.05
MIN_CYCLE_FRAMES = 6
MAX_CYCLE_FRAMES = 90

QC_MIN_AREA_EXCURSION_FRACTION = 0.06
QC_MAX_ED_AREA_MISMATCH_FRACTION = 0.60
SAVE_QC_PLOTS = True

NUM_FRAMES = 16
IMAGE_SIZE = 224
PATCH_SIZE = 16
TUBELET_SIZE = 2
FEATURE_DIM = 1024
TRANSFORMER_DEPTH = 24
TEMPORAL_TOKENS = NUM_FRAMES // TUBELET_SIZE
SPATIAL_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_TOKENS = TEMPORAL_TOKENS * SPATIAL_SIDE * SPATIAL_SIDE

ECHOJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
ECHOJEPA_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
USE_ACTIVATION_CHECKPOINTING = True

SSL_BATCH_SIZE = 1
SSL_ACCUMULATION_STEPS = 8
SSL_LR = 1e-5
SSL_WEIGHT_DECAY = 0.05
SSL_MASK_RATIO = 0.60
SSL_UNFREEZE_LAST_N_BLOCKS = 4
PREDICTOR_DIM = 256
PREDICTOR_DEPTH = 2
PREDICTOR_HEADS = 8
EMA_START = 0.996
EMA_END = 0.9999
CYCLE_SSL_LOSS_WEIGHT = 0.25
SSL_USE_UNMASKED_CONTEXT_FOR_CYCLE_LOSS = False

CLS_ACCUMULATION_STEPS = 4
TEMPORAL_HEADS = 8
TEMPORAL_DROPOUT = 0.10
BACKBONE_LR = 2e-6
TEMPORAL_LR = 2e-5
HEAD_LR = 1e-4
CLS_WEIGHT_DECAY = 1e-3
UNFREEZE_LAST_N_BLOCKS = 2
USE_CLASS_POS_WEIGHT = True

AREA_CURVE_POINTS = 64
AREA_EMBED_DIM = 128
CYCLE_FUSION_DIM = 512
MIL_ATTENTION_DIM = 128
TRAIN_MAX_CYCLES_PER_STUDY = 0
CYCLE_FORWARD_CHUNK_SIZE = 1
SAVE_VALIDATION_ATTENTION = True

AUG_BRIGHTNESS_PROB = 0.80
AUG_BRIGHTNESS_RANGE = (0.85, 1.15)
AUG_CONTRAST_PROB = 0.80
AUG_CONTRAST_RANGE = (0.85, 1.15)
AUG_GAMMA_PROB = 0.50
AUG_GAMMA_RANGE = (0.80, 1.25)
AUG_TRANSLATE_PROB = 0.50
AUG_MAX_TRANSLATE = 8
AUG_MAX_PHASE_ROLL = 0
AUG_PHASE_ROLL_PROB = 0.50


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

if str(ECHOJEPA_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(ECHOJEPA_REPO_DIR))

print("=" * 80)
print("SICM RAW NPY -> CYCLES -> PATIENT-LEVEL 5-FOLD CV")
print("=" * 80)
print("DEVICE        :", DEVICE)
print("RAW_NPY_ROOT  :", RAW_NPY_ROOT)
print("OUTPUT_ROOT   :", OUTPUT_ROOT)
print("QUICK_TEST    :", QUICK_TEST)
print("RUN_ONLY_FOLD :", RUN_ONLY_FOLD)
if DEVICE_TYPE == "cuda":
    print("GPU           :", torch.cuda.get_device_name(0))


# ============================================================
# 4. RAW DATA / LABELS
# ============================================================

EXCLUDED = {
    "outcome", "outputs", "cycles", "preprocessing", "cross_validation",
    "echojepa", "base", "__pycache__", OUTPUT_ROOT.name.lower(),
}


def excluded_path(path):
    names = {p.lower() for p in path.parts}
    return any(x in names for x in EXCLUDED)


def scan_raw_videos(root, expected_ids):
    """Find one exact <ID>.npy for each labeled ID; ignore all other files."""
    paths_by_id = {str(sample_id).strip(): [] for sample_id in expected_ids}

    for path in sorted(root.rglob("*.npy")):
        if excluded_path(path) or path.name.endswith(".tmp.npy"):
            continue
        sample_id = path.stem.strip()
        if sample_id in paths_by_id:
            paths_by_id[sample_id].append(path)

    duplicates = {
        sample_id: paths
        for sample_id, paths in paths_by_id.items()
        if len(paths) > 1
    }
    if duplicates:
        details = "\n".join(
            f"{sample_id}: " + ", ".join(map(str, paths))
            for sample_id, paths in duplicates.items()
        )
        raise RuntimeError(f"More than one exact ID-named NPY was found:\n{details}")

    records = [
        {"id": sample_id, "video_path": str(paths[0])}
        for sample_id, paths in sorted(paths_by_id.items())
        if len(paths) == 1
    ]
    return pd.DataFrame(records, columns=["id", "video_path"])


if not LABEL_CSV.exists():
    raise FileNotFoundError(f"Cannot find:\n{LABEL_CSV}")

label_df = pd.read_csv(LABEL_CSV, dtype=str, encoding="utf-8-sig")
label_df.columns = [str(c).strip().lower() for c in label_df.columns]
if len(label_df.columns) != 2 or set(label_df.columns) != {"id", "label"}:
    raise RuntimeError("label.csv must contain exactly two columns named: id,label")
label_df = label_df[["id", "label"]].copy()
label_df["id"] = label_df["id"].astype(str).str.strip()
label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce")
if label_df["id"].eq("").any():
    raise RuntimeError("label.csv contains an empty ID value.")

conflicts = (
    label_df[label_df["label"].isin([0, 1])]
    .groupby("id")["label"]
    .nunique()
)
if (conflicts > 1).any():
    raise RuntimeError("Some IDs contain conflicting labels.")
label_df = label_df.drop_duplicates("id").copy()
if not label_df["label"].isin([0, 1]).all():
    raise RuntimeError("Every label must be 0 or 1.")
label_df["label"] = label_df["label"].astype(int)

video_df = scan_raw_videos(RAW_NPY_ROOT, label_df["id"].tolist())
missing_ids = sorted(set(label_df["id"]) - set(video_df["id"]))
if missing_ids:
    missing_path = PREPROCESS_DIR / "ids_without_matching_npy.csv"
    pd.DataFrame({"id": missing_ids}).to_csv(
        missing_path, index=False, encoding="utf-8-sig"
    )
    raise RuntimeError(
        f"{len(missing_ids)} labeled IDs have no matching <id>.npy. "
        f"Details: {missing_path}"
    )

full_df = video_df.merge(label_df[["id", "label"]], on="id", how="inner")

print("Raw videos:", len(full_df))
print(full_df["label"].value_counts().sort_index())


# ============================================================
# 5. RAW VIDEO HELPERS
# ============================================================

def load_video_tchw(path):
    arr = np.asarray(np.load(path, mmap_mode="r"))

    if arr.ndim == 3:
        x = torch.from_numpy(arr.copy()).float().unsqueeze(1)
    elif arr.ndim == 4 and arr.shape[-1] in (1, 3):
        x = torch.from_numpy(arr.copy()).float().permute(0, 3, 1, 2)
    elif arr.ndim == 4 and arr.shape[1] in (1, 3):
        x = torch.from_numpy(arr.copy()).float()
    else:
        raise ValueError(f"Unsupported video shape {arr.shape}: {path}")

    return x


def to_three_channel_255(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError("Video must have 1 or 3 channels.")

    vmin, vmax = float(x.min()), float(x.max())
    if vmin >= 0 and vmax <= 1.5:
        x = x * 255.0
    elif vmin < 0 or vmax > 255:
        x = (x - vmin) / max(vmax - vmin, 1e-6) * 255.0

    return x


def to_grayscale_255(x):
    if x.shape[1] == 3:
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    vmin, vmax = float(x.min()), float(x.max())
    if vmin >= 0 and vmax <= 1.5:
        x = x * 255.0
    elif vmin < 0 or vmax > 255:
        x = (x - vmin) / max(vmax - vmin, 1e-6) * 255.0

    return x.clamp(0, 255)


def estimate_seg_normalization(df, save_path=None):
    if df.empty:
        raise RuntimeError(
            "Cannot estimate segmentation normalization from an empty training fold."
        )

    selected = (
        df if len(df) <= NORMALIZATION_SAMPLE_VIDEOS
        else df.sample(NORMALIZATION_SAMPLE_VIDEOS, random_state=SEED)
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Seg normalization"):
        x = load_video_tchw(row["video_path"])
        if x.shape[0] > NORMALIZATION_FRAMES_PER_VIDEO:
            idx = np.linspace(
                0, x.shape[0] - 1, NORMALIZATION_FRAMES_PER_VIDEO
            ).round().astype(int)
            x = x[idx]

        x = to_three_channel_255(x)
        x = F.interpolate(
            x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False
        )

        flat = x.permute(1, 0, 2, 3).reshape(3, -1).double()
        channel_sum += flat.sum(1)
        channel_sq_sum += (flat * flat).sum(1)
        total_pixels += flat.shape[1]

    if total_pixels == 0:
        raise RuntimeError("No pixels were available for segmentation normalization.")

    mean = (channel_sum / total_pixels).float()
    var = (channel_sq_sum / total_pixels - mean.double() ** 2).clamp_min(1e-8)
    std = torch.sqrt(var).float()

    if save_path is not None:
        pd.DataFrame(
            {"channel": ["R", "G", "B"], "mean": mean.numpy(), "std": std.numpy()}
        ).to_csv(save_path, index=False)

    return mean, std


# ============================================================
# 6. LV SEGMENTATION
# ============================================================

def build_lv_segmenter(path):
    try:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None, weights_backbone=None, aux_loss=False
        )
    except TypeError:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=False, aux_loss=False
        )

    old_last = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(
        old_last.in_channels, 1, kernel_size=old_last.kernel_size
    )

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "LV segmentation checkpoint mismatch.\n"
            f"Missing: {result.missing_keys[:10]}\n"
            f"Unexpected: {result.unexpected_keys[:10]}"
        )

    return model.to(DEVICE).eval()


@torch.no_grad()
def lv_area_curve(path, model, mean, std):
    original = load_video_tchw(path)
    x = to_three_channel_255(original.clone())
    x = F.interpolate(
        x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False
    )
    x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    areas = []
    for start in range(0, len(x), SEG_BATCH_SIZE):
        batch = x[start:start + SEG_BATCH_SIZE].to(DEVICE)
        logits = model(batch)["out"][:, 0]
        areas.append(
            (logits > 0).sum((1, 2)).cpu().numpy().astype(np.float32)
        )

    return original, np.concatenate(areas)


# ============================================================
# 7. CARDIAC CYCLE DETECTION
# ============================================================

def robust_range(x):
    return max(float(np.quantile(x, 0.95) - np.quantile(x, 0.05)), 1.0)


def ed_peaks(smooth, area_range, distance, prominence):
    peaks, _ = find_peaks(
        smooth, distance=distance, prominence=prominence * area_range
    )
    return peaks


def make_candidate(smooth, area_range, ed1, ed2, method):
    ed1, ed2 = sorted([int(ed1), int(ed2)])
    period = ed2 - ed1
    if period < 1:
        return None

    es = ed1 + int(np.argmin(smooth[ed1:ed2 + 1]))
    mean_ed = (smooth[ed1] + smooth[ed2]) / 2.0
    es_area = float(smooth[es])

    excursion = float((mean_ed - es_area) / area_range)
    mismatch = float(abs(smooth[ed1] - smooth[ed2]) / area_range)

    return {
        "method": method,
        "ed1": ed1,
        "es": es,
        "ed2": ed2,
        "cycle_frames": period,
        "excursion_fraction": excursion,
        "ed_area_mismatch_fraction": mismatch,
        "quality_score": excursion - 0.25 * mismatch,
        "qc_excursion_low": excursion < QC_MIN_AREA_EXCURSION_FRACTION,
        "qc_ed_mismatch_high": mismatch > QC_MAX_ED_AREA_MISMATCH_FRACTION,
        "qc_cycle_length_outside_preferred":
            period < MIN_CYCLE_FRAMES or period > MAX_CYCLE_FRAMES,
    }


def detect_cycles(raw_area):
    raw_area = np.asarray(raw_area, dtype=np.float32)
    if len(raw_area) < 2:
        raise RuntimeError("Video has fewer than 2 frames.")

    smooth = gaussian_filter1d(raw_area, sigma=SMOOTH_SIGMA, mode="nearest")
    area_range = robust_range(smooth)
    peaks = ed_peaks(
        smooth, area_range, MIN_ED_DISTANCE, ED_PROMINENCE_FRACTION
    )

    if len(peaks) < 2:
        raise RuntimeError(
            "Primary ED-peak detection found fewer than two peaks; "
            "cardiac-cycle detection failed."
        )

    cycles = []
    for ed1, ed2 in zip(peaks[:-1], peaks[1:]):
        candidate = make_candidate(
            smooth, area_range, ed1, ed2, "primary_peaks"
        )
        if candidate is not None:
            cycles.append(candidate)

    if not cycles:
        raise RuntimeError(
            "Primary ED-peak detection did not produce a valid ED-to-ED cycle; "
            "cardiac-cycle detection failed."
        )

    return {
        "cycles": cycles,
        "smooth_area": smooth,
        "detection_method": "primary_peaks",
    }


def resample_cycle(original, ed1, ed2):
    ed1, ed2 = sorted([int(ed1), int(ed2)])
    cycle = original[ed1:ed2 + 1]

    if len(cycle) < 2:
        raise RuntimeError("Cycle contains fewer than 2 frames.")

    cycle = to_grayscale_255(cycle)
    cycle = cycle.permute(1, 0, 2, 3).unsqueeze(0)

    cycle = F.interpolate(
        cycle,
        size=(NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
        mode="trilinear",
        align_corners=False,
    )

    return (
        cycle.squeeze(0).squeeze(0).round().clamp(0, 255)
        .to(torch.uint8).cpu().numpy()
    )


def build_area_representation(smooth, ed1, es, ed2):
    ed1, ed2 = sorted([int(ed1), int(ed2)])
    segment = np.asarray(smooth[ed1:ed2 + 1], dtype=np.float32)

    mean_ed = max(float((segment[0] + segment[-1]) / 2.0), 1e-6)
    normalized = segment / mean_ed

    curve = np.interp(
        np.linspace(0, 1, AREA_CURVE_POINTS),
        np.linspace(0, 1, len(normalized)),
        normalized,
    ).astype(np.float32)
    curve = np.clip(curve, 0, 3)

    es_local = int(np.argmin(segment))
    es_area = float(segment[es_local])

    fac = float((mean_ed - es_area) / mean_ed)
    systolic_fraction = max(es_local / max(len(segment) - 1, 1), 1e-3)
    diastolic_fraction = max(
        (len(segment) - 1 - es_local) / max(len(segment) - 1, 1), 1e-3
    )

    physiology = np.asarray(
        [
            fac,
            float((segment[0] - es_area) / mean_ed / systolic_fraction),
            float((segment[-1] - es_area) / mean_ed / diastolic_fraction),
            float(min((len(segment) - 1) / max(MAX_CYCLE_FRAMES, 1), 2.0)),
            float(abs(segment[0] - segment[-1]) / mean_ed),
        ],
        dtype=np.float32,
    )

    return curve, physiology


def save_qc(folder, raw_area, detection):
    if not SAVE_QC_PLOTS:
        return

    smooth = detection["smooth_area"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(raw_area, linewidth=0.8, alpha=0.5, label="Raw")
    ax.plot(smooth, linewidth=1.3, label="Smoothed")

    for i, c in enumerate(detection["cycles"], start=1):
        ax.scatter(
            [c["ed1"], c["es"], c["ed2"]],
            [smooth[c["ed1"]], smooth[c["es"]], smooth[c["ed2"]]],
            s=18,
        )
        ax.text(c["ed1"], smooth[c["ed1"]], f"C{i}", fontsize=7)

    ax.set_xlabel("Frame")
    ax.set_ylabel("LV segmented area")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "lv_area_multicycle_qc.png", dpi=180)
    plt.close(fig)


# ============================================================
# 8. FRESH PREPROCESSING CACHE
# ============================================================

MANIFEST_COLUMNS = [
    "id", "video_path", "status", "cycle_index", "n_cycles_in_study",
    "method", "original_frames", "ed1", "es", "ed2", "cycle_frames",
    "excursion_fraction", "ed_area_mismatch_fraction", "quality_score",
    "qc_excursion_low", "qc_ed_mismatch_high",
    "qc_cycle_length_outside_preferred", "fac", "systolic_slope",
    "diastolic_slope", "cycle_length_proxy",
    "ed_endpoint_mismatch_relative", "cycle_path", "area_curve_path", "error",
]


def preprocess_all(df, normalization_fit_df, fold_dir):
    """Extract cycles using normalization fitted only on this fold's train videos."""
    preprocessing_dir = fold_dir / "preprocessing"
    cycle_root = preprocessing_dir / "cycles"
    cycle_manifest = preprocessing_dir / "multi_cycle_manifest.csv"
    normalization_path = preprocessing_dir / "segmentation_input_normalization.csv"
    normalization_ids_path = (
        preprocessing_dir / "segmentation_normalization_training_ids.csv"
    )
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    target_ids = set(df["id"].astype(str))
    fit_ids = set(normalization_fit_df["id"].astype(str))
    if not fit_ids:
        raise RuntimeError("The fold has no training videos for normalization.")
    if not fit_ids.issubset(target_ids):
        raise RuntimeError("Normalization fit videos must belong to this fold's data.")
    if df["id"].astype(str).duplicated().any():
        raise RuntimeError("Cycle preprocessing expects one raw video per patient.")

    selected_fit_df = (
        normalization_fit_df
        if len(normalization_fit_df) <= NORMALIZATION_SAMPLE_VIDEOS
        else normalization_fit_df.sample(
            NORMALIZATION_SAMPLE_VIDEOS, random_state=SEED
        )
    )
    fit_id_order = selected_fit_df["id"].astype(str).tolist()
    mean, std = estimate_seg_normalization(normalization_fit_df)

    if not FORCE_REPROCESS_CYCLES and all(
        path.exists()
        for path in [cycle_manifest, normalization_path, normalization_ids_path]
    ):
        try:
            cached_manifest = pd.read_csv(cycle_manifest, dtype={"id": str})
            cached_norm = pd.read_csv(normalization_path).set_index("channel")
            cached_fit_ids = pd.read_csv(
                normalization_ids_path, dtype={"id": str}
            )["id"].tolist()
            cached_mean = cached_norm.loc[["R", "G", "B"], "mean"].to_numpy(float)
            cached_std = cached_norm.loc[["R", "G", "B"], "std"].to_numpy(float)
            used = cached_manifest[cached_manifest["status"] == "USED"]
            cached_ids_match = set(cached_manifest["id"].astype(str)) == target_ids
            cache_files_exist = len(used) > 0 and all(
                Path(row["cycle_path"]).exists()
                and Path(row["area_curve_path"]).exists()
                for _, row in used.iterrows()
            )
            cache_normalization_matches = (
                cached_fit_ids == fit_id_order
                and np.allclose(cached_mean, mean.numpy(), rtol=0, atol=1e-6)
                and np.allclose(cached_std, std.numpy(), rtol=0, atol=1e-6)
            )
            if cached_ids_match and cache_files_exist and cache_normalization_matches:
                print(f"Reusing fold {fold_dir.name} cycle cache.")
                return cached_manifest
        except (KeyError, ValueError, OSError, pd.errors.ParserError):
            pass

    if cycle_root.exists():
        shutil.rmtree(cycle_root)
    cycle_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"channel": ["R", "G", "B"], "mean": mean.numpy(), "std": std.numpy()}
    ).to_csv(normalization_path, index=False)
    pd.DataFrame({"id": fit_id_order}).to_csv(
        normalization_ids_path, index=False
    )

    segmenter = build_lv_segmenter(LV_SEGMENTATION_CHECKPOINT)

    records = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="Fresh cycle extraction")
    ):
        sample_id = str(row["id"])
        video_path = str(row["video_path"])
        folder = cycle_root / sample_id
        folder.mkdir(parents=True, exist_ok=True)

        try:
            original, raw_area = lv_area_curve(
                video_path, segmenter, mean, std
            )
            detection = detect_cycles(raw_area)
            save_qc(folder, raw_area, detection)

            n_cycles = len(detection["cycles"])

            for cycle_index, meta in enumerate(detection["cycles"], start=1):
                cycle = resample_cycle(original, meta["ed1"], meta["ed2"])
                curve, phys = build_area_representation(
                    detection["smooth_area"],
                    meta["ed1"],
                    meta["es"],
                    meta["ed2"],
                )

                cycle_path = (
                    folder / f"cycle_{cycle_index:02d}_{NUM_FRAMES}f_{IMAGE_SIZE}px.npy"
                )
                area_path = (
                    folder / f"lv_area_cycle_{cycle_index:02d}_{AREA_CURVE_POINTS}p.npy"
                )

                np.save(cycle_path, cycle)
                np.save(area_path, curve)

                records.append(
                    {
                        "id": sample_id,
                        "video_path": video_path,
                        "status": "USED",
                        "cycle_index": cycle_index,
                        "n_cycles_in_study": n_cycles,
                        "method": meta["method"],
                        "original_frames": int(original.shape[0]),
                        "ed1": meta["ed1"],
                        "es": meta["es"],
                        "ed2": meta["ed2"],
                        "cycle_frames": meta["cycle_frames"],
                        "excursion_fraction": meta["excursion_fraction"],
                        "ed_area_mismatch_fraction": meta["ed_area_mismatch_fraction"],
                        "quality_score": meta["quality_score"],
                        "qc_excursion_low": meta["qc_excursion_low"],
                        "qc_ed_mismatch_high": meta["qc_ed_mismatch_high"],
                        "qc_cycle_length_outside_preferred":
                            meta["qc_cycle_length_outside_preferred"],
                        "fac": float(phys[0]),
                        "systolic_slope": float(phys[1]),
                        "diastolic_slope": float(phys[2]),
                        "cycle_length_proxy": float(phys[3]),
                        "ed_endpoint_mismatch_relative": float(phys[4]),
                        "cycle_path": str(cycle_path),
                        "area_curve_path": str(area_path),
                        "error": "",
                    }
                )

        except Exception as exc:
            rec = {c: np.nan for c in MANIFEST_COLUMNS}
            rec.update(
                {
                    "id": sample_id,
                    "video_path": video_path,
                    "status": "ERROR",
                    "cycle_index": -1,
                    "method": "",
                    "cycle_path": "",
                    "area_curve_path": "",
                    "error": repr(exc),
                }
            )
            records.append(rec)

        if (i + 1) % 25 == 0:
            pd.DataFrame(records, columns=MANIFEST_COLUMNS).to_csv(
                cycle_manifest, index=False
            )

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    manifest.to_csv(cycle_manifest, index=False)

    del segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return manifest


patient_df = (
    full_df[["id", "label"]]
    .drop_duplicates("id")
    .reset_index(drop=True)
)

print("\nPatients entering fold assignment:", len(patient_df))
print(patient_df["label"].value_counts().sort_index())


# ============================================================
# 9. CACHED CYCLE / AUGMENTATION
# ============================================================

def load_cached_cycle(path):
    arr = np.load(path)

    if arr.ndim == 3:
        x = torch.from_numpy(arr.copy()).float().unsqueeze(1).repeat(1, 3, 1, 1)
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        x = torch.from_numpy(arr.copy()).float().permute(0, 3, 1, 2)
    elif arr.ndim == 4 and arr.shape[1] == 3:
        x = torch.from_numpy(arr.copy()).float()
    else:
        raise ValueError(f"Unsupported cached cycle shape: {arr.shape}")

    if arr.dtype == np.uint8 or float(x.max()) > 1.5:
        x /= 255.0

    return x.clamp(0, 1)


def random_translate(cycle, max_shift):
    dy = random.randint(-max_shift, max_shift)
    dx = random.randint(-max_shift, max_shift)

    padded = F.pad(
        cycle, (max_shift, max_shift, max_shift, max_shift), value=0.0
    )
    top = max_shift + dy
    left = max_shift + dx

    return padded[
        ...,
        top:top + cycle.shape[-2],
        left:left + cycle.shape[-1],
    ]


def augment_cycle(cycle):
    if AUG_MAX_PHASE_ROLL > 0 and random.random() < AUG_PHASE_ROLL_PROB:
        cycle = torch.roll(
            cycle,
            shifts=random.randint(-AUG_MAX_PHASE_ROLL, AUG_MAX_PHASE_ROLL),
            dims=0,
        )

    if AUG_MAX_TRANSLATE > 0 and random.random() < AUG_TRANSLATE_PROB:
        cycle = random_translate(cycle, AUG_MAX_TRANSLATE)

    if random.random() < AUG_BRIGHTNESS_PROB:
        cycle = (cycle * random.uniform(*AUG_BRIGHTNESS_RANGE)).clamp(0, 1)

    if random.random() < AUG_CONTRAST_PROB:
        factor = random.uniform(*AUG_CONTRAST_RANGE)
        mean = cycle.mean()
        cycle = ((cycle - mean) * factor + mean).clamp(0, 1)

    if random.random() < AUG_GAMMA_PROB:
        cycle = cycle.clamp(1e-6, 1).pow(
            random.uniform(*AUG_GAMMA_RANGE)
        )

    return cycle


def prepare_echojepa_input(cycle):
    cycle = (
        cycle - ECHOJEPA_MEAN.view(1, 3, 1, 1)
    ) / ECHOJEPA_STD.view(1, 3, 1, 1)

    return cycle.permute(1, 0, 2, 3).contiguous()


def load_area_curve(path):
    x = np.asarray(np.load(path), dtype=np.float32)
    if x.shape != (AREA_CURVE_POINTS,):
        raise ValueError(f"Bad LV area curve shape: {x.shape}")
    return torch.from_numpy(x.copy()).float()


# ============================================================
# 10. DATASETS
# ============================================================

class CycleSSLDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.id_to_indices = {
            str(k): g.index.tolist()
            for k, g in self.df.groupby("id", sort=False)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        anchor = load_cached_cycle(row["cycle_path"])

        view_context = prepare_echojepa_input(
            augment_cycle(anchor.clone())
        )
        view_token_target = prepare_echojepa_input(
            augment_cycle(anchor.clone())
        )

        candidates = [
            i for i in self.id_to_indices[sample_id] if i != idx
        ]

        if candidates:
            target_row = self.df.iloc[random.choice(candidates)]
            target = load_cached_cycle(target_row["cycle_path"])
            has_cross = True
        else:
            target = anchor.clone()
            has_cross = False

        return {
            "view_context": view_context,
            "view_token_target": view_token_target,
            "view_cycle_target": prepare_echojepa_input(
                augment_cycle(target)
            ),
            "has_cross_cycle": torch.tensor(has_cross, dtype=torch.bool),
        }


class SICMMILStudyDataset(Dataset):
    def __init__(self, df, train):
        self.train = train
        self.groups = []

        for sample_id, group in df.groupby("id", sort=False):
            group = group.sort_values("cycle_index").reset_index(drop=True)
            labels = group["label"].astype(int).unique()

            if len(labels) != 1:
                raise RuntimeError(f"Inconsistent labels for ID {sample_id}")

            self.groups.append((str(sample_id), int(labels[0]), group))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        sample_id, label, group = self.groups[idx]

        if (
            self.train
            and TRAIN_MAX_CYCLES_PER_STUDY > 0
            and len(group) > TRAIN_MAX_CYCLES_PER_STUDY
        ):
            ii = np.sort(
                np.random.choice(
                    len(group), TRAIN_MAX_CYCLES_PER_STUDY, replace=False
                )
            )
            group = group.iloc[ii].reset_index(drop=True)

        cycles, curves, phys = [], [], []
        cycle_indices = []

        for _, row in group.iterrows():
            cycle = load_cached_cycle(row["cycle_path"])
            if self.train:
                cycle = augment_cycle(cycle)

            cycles.append(prepare_echojepa_input(cycle))
            curves.append(load_area_curve(row["area_curve_path"]))

            phys.append(
                torch.tensor(
                    [
                        row["fac"],
                        row["systolic_slope"],
                        row["diastolic_slope"],
                        row["cycle_length_proxy"],
                        row["ed_endpoint_mismatch_relative"],
                    ],
                    dtype=torch.float32,
                )
            )
            cycle_indices.append(int(row["cycle_index"]))

        return {
            "id": sample_id,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "cycles": torch.stack(cycles),
            "area_curves": torch.stack(curves),
            "physiology": torch.stack(phys),
            "cycle_indices": cycle_indices,
        }


def mil_collate(batch):
    if len(batch) != 1:
        raise RuntimeError("MIL batch size must be 1.")
    return batch[0]


# ============================================================
# 11. EchoJEPA MODEL LOADER
# ============================================================

def choose_encoder_state(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt, "root"

    for key in [
        "target_encoder", "ema_encoder", "encoder",
        "backbone", "model", "state_dict",
    ]:
        if isinstance(ckpt.get(key), dict):
            return ckpt[key], key

    return ckpt, "root"


def clean_key(key):
    prefixes = [
        "module.", "backbone.", "encoder.",
        "target_encoder.", "ema_encoder.",
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True

    return key


def load_echojepa():
    from src.models import vision_transformer as vit_encoder

    encoder = vit_encoder.vit_large(
        patch_size=PATCH_SIZE,
        img_size=(IMAGE_SIZE, IMAGE_SIZE),
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        use_activation_checkpointing=USE_ACTIVATION_CHECKPOINTING,
        handle_nonsquare_inputs=True,
    )

    if (
        encoder.embed_dim != FEATURE_DIM
        or len(encoder.blocks) != TRANSFORMER_DEPTH
        or encoder.num_patches != NUM_TOKENS
    ):
        raise RuntimeError("EchoJEPA architecture mismatch.")

    try:
        ckpt = torch.load(
            ECHOJEPA_CHECKPOINT, map_location="cpu", weights_only=False
        )
    except TypeError:
        ckpt = torch.load(ECHOJEPA_CHECKPOINT, map_location="cpu")

    raw_state, selected_key = choose_encoder_state(ckpt)
    cleaned = {
        clean_key(k): v
        for k, v in raw_state.items()
        if torch.is_tensor(v)
    }

    model_state = encoder.state_dict()
    matched = {
        k: v
        for k, v in cleaned.items()
        if k in model_state and model_state[k].shape == v.shape
    }

    matched_numel = sum(v.numel() for v in matched.values())
    total_numel = sum(v.numel() for v in model_state.values())
    ratio = matched_numel / total_numel

    print("Checkpoint key:", selected_key)
    print("EchoJEPA match:", f"{ratio * 100:.2f}%")

    if ratio < 0.95:
        raise RuntimeError("EchoJEPA checkpoint match <95%.")

    encoder.load_state_dict(matched, strict=False)
    return encoder


# ============================================================
# 12. ATTENTION / SSL
# ============================================================

def token_grid(tokens):
    return tokens.reshape(
        tokens.shape[0],
        TEMPORAL_TOKENS,
        SPATIAL_SIDE * SPATIAL_SIDE,
        FEATURE_DIM,
    )


class SpatialAttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, TEMPORAL_TOKENS, 1, FEATURE_DIM)
        )
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.score = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        z = self.norm(x)
        scores = self.score(
            z + self.temporal_pos.to(z.device, z.dtype)
        )
        w = torch.softmax(scores, dim=2)
        return torch.sum(x * w, dim=2), w.squeeze(-1)


class TemporalAttentionPool(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm0 = nn.LayerNorm(FEATURE_DIM)
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, TEMPORAL_TOKENS, FEATURE_DIM)
        )
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.attn = nn.MultiheadAttention(
            FEATURE_DIM,
            TEMPORAL_HEADS,
            dropout=TEMPORAL_DROPOUT,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(FEATURE_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM * 2),
            nn.GELU(),
            nn.Dropout(TEMPORAL_DROPOUT),
            nn.Linear(FEATURE_DIM * 2, FEATURE_DIM),
        )
        self.norm2 = nn.LayerNorm(FEATURE_DIM)

        self.score = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.norm0(x)
        x = x + self.temporal_pos.to(x.device, x.dtype)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        w = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * w, dim=1), w.squeeze(-1)


def create_mask(batch_size):
    n_mask = int(NUM_TOKENS * SSL_MASK_RATIO)
    order = torch.argsort(
        torch.rand(batch_size, NUM_TOKENS, device=DEVICE), dim=1
    )

    mask = torch.zeros(
        batch_size, NUM_TOKENS, dtype=torch.bool, device=DEVICE
    )
    mask.scatter_(1, order[:, :n_mask], True)
    return mask


def apply_mask(clip, mask):
    grid = mask.reshape(
        clip.shape[0], 1, TEMPORAL_TOKENS, SPATIAL_SIDE, SPATIAL_SIDE
    ).to(clip.dtype)

    grid = F.interpolate(
        grid,
        size=(NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )

    return clip * (1 - grid)


class LatentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(FEATURE_DIM, PREDICTOR_DIM)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, PREDICTOR_DIM)
        )
        self.pos = nn.Parameter(
            torch.zeros(1, NUM_TOKENS, PREDICTOR_DIM)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=PREDICTOR_DIM,
            nhead=PREDICTOR_HEADS,
            dim_feedforward=PREDICTOR_DIM * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=PREDICTOR_DEPTH
        )
        self.out_proj = nn.Linear(PREDICTOR_DIM, FEATURE_DIM)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, context, mask):
        x = self.in_proj(context)
        mt = self.mask_token.to(x.device, x.dtype).expand_as(x)
        x = torch.where(mask.unsqueeze(-1), mt, x)
        x = x + self.pos.to(x.device, x.dtype)
        return self.out_proj(self.transformer(x))


def configure_ssl_blocks(model):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks[-SSL_UNFREEZE_LAST_N_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    for p in model.norm.parameters():
        p.requires_grad = True


class CycleAwareEchoJEPA(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.context_backbone = backbone
        configure_ssl_blocks(self.context_backbone)

        self.context_spatial_pool = SpatialAttentionPool()
        self.context_temporal_pool = TemporalAttentionPool()

        self.target_backbone = copy.deepcopy(backbone)
        self.target_spatial_pool = copy.deepcopy(self.context_spatial_pool)
        self.target_temporal_pool = copy.deepcopy(self.context_temporal_pool)

        for module in [
            self.target_backbone,
            self.target_spatial_pool,
            self.target_temporal_pool,
        ]:
            for p in module.parameters():
                p.requires_grad = False

        self.predictor = LatentPredictor()

    def pooled(self, tokens, spatial_pool, temporal_pool):
        temporal_features, _ = spatial_pool(token_grid(tokens))
        feature, _ = temporal_pool(temporal_features)
        return feature

    def forward(
        self,
        view_context,
        view_token_target,
        view_cycle_target,
        has_cross_cycle,
        mask,
    ):
        context_tokens = self.context_backbone(
            apply_mask(view_context, mask)
        )

        predicted = self.predictor(context_tokens, mask)

        with torch.no_grad():
            token_target = self.target_backbone(view_token_target)

        pred_masked = F.normalize(predicted[mask].float(), dim=-1)
        target_masked = F.normalize(
            token_target[mask].float().detach(), dim=-1
        )

        prediction_loss = F.smooth_l1_loss(
            pred_masked, target_masked
        )

        has_cross_cycle = has_cross_cycle.to(DEVICE).bool().view(-1)
        valid_count = int(has_cross_cycle.sum().item())

        if CYCLE_SSL_LOSS_WEIGHT > 0 and valid_count > 0:
            source_tokens = (
                self.context_backbone(view_context)
                if SSL_USE_UNMASKED_CONTEXT_FOR_CYCLE_LOSS
                else context_tokens
            )

            source_feature = F.normalize(
                self.pooled(
                    source_tokens,
                    self.context_spatial_pool,
                    self.context_temporal_pool,
                ).float(),
                dim=-1,
            )

            with torch.no_grad():
                target_feature = F.normalize(
                    self.pooled(
                        self.target_backbone(view_cycle_target),
                        self.target_spatial_pool,
                        self.target_temporal_pool,
                    ).float(),
                    dim=-1,
                )

            cosine = (source_feature * target_feature).sum(-1)
            valid_cosine = cosine[has_cross_cycle]
            cycle_loss = (1 - valid_cosine).mean()
            mean_cosine = valid_cosine.mean()
        else:
            cycle_loss = torch.zeros((), device=DEVICE)
            mean_cosine = torch.zeros((), device=DEVICE)

        total = prediction_loss + CYCLE_SSL_LOSS_WEIGHT * cycle_loss

        return (
            total, prediction_loss, cycle_loss,
            mean_cosine, valid_count,
        )

    @torch.no_grad()
    def update_target(self, momentum):
        for online, target in [
            (self.context_backbone, self.target_backbone),
            (self.context_spatial_pool, self.target_spatial_pool),
            (self.context_temporal_pool, self.target_temporal_pool),
        ]:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.mul_(momentum).add_(
                    po.data, alpha=1 - momentum
                )


# ============================================================
# 13. MIL CLASSIFIER
# ============================================================

class LVAreaEncoder(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("phys_mean", mean.clone())
        self.register_buffer("phys_std", std.clone())

        self.curve_encoder = nn.Sequential(
            nn.Linear(AREA_CURVE_POINTS, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 96),
            nn.GELU(),
        )

        self.phys_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128, AREA_EMBED_DIM),
            nn.GELU(),
            nn.LayerNorm(AREA_EMBED_DIM),
        )

    def forward(self, curve, phys):
        curve_feature = self.curve_encoder(curve)
        phys = (phys - self.phys_mean) / (self.phys_std + 1e-6)
        phys_feature = self.phys_encoder(phys)

        return self.fusion(
            torch.cat([curve_feature, phys_feature], dim=-1)
        )


class MILAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.score = nn.Sequential(
            nn.LayerNorm(CYCLE_FUSION_DIM),
            nn.Linear(CYCLE_FUSION_DIM, MIL_ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(MIL_ATTENTION_DIM, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=0)
        return torch.sum(x * w.unsqueeze(-1), dim=0), w


class SICMClassifier(nn.Module):
    def __init__(
        self,
        backbone,
        phys_mean,
        phys_std,
        spatial_pool,
        temporal_pool,
    ):
        super().__init__()

        self.backbone = backbone
        self.spatial_pool = spatial_pool
        self.temporal_pool = temporal_pool
        self.area_encoder = LVAreaEncoder(phys_mean, phys_std)

        self.cycle_fusion = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM + AREA_EMBED_DIM),
            nn.Linear(
                FEATURE_DIM + AREA_EMBED_DIM,
                CYCLE_FUSION_DIM,
            ),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.LayerNorm(CYCLE_FUSION_DIM),
        )

        self.mil_attention = MILAttention()

        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(CYCLE_FUSION_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def backbone_trainable(self):
        return any(p.requires_grad for p in self.backbone.parameters())

    def encode_visual(self, cycles):
        visuals, spatials, temporals = [], [], []

        for start in range(0, len(cycles), CYCLE_FORWARD_CHUNK_SIZE):
            chunk = cycles[start:start + CYCLE_FORWARD_CHUNK_SIZE]

            if self.backbone_trainable():
                tokens = self.backbone(chunk)
            else:
                with torch.no_grad():
                    tokens = self.backbone(chunk)

            temporal_features, spatial_w = self.spatial_pool(
                token_grid(tokens)
            )
            visual, temporal_w = self.temporal_pool(
                temporal_features
            )

            visuals.append(visual)
            spatials.append(spatial_w)
            temporals.append(temporal_w)

        return (
            torch.cat(visuals),
            torch.cat(spatials),
            torch.cat(temporals),
        )

    def forward(self, cycles, curves, physiology):
        visual, spatial_w, temporal_w = self.encode_visual(cycles)

        area_feature = self.area_encoder(curves, physiology)

        fused = self.cycle_fusion(
            torch.cat([visual, area_feature], dim=-1)
        )

        study_feature, cycle_attention = self.mil_attention(fused)
        logit = self.classifier(study_feature).squeeze(-1)

        return {
            "study_logit": logit,
            "cycle_attention": cycle_attention,
            "spatial_attention": spatial_w,
            "temporal_attention": temporal_w,
        }


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for block in model.backbone.blocks[-UNFREEZE_LAST_N_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    for p in model.backbone.norm.parameters():
        p.requires_grad = True


# ============================================================
# 14. METRICS / EVALUATION
# ============================================================

def metrics_from_prob(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / max(tn + fn, 1)),
        "brier": float(brier_score_loss(y, p)),
    }


@torch.no_grad()
def evaluate(model, loader, collect_attention=False):
    model.eval()

    studies = []
    cycles_out = []
    spatial_all = []

    for batch in tqdm(loader, desc="Evaluation", leave=False):
        cycles = batch["cycles"].to(DEVICE)
        curves = batch["area_curves"].to(DEVICE)
        phys = batch["physiology"].to(DEVICE)

        with torch.amp.autocast(
            device_type=DEVICE_TYPE,
            enabled=AMP_ENABLED,
        ):
            out = model(cycles, curves, phys)

        probability = float(
            torch.sigmoid(out["study_logit"].float()).cpu().item()
        )
        label = int(batch["label"].item())

        studies.append(
            {
                "id": batch["id"],
                "label": label,
                "probability": probability,
                "n_cycles": len(batch["cycle_indices"]),
            }
        )

        if collect_attention:
            mil_w = out["cycle_attention"].float().cpu().numpy()
            temp_w = out["temporal_attention"].float().cpu().numpy()
            spatial = out["spatial_attention"].float().cpu().numpy()

            spatial_all.append(spatial)

            for pos, cycle_index in enumerate(batch["cycle_indices"]):
                row = {
                    "id": batch["id"],
                    "label": label,
                    "cycle_index": int(cycle_index),
                    "mil_attention": float(mil_w[pos]),
                }
                for t in range(TEMPORAL_TOKENS):
                    row[f"temporal_attention_{t + 1}"] = float(
                        temp_w[pos, t]
                    )
                cycles_out.append(row)

    study_df = pd.DataFrame(studies)
    result = {
        "study_df": study_df,
        "metrics": metrics_from_prob(
            study_df["label"], study_df["probability"]
        ),
    }

    if collect_attention:
        result["cycle_df"] = pd.DataFrame(cycles_out)
        if spatial_all:
            result["spatial_attention"] = np.concatenate(spatial_all, axis=0)

    return result


def bootstrap_auc_ci(y, p, n_bootstrap=2000):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(SEED)

    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))

    return (
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


# ============================================================
# 15. PATIENT-LEVEL 5-FOLD SPLIT
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

patient_df["fold"] = -1

for fold, (_, val_idx) in enumerate(
    skf.split(np.zeros(len(patient_df)), patient_df["label"]),
    start=1,
):
    patient_df.loc[val_idx, "fold"] = fold

patient_df["fold"] = patient_df["fold"].astype(int)

patient_df.to_csv(
    CV_DIR / "patient_level_5fold_assignment.csv",
    index=False,
)

(
    patient_df.groupby(["fold", "label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
    .to_csv(CV_DIR / "fold_distribution.csv", index=False)
)


# ============================================================
# 16. FOLD-SPECIFIC SSL
# ============================================================

def train_ssl(train_cycle_df, fold_dir):
    dataset = CycleSSLDataset(train_cycle_df)
    loader = DataLoader(
        dataset,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE_TYPE == "cuda",
    )

    base = load_echojepa()
    model = CycleAwareEchoJEPA(base).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=SSL_LR,
        betas=(0.9, 0.95),
        weight_decay=SSL_WEIGHT_DECAY,
    )

    steps_per_epoch = max(
        1,
        math.ceil(len(loader) / SSL_ACCUMULATION_STEPS),
    )
    total_steps = steps_per_epoch * SSL_EPOCHS
    warmup = max(int(total_steps * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup

        progress = (
            step - warmup
        ) / max(total_steps - warmup, 1)

        return 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda
    )
    scaler = torch.amp.GradScaler(
        DEVICE_TYPE, enabled=AMP_ENABLED
    )

    history = []
    optimizer_step = 0

    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()
        model.target_backbone.eval()
        model.target_spatial_pool.eval()
        model.target_temporal_pool.eval()

        total_loss_sum = 0.0
        pred_loss_sum = 0.0
        cycle_loss_sum = 0.0
        cosine_sum = 0.0
        pair_count = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"SSL {epoch}/{SSL_EPOCHS}")
        ):
            context = batch["view_context"].to(DEVICE)
            token_target = batch["view_token_target"].to(DEVICE)
            cycle_target = batch["view_cycle_target"].to(DEVICE)
            has_cross = batch["has_cross_cycle"].to(DEVICE)
            mask = create_mask(context.shape[0])

            with torch.amp.autocast(
                device_type=DEVICE_TYPE,
                enabled=AMP_ENABLED,
            ):
                (
                    total_loss,
                    pred_loss,
                    cycle_loss,
                    mean_cosine,
                    valid_count,
                ) = model(
                    context,
                    token_target,
                    cycle_target,
                    has_cross,
                    mask,
                )

                scaled = total_loss / SSL_ACCUMULATION_STEPS

            scaler.scale(scaled).backward()

            total_loss_sum += float(total_loss.item())
            pred_loss_sum += float(pred_loss.item())
            cycle_loss_sum += float(cycle_loss.item())

            if valid_count > 0:
                cosine_sum += float(mean_cosine.item()) * valid_count
                pair_count += valid_count

            is_last = batch_idx + 1 == len(loader)

            if (
                (batch_idx + 1) % SSL_ACCUMULATION_STEPS == 0
                or is_last
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable, max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                progress = optimizer_step / max(total_steps - 1, 1)
                momentum = EMA_START + (
                    EMA_END - EMA_START
                ) * min(progress, 1.0)

                model.update_target(momentum)
                optimizer_step += 1

        n = max(len(loader), 1)

        history.append(
            {
                "epoch": epoch,
                "total_ssl_loss": total_loss_sum / n,
                "same_cycle_latent_prediction_loss": pred_loss_sum / n,
                "same_id_cross_cycle_consistency_loss": cycle_loss_sum / n,
                "same_id_cross_cycle_cosine":
                    cosine_sum / pair_count if pair_count else np.nan,
                "cross_cycle_pairs": pair_count,
            }
        )

        pd.DataFrame(history).to_csv(
            fold_dir / "ssl_history.csv", index=False
        )

    torch.save(
        {
            "backbone_state_dict":
                model.context_backbone.state_dict(),
            "spatial_pool_state_dict":
                model.context_spatial_pool.state_dict(),
            "temporal_pool_state_dict":
                model.context_temporal_pool.state_dict(),
        },
        fold_dir / "echojepa_cycle_ssl_pretrained.pt",
    )

    backbone = model.context_backbone.cpu()
    spatial_pool = model.context_spatial_pool.cpu()
    temporal_pool = model.context_temporal_pool.cpu()

    model.context_backbone = nn.Identity()
    model.context_spatial_pool = nn.Identity()
    model.context_temporal_pool = nn.Identity()

    del model, base, optimizer, scheduler, scaler, trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return backbone, spatial_pool, temporal_pool


# ============================================================
# 17. TRAIN ONE FOLD
# ============================================================

PHYS_COLUMNS = [
    "fac",
    "systolic_slope",
    "diastolic_slope",
    "cycle_length_proxy",
    "ed_endpoint_mismatch_relative",
]


def stage_a_optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params":
                    list(model.spatial_pool.parameters())
                    + list(model.temporal_pool.parameters()),
                "lr": TEMPORAL_LR,
            },
            {
                "params":
                    list(model.area_encoder.parameters())
                    + list(model.cycle_fusion.parameters())
                    + list(model.mil_attention.parameters())
                    + list(model.classifier.parameters()),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=CLS_WEIGHT_DECAY,
    )


def stage_b_optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.backbone.parameters()
                    if p.requires_grad
                ],
                "lr": BACKBONE_LR,
            },
            {
                "params":
                    list(model.spatial_pool.parameters())
                    + list(model.temporal_pool.parameters()),
                "lr": TEMPORAL_LR,
            },
            {
                "params":
                    list(model.area_encoder.parameters())
                    + list(model.cycle_fusion.parameters())
                    + list(model.mil_attention.parameters())
                    + list(model.classifier.parameters()),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=CLS_WEIGHT_DECAY,
    )


def train_one_fold(fold):
    seed_everything(SEED + fold)

    fold_dir = CV_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    assigned_train_patients = patient_df[
        patient_df["fold"] != fold
    ].copy()
    assigned_val_patients = patient_df[
        patient_df["fold"] == fold
    ].copy()
    train_ids = set(assigned_train_patients["id"].astype(str))
    val_ids = set(assigned_val_patients["id"].astype(str))

    if train_ids & val_ids:
        raise RuntimeError("Patient leakage detected.")

    assigned_train_patients.to_csv(
        fold_dir / "assigned_train_ids.csv", index=False
    )
    assigned_val_patients.to_csv(
        fold_dir / "assigned_validation_ids.csv", index=False
    )

    train_video_df = full_df[full_df["id"].isin(train_ids)].copy()
    val_video_df = full_df[full_df["id"].isin(val_ids)].copy()
    fold_video_df = full_df[full_df["id"].isin(train_ids | val_ids)].copy()

    manifest = preprocess_all(
        fold_video_df,
        normalization_fit_df=train_video_df,
        fold_dir=fold_dir,
    )
    manifest.to_csv(
        fold_dir / "preprocessing" / "multi_cycle_manifest.csv", index=False
    )
    print(f"\nFold {fold} cycle extraction status:")
    print(manifest["status"].value_counts(dropna=False))

    used = manifest[manifest["status"] == "USED"].copy()
    classification_df = used.merge(
        full_df[["id", "label"]],
        on="id",
        how="inner",
    )
    classification_df["label"] = classification_df["label"].astype(int)

    train_cycles = (
        classification_df[
            classification_df["id"].isin(train_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )
    val_cycles = (
        classification_df[
            classification_df["id"].isin(val_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    successful_ids = set(classification_df["id"].astype(str))
    train_patients = assigned_train_patients[
        assigned_train_patients["id"].isin(train_ids & successful_ids)
    ].copy()
    val_patients = assigned_val_patients[
        assigned_val_patients["id"].isin(val_ids & successful_ids)
    ].copy()

    train_failed_ids = train_ids - successful_ids
    val_failed_ids = val_ids - successful_ids

    def write_failed_ids(ids, output_path):
        failure_rows = manifest[
            manifest["id"].astype(str).isin(ids)
            & (manifest["status"] != "USED")
        ][["id", "video_path", "status", "error"]].drop_duplicates("id")
        missing_rows = sorted(ids - set(failure_rows["id"].astype(str)))
        if missing_rows:
            extra = full_df[full_df["id"].astype(str).isin(missing_rows)][
                ["id", "video_path"]
            ].copy()
            extra["status"] = "NO_CYCLES_IN_MANIFEST"
            extra["error"] = "No successful cycle record was written for this ID."
            failure_rows = pd.concat([failure_rows, extra], ignore_index=True)
        failure_rows.to_csv(output_path, index=False)

    write_failed_ids(
        train_failed_ids,
        fold_dir / "training_preprocessing_failures.csv",
    )
    write_failed_ids(
        val_failed_ids,
        fold_dir / "validation_preprocessing_failures.csv",
    )

    if train_cycles.empty or train_patients.empty:
        raise RuntimeError(
            f"Fold {fold} has no successfully preprocessed training patients."
        )
    if val_cycles.empty or val_patients.empty:
        raise RuntimeError(
            f"Fold {fold} has no successfully preprocessed validation patients."
        )
    if train_patients["label"].nunique() < 2:
        raise RuntimeError(
            f"Fold {fold} training patients contain only one class after preprocessing."
        )
    if val_patients["label"].nunique() < 2:
        raise RuntimeError(
            f"Fold {fold} validation patients contain only one class after preprocessing; AUC is undefined."
        )

    train_patients.to_csv(fold_dir / "train_ids.csv", index=False)
    val_patients.to_csv(fold_dir / "validation_ids.csv", index=False)

    phys_np = train_cycles[PHYS_COLUMNS].to_numpy(np.float32)

    phys_mean = torch.tensor(
        phys_np.mean(0), dtype=torch.float32
    )
    phys_std = torch.clamp(
        torch.tensor(phys_np.std(0), dtype=torch.float32),
        min=1e-6,
    )

    pd.DataFrame(
        {
            "feature": PHYS_COLUMNS,
            "training_mean": phys_mean.numpy(),
            "training_std": phys_std.numpy(),
        }
    ).to_csv(
        fold_dir / "physiology_normalization.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("Train candidates / used:", len(assigned_train_patients), "/", len(train_patients))
    print("Validation candidates / used:", len(assigned_val_patients), "/", len(val_patients))
    print("=" * 80)

    backbone, spatial_pool, temporal_pool = train_ssl(
        train_cycles, fold_dir
    )

    model = SICMClassifier(
        backbone,
        phys_mean,
        phys_std,
        spatial_pool,
        temporal_pool,
    ).to(DEVICE)

    train_loader = DataLoader(
        SICMMILStudyDataset(train_cycles, train=True),
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE_TYPE == "cuda",
        collate_fn=mil_collate,
    )

    train_eval_loader = DataLoader(
        SICMMILStudyDataset(train_cycles, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE_TYPE == "cuda",
        collate_fn=mil_collate,
    )

    val_loader = DataLoader(
        SICMMILStudyDataset(val_cycles, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE_TYPE == "cuda",
        collate_fn=mil_collate,
    )

    n_pos = int((train_patients["label"] == 1).sum())
    n_neg = int((train_patients["label"] == 0).sum())

    if USE_CLASS_POS_WEIGHT:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [n_neg / max(n_pos, 1)],
                device=DEVICE,
            )
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    freeze_backbone(model)

    optimizer = stage_a_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(STAGE_A_EPOCHS, 1),
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        DEVICE_TYPE, enabled=AMP_ENABLED
    )

    history = []

    for epoch in range(1, TOTAL_CLS_EPOCHS + 1):
        if epoch == STAGE_A_EPOCHS + 1:
            unfreeze_last_blocks(model)
            optimizer = stage_b_optimizer(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(STAGE_B_EPOCHS, 1),
                eta_min=1e-7,
            )

        stage = "Stage A" if epoch <= STAGE_A_EPOCHS else "Stage B"

        model.train()
        if stage == "Stage A":
            model.backbone.eval()

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        n_studies = 0

        for idx, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Fold {fold} MIL {epoch}/{TOTAL_CLS_EPOCHS} [{stage}]",
            )
        ):
            cycles = batch["cycles"].to(DEVICE)
            curves = batch["area_curves"].to(DEVICE)
            phys = batch["physiology"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE_TYPE,
                enabled=AMP_ENABLED,
            ):
                out = model(cycles, curves, phys)
                loss = criterion(
                    out["study_logit"].view(1),
                    label.view(1),
                )
                scaled = loss / CLS_ACCUMULATION_STEPS

            scaler.scale(scaled).backward()

            running += float(loss.item())
            n_studies += 1

            is_last = idx + 1 == len(train_loader)

            if (
                (idx + 1) % CLS_ACCUMULATION_STEPS == 0
                or is_last
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in model.parameters()
                        if p.requires_grad
                    ],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        train_eval = evaluate(
            model, train_eval_loader, collect_attention=False
        )

        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": running / max(n_studies, 1),
                "train_auc": train_eval["metrics"]["auc"],
                "train_accuracy": train_eval["metrics"]["accuracy"],
            }
        )

        pd.DataFrame(history).to_csv(
            fold_dir / "classification_history.csv",
            index=False,
        )

        torch.save(
            {
                "fold": fold,
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": model.state_dict(),
                "phys_mean": phys_mean,
                "phys_std": phys_std,
            },
            fold_dir / "checkpoint_last.pt",
        )

    torch.save(
        {
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "base_model": "EchoJEPA ViT-L V-JEPA2 vitl-vmix22m-pt220-c55",
            "architecture":
                "EchoJEPA + Spatial Attention + Temporal Attention + "
                "LV-area physiology + multi-cycle MIL",
            "num_frames": NUM_FRAMES,
            "image_size": IMAGE_SIZE,
            "phys_mean": phys_mean,
            "phys_std": phys_std,
        },
        fold_dir / "final_model.pt",
    )

    val_result = evaluate(
        model,
        val_loader,
        collect_attention=SAVE_VALIDATION_ATTENTION,
    )

    val_pred = val_result["study_df"].copy()
    val_pred["prediction"] = (
        val_pred["probability"] >= 0.5
    ).astype(int)
    val_pred["fold"] = fold

    val_pred.to_csv(
        fold_dir / "validation_id_predictions.csv",
        index=False,
    )

    if SAVE_VALIDATION_ATTENTION:
        val_result["cycle_df"].to_csv(
            fold_dir / "validation_cycle_attention.csv",
            index=False,
        )

        if "spatial_attention" in val_result:
            np.save(
                fold_dir / "validation_spatial_attention.npy",
                val_result["spatial_attention"],
            )

    metrics = {
        "fold": fold,
        "training_candidate_patients": len(assigned_train_patients),
        "validation_candidate_patients": len(assigned_val_patients),
        "train_patients": len(train_patients),
        "validation_patients": len(val_patients),
        "training_preprocessing_failures": len(train_failed_ids),
        "validation_preprocessing_failures": len(val_failed_ids),
        **{
            f"val_{k}": v
            for k, v in val_result["metrics"].items()
        },
    }

    pd.DataFrame([metrics]).to_csv(
        fold_dir / "fold_metrics.csv",
        index=False,
    )

    del model, optimizer, scheduler, scaler
    del backbone, spatial_pool, temporal_pool
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_pred, metrics


# ============================================================
# 18. RUN FOLDS
# ============================================================

folds = (
    list(range(1, N_SPLITS + 1))
    if RUN_ONLY_FOLD is None
    else [RUN_ONLY_FOLD]
)

oof_parts = []
metric_rows = []

for fold in folds:
    pred, met = train_one_fold(fold)
    oof_parts.append(pred)
    metric_rows.append(met)

fold_metrics_df = pd.DataFrame(metric_rows)
fold_metrics_df.to_csv(
    CV_DIR / "fold_metrics_summary.csv",
    index=False,
)

validation_failure_parts = []
for fold in folds:
    failure_path = CV_DIR / f"fold_{fold}" / "validation_preprocessing_failures.csv"
    if failure_path.exists():
        failure_df = pd.read_csv(failure_path, dtype={"id": str})
        failure_df["fold"] = fold
        validation_failure_parts.append(failure_df)

if validation_failure_parts:
    pd.concat(validation_failure_parts, ignore_index=True).to_csv(
        CV_DIR / "validation_preprocessing_failures.csv", index=False
    )


# ============================================================
# 19. FINAL OOF
# ============================================================

if RUN_ONLY_FOLD is not None:
    pd.concat(oof_parts, ignore_index=True).to_csv(
        CV_DIR / f"fold_{RUN_ONLY_FOLD}_predictions.csv",
        index=False,
    )
    print("Single-fold run completed. No pooled OOF summary.")

else:
    oof = pd.concat(oof_parts, ignore_index=True)

    if oof["id"].duplicated().any():
        raise RuntimeError("Duplicate IDs found in OOF predictions.")

    expected_oof_count = int(fold_metrics_df["validation_patients"].sum())
    if len(oof) != expected_oof_count:
        raise RuntimeError(
            f"OOF patient count mismatch: {len(oof)} vs {expected_oof_count} successfully preprocessed validation patients"
        )

    assigned_fold_by_id = patient_df.set_index("id")["fold"]
    expected_fold = oof["id"].map(assigned_fold_by_id)
    if expected_fold.isna().any() or not np.array_equal(
        expected_fold.to_numpy(dtype=int), oof["fold"].to_numpy(dtype=int)
    ):
        raise RuntimeError("OOF predictions do not match the patient-level fold assignment.")

    missing_oof_ids = set(patient_df["id"].astype(str)) - set(oof["id"].astype(str))
    if missing_oof_ids:
        missing_path = CV_DIR / "validation_preprocessing_failures.csv"
        if not missing_path.exists():
            raise RuntimeError(
                "Some assigned patients have no OOF prediction and no recorded validation preprocessing failure."
            )
        recorded_failures = set(
            pd.read_csv(missing_path, dtype={"id": str})["id"].astype(str)
        )
        if missing_oof_ids != recorded_failures:
            raise RuntimeError(
                "Patients without OOF predictions do not match the recorded validation preprocessing failures."
            )

    oof = oof.sort_values("id").reset_index(drop=True)
    oof.to_csv(
        CV_DIR / "oof_id_predictions.csv",
        index=False,
    )

    metrics = metrics_from_prob(
        oof["label"],
        oof["probability"],
    )

    ci_low, ci_high = bootstrap_auc_ci(
        oof["label"],
        oof["probability"],
    )

    fold_aucs = fold_metrics_df["val_auc"].to_numpy(float)

    summary = {
        "n_patients": len(oof),
        "oof_auc": metrics["auc"],
        "oof_auc_95ci_low": ci_low,
        "oof_auc_95ci_high": ci_high,
        "oof_accuracy": metrics["accuracy"],
        "oof_f1": metrics["f1"],
        "oof_sensitivity": metrics["sensitivity"],
        "oof_specificity": metrics["specificity"],
        "oof_ppv": metrics["ppv"],
        "oof_npv": metrics["npv"],
        "oof_brier": metrics["brier"],
        "mean_fold_auc": float(fold_aucs.mean()),
        "sd_fold_auc": float(fold_aucs.std(ddof=1)),
    }

    pd.DataFrame([summary]).to_csv(
        CV_DIR / "oof_metrics_summary.csv",
        index=False,
    )

    fpr, tpr, _ = roc_curve(
        oof["label"], oof["probability"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        fpr,
        tpr,
        linewidth=1.5,
        label=(
            f"OOF AUC = {summary['oof_auc']:.3f} "
            f"(95% CI {ci_low:.3f}-{ci_high:.3f})"
        ),
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(
        CV_DIR / "oof_roc.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        CV_DIR / "oof_roc.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n" + "=" * 80)
    print("5-FOLD CV COMPLETE")
    print("=" * 80)
    print(
        f"OOF AUC = {summary['oof_auc']:.4f} "
        f"(95% CI {ci_low:.4f}-{ci_high:.4f})"
    )
    print(
        f"Mean fold AUC ± SD = "
        f"{summary['mean_fold_auc']:.4f} ± "
        f"{summary['sd_fold_auc']:.4f}"
    )

print("\nFold-specific cycle manifests:")
for fold in folds:
    print(CV_DIR / f"fold_{fold}" / "preprocessing" / "multi_cycle_manifest.csv")

print("\nCV output:")
print(CV_DIR)

