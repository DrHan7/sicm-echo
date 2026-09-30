"""
Pure EchoJEPA baseline for SICM classification.

Purpose
-------
This script provides a clean foundation-model baseline for comparison with the
full physiology-informed SICM framework.

Baseline pipeline
-----------------
Raw/cropped echocardiographic NPY
    -> EchoNet-Dynamic LV segmentation for cycle localization only
    -> detect primary ED peaks from the LV area-time curve
    -> retain every adjacent ED-to-ED cycle (QC flags never exclude cycles)
    -> resample each cycle to 16 frames at 224 x 224
    -> grayscale replicated to 3 channels
    -> EchoJEPA ViT-L
    -> global mean pooling over spatiotemporal tokens and all cycles
    -> MLP classifier
    -> patient-level SICM probability

NOT included in this baseline
-----------------------------
- LV-area physiological feature branch
- self-supervised cross-cycle adaptation
- spatial attention
- temporal attention
- physiological descriptors
- learned multiple-instance learning

Cycle handling matches the full-model preprocessing: all adjacent primary ED
peak pairs are retained, including cycles with QC flags. QC is descriptive
only. If fewer than two primary ED peaks are detected, no fallback interval
is created and the study is recorded as an extraction error.

Dataset structure
-----------------
VIDEO_ROOT/
    label.csv
    Center_A/
        ID001/
            ID001.npy
        ID002/
            ID002.npy
    Center_B/
        ID003/
            ID003.npy

The center folder is used only to locate data.
The second-level folder name is the patient ID.
IDs must be unique across centers.
The first column of label.csv is ID and the second column is label.

No internal validation/test split is created.
"""

import gc
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 0. PATHS
# ============================================================

def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return Path(value).expanduser()


VIDEO_ROOT = required_env_path("SICM_VIDEO_ROOT")
LABEL_CSV = required_env_path("SICM_LABEL_CSV")
CHECKPOINT_DIR = Path(os.environ.get("SICM_CHECKPOINT_DIR", "checkpoints")).expanduser()
ECHOJEPA_REPO_DIR = Path(
    os.environ.get("ECHOJEPA_REPO_DIR", str(CHECKPOINT_DIR / "EchoJEPA"))
).expanduser()
ECHOJEPA_CHECKPOINT = Path(
    os.environ.get(
        "ECHOJEPA_CHECKPOINT",
        str(CHECKPOINT_DIR / "vitl-vmix22m-pt220-c55.pt"),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get("SICM_BASELINE_OUTPUT_DIR", "outputs/echojepa_baseline")
).expanduser()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LV_SEGMENTATION_CHECKPOINT = Path(
    os.environ.get(
        "LV_SEGMENTATION_CHECKPOINT",
        str(CHECKPOINT_DIR / "deeplabv3_resnet50_random.pt"),
    )
).expanduser()

CYCLE_ROOT = (
    OUTPUT_DIR
    / "cycles"
)

CYCLE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CYCLE_MANIFEST_PATH = (
    OUTPUT_DIR
    / "baseline_cycle_manifest.csv"
)

CHECKPOINT_LAST = (
    OUTPUT_DIR
    / "checkpoint_last.pt"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "final_model.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "classification_history.csv"
)

TRAIN_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "train_id_predictions.csv"
)


# ============================================================
# 1. RUN SETTINGS
# ============================================================

SEED = 42

QUICK_TEST = False

RESUME_IF_AVAILABLE = True

if QUICK_TEST:
    STAGE_A_EPOCHS = 2
    STAGE_B_EPOCHS = 2
else:
    STAGE_A_EPOCHS = 10
    STAGE_B_EPOCHS = 20

TOTAL_EPOCHS = (
    STAGE_A_EPOCHS
    + STAGE_B_EPOCHS
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DEVICE_TYPE = DEVICE.type

USE_AMP = True

AMP_ENABLED = (
    USE_AMP
    and DEVICE_TYPE == "cuda"
)

NUM_WORKERS = 0

PHYSICAL_BATCH_SIZE = 1

GRAD_ACCUM_STEPS = 4

GRAD_CLIP_NORM = 1.0


# ============================================================
# 2. EchoJEPA SETTINGS
# ============================================================

NUM_FRAMES = 16

IMAGE_SIZE = 224

PATCH_SIZE = 16

TUBELET_SIZE = 2

FEATURE_DIM = 1024

TRANSFORMER_DEPTH = 24

TEMPORAL_TOKENS = (
    NUM_FRAMES
    // TUBELET_SIZE
)

SPATIAL_SIDE = (
    IMAGE_SIZE
    // PATCH_SIZE
)

NUM_TOKENS = (
    TEMPORAL_TOKENS
    * SPATIAL_SIDE
    * SPATIAL_SIDE
)

# EchoNet-Dynamic LV segmentation is used only to locate cardiac cycles.
# These settings mirror the full-model preprocessing.
SEG_SIZE = 112
SEG_BATCH_SIZE = 64
NORMALIZATION_SAMPLE_VIDEOS = 64
NORMALIZATION_FRAMES_PER_VIDEO = 16

SMOOTH_SIGMA = 1.2
MIN_ED_DISTANCE = 6
ED_PROMINENCE_FRACTION = 0.05

# These limits only set manifest QC flags. They never remove cycles.
MIN_CYCLE_FRAMES = 6
MAX_CYCLE_FRAMES = 90
QC_MIN_AREA_EXCURSION_FRACTION = 0.06
QC_MAX_ED_AREA_MISMATCH_FRACTION = 0.60

CYCLE_FILENAME_TEMPLATE = (
    f"cycle_{{cycle_index:02d}}_{NUM_FRAMES}f_{IMAGE_SIZE}px.npy"
)

FORCE_REPROCESS_CYCLES = False

DATA_MEAN = None
DATA_STD = None

ECHOJEPA_MEAN = torch.tensor(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=torch.float32,
)

ECHOJEPA_STD = torch.tensor(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=torch.float32,
)

USE_ACTIVATION_CHECKPOINTING = True


# ============================================================
# 3. TRAINING SETTINGS
# ============================================================

# Stage A: frozen EchoJEPA backbone
STAGE_A_HEAD_LR = 1e-4

# Stage B: conservative partial fine-tuning
STAGE_B_BACKBONE_LR = 2e-6
STAGE_B_HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-3

UNFREEZE_LAST_N_BLOCKS = 2


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

AUG_BRIGHTNESS_PROB = 0.80
AUG_BRIGHTNESS_RANGE = (
    0.85,
    1.15,
)

AUG_CONTRAST_PROB = 0.80
AUG_CONTRAST_RANGE = (
    0.85,
    1.15,
)

AUG_GAMMA_PROB = 0.50
AUG_GAMMA_RANGE = (
    0.80,
    1.25,
)

AUG_TRANSLATE_PROB = 0.50
AUG_MAX_TRANSLATE = 8

# No horizontal flip.
# No temporal reversal.
# No phase rolling.


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(
    SEED
)


# ============================================================
# 6. STARTUP CHECKS
# ============================================================

for path, description in [
    (
        VIDEO_ROOT,
        "training data root",
    ),
    (
        LABEL_CSV,
        "label.csv",
    ),
    (
        ECHOJEPA_REPO_DIR,
        "EchoJEPA repository",
    ),
    (
        ECHOJEPA_CHECKPOINT,
        "EchoJEPA pretrained checkpoint",
    ),
    (
        LV_SEGMENTATION_CHECKPOINT,
        "EchoNet-Dynamic LV segmentation checkpoint",
    ),
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find {description}:\n"
            f"{path}"
        )


if str(
    ECHOJEPA_REPO_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            ECHOJEPA_REPO_DIR
        ),
    )


print(
    "=" * 80
)

print(
    "PURE EchoJEPA BASELINE"
)

print(
    "=" * 80
)

print(
    "Device      :",
    DEVICE,
)

print(
    "Video root  :",
    VIDEO_ROOT,
)

print(
    "Output dir  :",
    OUTPUT_DIR,
)

print(
    "Stage A     :",
    STAGE_A_EPOCHS,
    "epochs",
)

print(
    "Stage B     :",
    STAGE_B_EPOCHS,
    "epochs",
)

print(
    "=" * 80
)


# ============================================================
# 7. LOAD ID/LABEL TABLE AND MATCH EXACT ID-NAMED NPY FILES
#
# Match the full-model development input convention:
#   - label.csv contains exactly two columns: id and label
#   - search VIDEO_ROOT recursively for files named exactly <id>.npy
#   - only IDs listed in label.csv are used
#   - every labeled ID must have exactly one matching NPY file
# ============================================================

label_df = pd.read_csv(
    LABEL_CSV,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

normalized_columns = [
    str(column).strip().lower()
    for column in label_df.columns
]

if (
    len(normalized_columns) != 2
    or normalized_columns.count("id") != 1
    or normalized_columns.count("label") != 1
):
    raise RuntimeError(
        "label.csv must contain exactly two columns named 'id' and 'label'. "
        f"Current columns: {list(label_df.columns)}"
    )

label_df.columns = normalized_columns
label_df = label_df[["id", "label"]].copy()
label_df["id"] = label_df["id"].astype(str).str.strip()
label_df["label"] = pd.to_numeric(
    label_df["label"].astype(str).str.strip(),
    errors="coerce",
)

if label_df["id"].eq("").any():
    raise RuntimeError("label.csv contains an empty ID value.")

conflicts = (
    label_df[label_df["label"].isin([0, 1])]
    .groupby("id")["label"]
    .nunique()
)
conflicts = conflicts[conflicts > 1]

if not conflicts.empty:
    conflict_path = OUTPUT_DIR / "id_label_conflicts.csv"
    conflicts.to_csv(
        conflict_path,
        encoding="utf-8-sig",
    )
    raise RuntimeError(
        f"{len(conflicts)} IDs have conflicting labels. "
        f"Please check: {conflict_path}"
    )

label_df = label_df.drop_duplicates(
    subset=["id"],
    keep="first",
)


def scan_videos(root, label_ids):
    """Find one exact <ID>.npy file for each labeled ID; ignore everything else."""

    expected_ids = {
        str(sample_id).strip()
        for sample_id in label_ids
        if str(sample_id).strip()
    }
    paths_by_id = {
        sample_id: []
        for sample_id in expected_ids
    }
    output_dir = OUTPUT_DIR.resolve()

    for path in root.rglob("*.npy"):
        if path.name.endswith(".tmp.npy"):
            continue

        relative_parts = path.relative_to(root).parts
        if any(part.lower() == "outcome" for part in relative_parts):
            continue

        try:
            path.resolve().relative_to(output_dir)
        except ValueError:
            pass
        else:
            continue

        sample_id = path.stem.strip()
        if sample_id in paths_by_id:
            paths_by_id[sample_id].append(path)

    duplicate_rows = [
        {
            "id": sample_id,
            "video_path": str(path),
        }
        for sample_id, paths in paths_by_id.items()
        if len(paths) > 1
        for path in sorted(paths)
    ]

    if duplicate_rows:
        duplicate_path = OUTPUT_DIR / "duplicate_id_npy_files.csv"
        pd.DataFrame(duplicate_rows).to_csv(
            duplicate_path,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            "More than one exact ID-named NPY file was found for at least "
            "one labeled ID. Keep one <id>.npy per ID. Details: "
            f"{duplicate_path}"
        )

    records = [
        {
            "id": sample_id,
            "video_path": str(paths[0]),
        }
        for sample_id, paths in sorted(paths_by_id.items())
        if len(paths) == 1
    ]
    return pd.DataFrame(records, columns=["id", "video_path"])


video_df = scan_videos(
    VIDEO_ROOT,
    label_df["id"].tolist(),
)

print("\nID-named NPY files matched:", len(video_df))

full_df = label_df.merge(
    video_df,
    on="id",
    how="left",
    indicator=True,
)

matched_mask = full_df["_merge"] == "both"
n_matched = int(matched_mask.sum())
n_missing = int((~matched_mask).sum())

print("IDs with matching <id>.npy :", n_matched)
print("IDs without matching file  :", n_missing)

if n_missing > 0:
    missing_path = OUTPUT_DIR / "ids_without_matching_npy.csv"
    full_df.loc[
        ~matched_mask,
        ["id", "label"],
    ].to_csv(
        missing_path,
        index=False,
        encoding="utf-8-sig",
    )
    raise RuntimeError(
        "Training stopped because each labeled ID must have one matching "
        f"<id>.npy file. Missing IDs are listed in: {missing_path}"
    )

train_df = (
    full_df.loc[
        matched_mask & full_df["label"].isin([0, 1]),
        ["id", "video_path", "label"],
    ]
    .copy()
)

train_df["label"] = train_df["label"].astype(int)

if train_df.empty:
    raise RuntimeError("No labeled videos were matched.")

if train_df["label"].nunique() != 2:
    raise RuntimeError("Both label=0 and label=1 are required.")

train_df.to_csv(
    OUTPUT_DIR / "training_ids.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nTraining IDs:", len(train_df))
print("\nLabel distribution:")
print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. RAW NPY LOADING
# ============================================================

def load_video_tchw(
    npy_path,
):

    array = np.asarray(
        np.load(
            npy_path,
            mmap_mode="r",
        )
    )

    if array.ndim == 3:

        # T,H,W
        x = (
            torch.from_numpy(
                array.copy()
            )
            .float()
            .unsqueeze(
                1
            )
        )

    elif (
        array.ndim == 4
        and array.shape[
            -1
        ] in (
            1,
            3,
        )
    ):

        # T,H,W,C
        x = (
            torch.from_numpy(
                array.copy()
            )
            .float()
            .permute(
                0,
                3,
                1,
                2,
            )
        )

    elif (
        array.ndim == 4
        and array.shape[
            1
        ] in (
            1,
            3,
        )
    ):

        # T,C,H,W
        x = (
            torch.from_numpy(
                array.copy()
            )
            .float()
        )

    else:

        raise ValueError(
            f"Unsupported video shape {array.shape}: {npy_path}"
        )

    return x


# ============================================================
# 10. FULL-MODEL CARDIAC-CYCLE PREPROCESSING
# ============================================================

def to_three_channel_255(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"Channel count must be 1 or 3; got {x.shape[1]}")

    vmin = float(x.min())
    vmax = float(x.max())
    if vmin >= 0 and vmax <= 1.5:
        x = x * 255.0
    elif vmin < 0 or vmax > 255.0:
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin) * 255.0
        else:
            x = torch.zeros_like(x)
    return x


def to_grayscale_255(x):
    if x.shape[1] == 3:
        x = (
            0.2989 * x[:, 0:1]
            + 0.5870 * x[:, 1:2]
            + 0.1140 * x[:, 2:3]
        )
    elif x.shape[1] != 1:
        raise ValueError(f"Channel count must be 1 or 3; got {x.shape[1]}")

    vmin = float(x.min())
    vmax = float(x.max())
    if vmin >= 0 and vmax <= 1.5:
        x = x * 255.0
    elif vmin < 0 or vmax > 255.0:
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin) * 255.0
        else:
            x = torch.zeros_like(x)
    return x.clamp(0, 255)


def estimate_mean_std(dataframe, max_videos, frames_per_video):
    if len(dataframe) == 0:
        raise RuntimeError("No videos available to estimate normalization.")

    selected = dataframe
    if len(dataframe) > max_videos:
        selected = dataframe.sample(n=max_videos, random_state=SEED)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Segmentation normalization"):
        x = load_video_tchw(row["video_path"])
        if x.shape[0] > frames_per_video:
            indices = np.linspace(0, x.shape[0] - 1, frames_per_video).round().astype(int)
            x = x[indices]

        x = to_three_channel_255(x)
        x = F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
        flat = x.permute(1, 0, 2, 3).reshape(3, -1).double()
        channel_sum += flat.sum(dim=1)
        channel_sq_sum += (flat * flat).sum(dim=1)
        total_pixels += flat.shape[1]

    if total_pixels == 0:
        raise RuntimeError("Normalization accumulated zero pixels.")

    mean = channel_sum / total_pixels
    variance = (channel_sq_sum / total_pixels - mean * mean).clamp_min(1e-8)
    std = torch.sqrt(variance)
    mean = mean.float()
    std = std.float()

    pd.DataFrame(
        {"channel": ["R", "G", "B"], "mean": mean.numpy(), "std": std.numpy()}
    ).to_csv(OUTPUT_DIR / "data_normalization.csv", index=False, encoding="utf-8-sig")
    return mean, std


def build_lv_segmenter(weight_path):
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
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    result = model.load_state_dict(cleaned_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "LV segmentation checkpoint does not match DeepLabV3-ResNet50: "
            f"missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}"
        )

    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def get_lv_area_curve(npy_path, segmenter):
    original = load_video_tchw(npy_path)
    x = to_three_channel_255(original.clone())
    x = F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
    x = (x - DATA_MEAN.view(1, 3, 1, 1)) / DATA_STD.view(1, 3, 1, 1)

    areas = []
    for start in range(0, x.shape[0], SEG_BATCH_SIZE):
        batch = x[start:start + SEG_BATCH_SIZE].to(DEVICE, non_blocking=True)
        logits = segmenter(batch)["out"][:, 0]
        areas.append((logits > 0).sum(dim=(1, 2)).cpu().numpy().astype(np.float32))
    return original, np.concatenate(areas)


def build_cycle_candidate(smooth, area_range, ed1, ed2):
    ed1, ed2 = int(ed1), int(ed2)
    period = ed2 - ed1
    if period < 1:
        return None

    es = ed1 + int(np.argmin(smooth[ed1:ed2 + 1]))
    mean_ed_area = (smooth[ed1] + smooth[ed2]) / 2.0
    es_area = float(smooth[es])
    excursion_fraction = float((mean_ed_area - es_area) / area_range)
    ed_area_mismatch_fraction = float(abs(smooth[ed1] - smooth[ed2]) / area_range)
    return {
        "method": "primary_peaks",
        "ed1": ed1,
        "es": es,
        "ed2": ed2,
        "cycle_frames": int(period),
        "excursion_fraction": excursion_fraction,
        "ed_area_mismatch_fraction": ed_area_mismatch_fraction,
        "quality_score": float(excursion_fraction - 0.25 * ed_area_mismatch_fraction),
        "qc_excursion_low": excursion_fraction < QC_MIN_AREA_EXCURSION_FRACTION,
        "qc_ed_mismatch_high": ed_area_mismatch_fraction > QC_MAX_ED_AREA_MISMATCH_FRACTION,
        "qc_cycle_length_outside_preferred": period < MIN_CYCLE_FRAMES or period > MAX_CYCLE_FRAMES,
    }


def detect_all_cycles(raw_area):
    """Keep every adjacent primary ED-peak pair; never create fallback cycles."""
    raw_area = np.asarray(raw_area, dtype=np.float32)
    if len(raw_area) < 2:
        raise RuntimeError("Video has fewer than 2 frames; no temporal interval can be formed.")

    smooth = gaussian_filter1d(raw_area, sigma=SMOOTH_SIGMA, mode="nearest")
    q05, q95 = np.quantile(smooth, [0.05, 0.95])
    area_range = max(float(q95 - q05), 1.0)
    ed_candidates, _ = find_peaks(
        smooth,
        distance=MIN_ED_DISTANCE,
        prominence=ED_PROMINENCE_FRACTION * area_range,
    )
    if len(ed_candidates) < 2:
        raise RuntimeError(
            "Fewer than two ED peaks were detected with the primary settings; "
            "cardiac-cycle detection failed."
        )

    cycles = []
    for index in range(len(ed_candidates) - 1):
        candidate = build_cycle_candidate(
            smooth, area_range, ed_candidates[index], ed_candidates[index + 1]
        )
        if candidate is not None:
            cycles.append(candidate)
    if not cycles:
        raise RuntimeError("No valid adjacent ED-to-ED cycles were found.")

    return {"cycles": cycles, "smooth_area": smooth, "ed_candidates": ed_candidates}


def resample_cycle(original_video, ed1, ed2):
    cycle = original_video[int(ed1):int(ed2) + 1]
    if cycle.shape[0] < 2:
        raise RuntimeError("Detected cycle contains fewer than 2 frames.")

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


CYCLE_MANIFEST_COLUMNS = [
    "id", "video_path", "status", "cycle_index", "n_cycles_in_study", "method",
    "original_frames", "ed1", "es", "ed2", "cycle_frames", "excursion_fraction",
    "ed_area_mismatch_fraction", "quality_score", "qc_excursion_low",
    "qc_ed_mismatch_high", "qc_cycle_length_outside_preferred", "cycle_path", "error",
]


def process_all_cycles(dataframe, segmenter):
    previous_groups = {}
    if CYCLE_MANIFEST_PATH.exists() and not FORCE_REPROCESS_CYCLES:
        existing = pd.read_csv(CYCLE_MANIFEST_PATH, dtype={"id": str})
        previous_groups = {
            str(key): group.copy()
            for key, group in existing.groupby("id", sort=False)
        }

    records = []
    for sample_counter, (_, row) in enumerate(
        tqdm(dataframe.iterrows(), total=len(dataframe), desc="Cardiac cycles")
    ):
        sample_id = str(row["id"])
        video_path = row["video_path"]
        old_group = previous_groups.get(sample_id)
        if old_group is not None and not FORCE_REPROCESS_CYCLES:
            used = old_group[old_group["status"] == "USED"]
            cache_valid = len(used) > 0 and used["method"].astype(str).eq("primary_peaks").all()
            if cache_valid:
                cache_valid = all(Path(path).exists() for path in used["cycle_path"].astype(str))
            if cache_valid and old_group["video_path"].astype(str).eq(str(video_path)).all():
                records.extend(
                    {column: old_row.get(column, np.nan) for column in CYCLE_MANIFEST_COLUMNS}
                    for _, old_row in old_group.iterrows()
                )
                continue

        study_root = CYCLE_ROOT / sample_id
        study_root.mkdir(parents=True, exist_ok=True)
        try:
            original_video, raw_area = get_lv_area_curve(video_path, segmenter)
            detection = detect_all_cycles(raw_area)
            cycles = detection["cycles"]
            for cycle_index, cycle in enumerate(cycles, start=1):
                cycle_path = study_root / CYCLE_FILENAME_TEMPLATE.format(cycle_index=cycle_index)
                np.save(cycle_path, resample_cycle(original_video, cycle["ed1"], cycle["ed2"]))
                records.append({
                    "id": sample_id,
                    "video_path": str(video_path),
                    "status": "USED",
                    "cycle_index": cycle_index,
                    "n_cycles_in_study": len(cycles),
                    "method": cycle["method"],
                    "original_frames": int(original_video.shape[0]),
                    "ed1": cycle["ed1"],
                    "es": cycle["es"],
                    "ed2": cycle["ed2"],
                    "cycle_frames": cycle["cycle_frames"],
                    "excursion_fraction": cycle["excursion_fraction"],
                    "ed_area_mismatch_fraction": cycle["ed_area_mismatch_fraction"],
                    "quality_score": cycle["quality_score"],
                    "qc_excursion_low": cycle["qc_excursion_low"],
                    "qc_ed_mismatch_high": cycle["qc_ed_mismatch_high"],
                    "qc_cycle_length_outside_preferred": cycle["qc_cycle_length_outside_preferred"],
                    "cycle_path": str(cycle_path),
                    "error": "",
                })
        except Exception as exc:
            records.append({
                "id": sample_id,
                "video_path": str(video_path),
                "status": "ERROR",
                "cycle_index": -1,
                "method": "",
                "cycle_path": "",
                "error": repr(exc),
            })

        if (sample_counter + 1) % 50 == 0:
            pd.DataFrame(records, columns=CYCLE_MANIFEST_COLUMNS).to_csv(
                CYCLE_MANIFEST_PATH, index=False, encoding="utf-8-sig"
            )

    manifest = pd.DataFrame(records, columns=CYCLE_MANIFEST_COLUMNS)
    manifest.to_csv(CYCLE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


DATA_MEAN, DATA_STD = estimate_mean_std(
    train_df,
    max_videos=NORMALIZATION_SAMPLE_VIDEOS,
    frames_per_video=NORMALIZATION_FRAMES_PER_VIDEO,
)
lv_model = build_lv_segmenter(LV_SEGMENTATION_CHECKPOINT)
cycle_manifest = process_all_cycles(train_df, lv_model)
used_cycle_manifest = cycle_manifest[cycle_manifest["status"] == "USED"].copy()
if used_cycle_manifest.empty:
    raise RuntimeError("No cardiac-cycle instances were generated successfully.")

cycle_paths_by_id = {
    str(study_id): group.sort_values("cycle_index")["cycle_path"].astype(str).tolist()
    for study_id, group in used_cycle_manifest.groupby("id", sort=False)
}
train_df = train_df[train_df["id"].astype(str).isin(cycle_paths_by_id)].copy()
train_df = train_df.reset_index(drop=True)
print(
    f"Retained {len(used_cycle_manifest)} adjacent ED-to-ED cycles from "
    f"{len(cycle_paths_by_id)} studies; QC flags were not used for exclusion."
)
del lv_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 11. TEMPORALLY CONSISTENT AUGMENTATION
# ============================================================

def random_translate(
    video,
    max_translate,
):

    if max_translate <= 0:
        return video

    shift_x = random.randint(
        -max_translate,
        max_translate,
    )

    shift_y = random.randint(
        -max_translate,
        max_translate,
    )

    if (
        shift_x == 0
        and shift_y == 0
    ):

        return video

    translated = torch.roll(
        video,
        shifts=(
            shift_y,
            shift_x,
        ),
        dims=(
            2,
            3,
        ),
    )

    # Remove wrapped pixels.
    if shift_y > 0:
        translated[
            :,
            :,
            :shift_y,
            :
        ] = 0

    elif shift_y < 0:
        translated[
            :,
            :,
            shift_y:,
            :
        ] = 0

    if shift_x > 0:
        translated[
            :,
            :,
            :,
            :shift_x
        ] = 0

    elif shift_x < 0:
        translated[
            :,
            :,
            :,
            shift_x:
        ] = 0

    return translated


def augment_video(
    video,
):

    if (
        random.random()
        < AUG_TRANSLATE_PROB
    ):

        video = random_translate(
            video,
            AUG_MAX_TRANSLATE,
        )

    if (
        random.random()
        < AUG_BRIGHTNESS_PROB
    ):

        factor = random.uniform(
            *AUG_BRIGHTNESS_RANGE
        )

        video = (
            video
            * factor
        ).clamp(
            0.0,
            1.0,
        )

    if (
        random.random()
        < AUG_CONTRAST_PROB
    ):

        factor = random.uniform(
            *AUG_CONTRAST_RANGE
        )

        mean_value = video.mean()

        video = (
            (
                video
                - mean_value
            )
            * factor
            + mean_value
        ).clamp(
            0.0,
            1.0,
        )

    if (
        random.random()
        < AUG_GAMMA_PROB
    ):

        gamma = random.uniform(
            *AUG_GAMMA_RANGE
        )

        video = (
            video
            .clamp(
                1e-6,
                1.0,
            )
            .pow(
                gamma
            )
        )

    return video


# ============================================================
# 12. EchoJEPA INPUT PREPARATION
# ============================================================

def load_cached_cycle(cycle_path):
    array = np.load(cycle_path)
    if array.ndim == 3:
        video = torch.from_numpy(array.copy()).float().unsqueeze(1).repeat(1, 3, 1, 1)
    elif array.ndim == 4 and array.shape[-1] == 3:
        video = torch.from_numpy(array.copy()).float().permute(0, 3, 1, 2)
    elif array.ndim == 4 and array.shape[1] == 3:
        video = torch.from_numpy(array.copy()).float()
    else:
        raise ValueError(f"Unsupported cached cycle shape: {array.shape}")

    if array.dtype == np.uint8 or float(video.max()) > 1.5:
        video = video / 255.0
    return video.clamp(0.0, 1.0)

def prepare_echojepa_input(
    cycle_path,
    train,
):

    # Cached cycles are already cropped ED1..ED2 and resampled to the
    # full-model 16 x 224 x 224 geometry.
    video = load_cached_cycle(cycle_path)

    if train:

        video = augment_video(
            video
        )

    mean = ECHOJEPA_MEAN.view(
        1,
        3,
        1,
        1,
    )

    std = ECHOJEPA_STD.view(
        1,
        3,
        1,
        1,
    )

    video = (
        video
        - mean
    ) / std

    # T,C,H,W -> C,T,H,W
    return video.permute(
        1,
        0,
        2,
        3,
    ).contiguous()


# ============================================================
# 13. DATASET
# ============================================================

class EchoJEPABaselineDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        train,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.train = train

    def __len__(
        self
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[
            idx
        ]

        study_id = str(row["id"])
        cycle_paths = cycle_paths_by_id[study_id]
        video = torch.stack(
            [
                prepare_echojepa_input(path, train=self.train)
                for path in cycle_paths
            ],
            dim=0,
        )

        return {
            "video":
                video,

            "label":
                torch.tensor(
                    float(
                        row[
                            "label"
                        ]
                    ),
                    dtype=torch.float32,
                ),

            "id":
                str(
                    row[
                        "id"
                    ]
                ),
        }


def collate_fn(
    batch,
):

    # Physical batch size is 1.
    return batch[
        0
    ]


train_dataset = EchoJEPABaselineDataset(
    train_df,
    train=True,
)

eval_dataset = EchoJEPABaselineDataset(
    train_df,
    train=False,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=PHYSICAL_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE_TYPE == "cuda"
    ),
    collate_fn=collate_fn,
)


eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE_TYPE == "cuda"
    ),
    collate_fn=collate_fn,
)


# ============================================================
# 14. EchoJEPA CHECKPOINT HELPERS
# ============================================================

def _choose_encoder_state_dict(
    checkpoint,
):

    if not isinstance(
        checkpoint,
        dict,
    ):

        return (
            checkpoint,
            "root",
        )

    preferred_keys = [
        "target_encoder",
        "ema_encoder",
        "encoder",
        "backbone",
        "model",
        "state_dict",
    ]

    for key in preferred_keys:

        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):

            return (
                value,
                key,
            )

    if all(
        isinstance(
            key,
            str,
        )
        for key in checkpoint.keys()
    ):

        return (
            checkpoint,
            "root",
        )

    raise RuntimeError(
        "Could not identify encoder state_dict in checkpoint."
    )


def _clean_echojepa_key(
    key,
):

    prefixes = [
        "module.",
        "backbone.",
        "encoder.",
        "target_encoder.",
        "ema_encoder.",
    ]

    changed = True

    while changed:

        changed = False

        for prefix in prefixes:

            if key.startswith(
                prefix
            ):

                key = key[
                    len(
                        prefix
                    ):
                ]

                changed = True

    return key


# ============================================================
# 15. LOAD OFFICIAL EchoJEPA ViT-L
# ============================================================

def load_echojepa_vitl(
    checkpoint_path,
):

    try:

        from src.models import (
            vision_transformer
            as vit_encoder
        )

    except Exception as exc:

        raise ImportError(
            "Could not import EchoJEPA repository.\n"
            f"Repo: {ECHOJEPA_REPO_DIR}\n"
            f"Error: {repr(exc)}"
        )

    encoder = vit_encoder.vit_large(
        patch_size=PATCH_SIZE,
        img_size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        use_activation_checkpointing=(
            USE_ACTIVATION_CHECKPOINTING
        ),
        handle_nonsquare_inputs=True,
    )

    if encoder.embed_dim != FEATURE_DIM:

        raise RuntimeError(
            f"Unexpected embed_dim: {encoder.embed_dim}"
        )

    if len(
        encoder.blocks
    ) != TRANSFORMER_DEPTH:

        raise RuntimeError(
            f"Unexpected depth: {len(encoder.blocks)}"
        )

    if encoder.num_patches != NUM_TOKENS:

        raise RuntimeError(
            f"Unexpected token count: {encoder.num_patches}"
        )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    raw_state, selected_key = (
        _choose_encoder_state_dict(
            checkpoint
        )
    )

    cleaned_state = {
        _clean_echojepa_key(
            key
        ):
        value

        for key, value
        in raw_state.items()

        if torch.is_tensor(
            value
        )
    }

    model_state = encoder.state_dict()

    matched_state = {}

    matched_numel = 0

    total_numel = sum(
        tensor.numel()
        for tensor
        in model_state.values()
    )

    for key, value in cleaned_state.items():

        if key not in model_state:
            continue

        if model_state[
            key
        ].shape != value.shape:
            continue

        matched_state[
            key
        ] = value

        matched_numel += value.numel()

    match_ratio = (
        matched_numel
        / max(
            total_numel,
            1,
        )
    )

    print(
        "\nEchoJEPA checkpoint key:",
        selected_key
    )

    print(
        "Matched parameter ratio:",
        f"{100 * match_ratio:.2f}%"
    )

    if match_ratio < 0.95:

        raise RuntimeError(
            "Less than 95% of EchoJEPA parameters matched."
        )

    encoder.load_state_dict(
        matched_state,
        strict=False,
    )

    print(
        "EchoJEPA ViT-L loaded successfully."
    )

    return encoder


# ============================================================
# 16. PURE EchoJEPA BASELINE CLASSIFIER
# ============================================================

class EchoJEPABaselineClassifier(
    nn.Module
):

    def __init__(
        self,
        backbone,
    ):

        super().__init__()

        self.backbone = backbone

        # No spatial attention.
        # No temporal attention.
        # No physiology branch.
        # No learned MIL; all retained cycles contribute equally by mean pooling.
        self.classifier = nn.Sequential(
            nn.LayerNorm(
                FEATURE_DIM
            ),
            nn.Dropout(
                0.20
            ),
            nn.Linear(
                FEATURE_DIM,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        video,
    ):

        # video: N_cycles,C,T,H,W (or C,T,H,W for one cycle)
        if video.ndim == 4:

            video = video.unsqueeze(
                0
            )

        if video.ndim != 5:
            raise RuntimeError(
                f"Expected cycles with shape (N,C,T,H,W); got {video.shape}"
            )

        # Encode one cycle at a time to limit peak memory. Every cycle is
        # retained, and equal-sized token grids make this equivalent to global
        # mean pooling over all tokens from every cycle in the study.
        cycle_features = []
        for cycle in video:
            tokens = self.backbone(cycle.unsqueeze(0))
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
            if tokens.ndim != 3:
                raise RuntimeError(
                    f"Unexpected EchoJEPA output shape: {tokens.shape}"
                )
            cycle_features.append(tokens.mean(dim=1).squeeze(0))

        feature = torch.stack(cycle_features, dim=0).mean(dim=0, keepdim=True)

        logit = (
            self.classifier(
                feature
            )
            .squeeze(
                -1
            )
        )

        return logit


backbone = load_echojepa_vitl(
    ECHOJEPA_CHECKPOINT
)

model = EchoJEPABaselineClassifier(
    backbone
).to(
    DEVICE
)


# ============================================================
# 17. FREEZE / UNFREEZE
# ============================================================

def freeze_backbone(
    model,
):

    for parameter in model.backbone.parameters():

        parameter.requires_grad = False

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True


def unfreeze_last_blocks(
    model,
    n_blocks,
):

    for parameter in model.backbone.parameters():

        parameter.requires_grad = False

    blocks = model.backbone.blocks

    for block in blocks[
        -n_blocks:
    ]:

        for parameter in block.parameters():

            parameter.requires_grad = True

    if hasattr(
        model.backbone,
        "norm",
    ):

        for parameter in model.backbone.norm.parameters():

            parameter.requires_grad = True

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True


# ============================================================
# 18. CLASS WEIGHT
# ============================================================

n_negative = int(
    (
        train_df[
            "label"
        ]
        == 0
    ).sum()
)

n_positive = int(
    (
        train_df[
            "label"
        ]
        == 1
    ).sum()
)

POS_WEIGHT = float(
    n_negative
    / max(
        n_positive,
        1,
    )
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        POS_WEIGHT,
        dtype=torch.float32,
        device=DEVICE,
    )
)

print(
    "\nPositive class weight:",
    POS_WEIGHT,
)


# ============================================================
# 19. OPTIMIZERS
# ============================================================

def make_stage_a_optimizer(
    model,
):

    freeze_backbone(
        model
    )

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=STAGE_A_HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=max(
                STAGE_A_EPOCHS,
                1,
            ),
            eta_min=1e-7,
        )
    )

    return (
        optimizer,
        scheduler,
    )


def make_stage_b_optimizer(
    model,
):

    unfreeze_last_blocks(
        model,
        UNFREEZE_LAST_N_BLOCKS,
    )

    backbone_params = [
        parameter

        for parameter
        in model.backbone.parameters()

        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_params,

                "lr":
                    STAGE_B_BACKBONE_LR,
            },
            {
                "params":
                    model.classifier.parameters(),

                "lr":
                    STAGE_B_HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=max(
                STAGE_B_EPOCHS,
                1,
            ),
            eta_min=1e-7,
        )
    )

    return (
        optimizer,
        scheduler,
    )


# ============================================================
# 20. METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= 0.5
    ).astype(
        np.int64
    )

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    sensitivity = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
        ],
    ).ravel()

    specificity = (
        tn
        / max(
            tn + fp,
            1,
        )
    )

    return {
        "auc":
            float(
                auc
            ),

        "accuracy":
            float(
                accuracy
            ),

        "f1":
            float(
                f1
            ),

        "sensitivity":
            float(
                sensitivity
            ),

        "specificity":
            float(
                specificity
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }


# ============================================================
# 21. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
):

    model.train()

    # Keep frozen backbone deterministic.
    if not any(
        parameter.requires_grad
        for parameter
        in model.backbone.parameters()
    ):

        model.backbone.eval()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0

    n_samples = 0

    progress = tqdm(
        loader,
        desc="Training",
    )

    for step, batch in enumerate(
        progress,
        start=1,
    ):

        video = batch[
            "video"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        label = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=DEVICE_TYPE,
            enabled=AMP_ENABLED,
        ):

            logit = model(
                video
            )

            loss = criterion(
                logit.view(
                    -1
                ),
                label.view(
                    -1
                ),
            )

            scaled_loss = (
                loss
                / GRAD_ACCUM_STEPS
            )

        scaler.scale(
            scaled_loss
        ).backward()

        should_step = (
            step
            % GRAD_ACCUM_STEPS
            == 0
            or step
            == len(
                loader
            )
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter

                    for parameter
                    in model.parameters()

                    if parameter.requires_grad
                ],
                GRAD_CLIP_NORM,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += float(
            loss.item()
        )

        n_samples += 1

        progress.set_postfix(
            {
                "loss":
                    f"{running_loss / n_samples:.4f}",
            }
        )

    return (
        running_loss
        / max(
            n_samples,
            1,
        )
    )


# ============================================================
# 22. EVALUATE TRAINING COHORT
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()

    records = []

    for batch in tqdm(
        loader,
        desc="Train-set inference",
    ):

        video = batch[
            "video"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        label = int(
            batch[
                "label"
            ].item()
        )

        with torch.amp.autocast(
            device_type=DEVICE_TYPE,
            enabled=AMP_ENABLED,
        ):

            logit = model(
                video
            )

        probability = float(
            torch.sigmoid(
                logit.float()
            )
            .view(
                -1
            )[
                0
            ]
            .cpu()
            .item()
        )

        records.append(
            {
                "id":
                    batch[
                        "id"
                    ],

                "label":
                    label,

                "probability":
                    probability,

                "prediction":
                    int(
                        probability
                        >= 0.5
                    ),
            }
        )

    pred_df = pd.DataFrame(
        records
    )

    metrics = calculate_metrics(
        pred_df[
            "label"
        ].values,
        pred_df[
            "probability"
        ].values,
    )

    return (
        metrics,
        pred_df,
    )


# ============================================================
# 23. AMP SCALER
# ============================================================

scaler = torch.amp.GradScaler(
    DEVICE_TYPE,
    enabled=AMP_ENABLED,
)


# ============================================================
# 24. RESUME OR FRESH START
# ============================================================

history = []

start_epoch = 1

optimizer = None

scheduler = None


if (
    RESUME_IF_AVAILABLE
    and CHECKPOINT_LAST.exists()
):

    print(
        "\nResume checkpoint found:"
    )

    print(
        CHECKPOINT_LAST
    )

    checkpoint = torch.load(
        CHECKPOINT_LAST,
        map_location="cpu",
        weights_only=False,
    )

    last_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    if last_epoch < STAGE_A_EPOCHS:

        optimizer, scheduler = (
            make_stage_a_optimizer(
                model
            )
        )

    else:

        optimizer, scheduler = (
            make_stage_b_optimizer(
                model
            )
        )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    if (
        "scaler_state_dict"
        in checkpoint
        and checkpoint[
            "scaler_state_dict"
        ] is not None
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    # Move optimizer states to GPU.
    for state in optimizer.state.values():

        for key, value in list(
            state.items()
        ):

            if torch.is_tensor(
                value
            ):

                state[
                    key
                ] = value.to(
                    DEVICE
                )

    start_epoch = (
        last_epoch
        + 1
    )

    if HISTORY_PATH.exists():

        history_df = pd.read_csv(
            HISTORY_PATH
        )

        history_df = history_df[
            history_df[
                "epoch"
            ]
            <= last_epoch
        ]

        history = (
            history_df
            .to_dict(
                orient="records"
            )
        )

    print(
        "Last completed epoch:",
        last_epoch
    )

    print(
        "Resume from epoch:",
        start_epoch
    )

    del checkpoint

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


else:

    optimizer, scheduler = (
        make_stage_a_optimizer(
            model
        )
    )


# ============================================================
# 25. TRAINING LOOP
# ============================================================

for epoch in range(
    start_epoch,
    TOTAL_EPOCHS + 1,
):

    # --------------------------------------------------------
    # Switch to Stage B exactly once.
    # --------------------------------------------------------

    if epoch == (
        STAGE_A_EPOCHS
        + 1
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "STAGE B: UNFREEZE LAST "
            f"{UNFREEZE_LAST_N_BLOCKS} EchoJEPA BLOCKS"
        )

        print(
            "=" * 80
        )

        optimizer, scheduler = (
            make_stage_b_optimizer(
                model
            )
        )

        # A new optimizer starts a new Stage-B schedule.
        scaler = torch.amp.GradScaler(
            DEVICE_TYPE,
            enabled=AMP_ENABLED,
        )


    stage_name = (
        "Stage A - frozen EchoJEPA"
        if epoch
        <= STAGE_A_EPOCHS
        else
        "Stage B - partial EchoJEPA fine-tuning"
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"EPOCH {epoch}/{TOTAL_EPOCHS}"
    )

    print(
        stage_name
    )

    print(
        "=" * 80
    )


    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
    )


    metrics, pred_df = evaluate(
        model,
        eval_loader,
    )


    scheduler.step()


    row = {
        "epoch":
            epoch,

        "stage":
            stage_name,

        "train_loss":
            train_loss,

        "train_auc":
            metrics[
                "auc"
            ],

        "accuracy":
            metrics[
                "accuracy"
            ],

        "f1":
            metrics[
                "f1"
            ],

        "sensitivity":
            metrics[
                "sensitivity"
            ],

        "specificity":
            metrics[
                "specificity"
            ],

        "lr_group_0":
            optimizer.param_groups[
                0
            ][
                "lr"
            ],

        "lr_group_1":
            (
                optimizer.param_groups[
                    1
                ][
                    "lr"
                ]
                if len(
                    optimizer.param_groups
                )
                > 1
                else np.nan
            ),
    }


    history.append(
        row
    )


    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )


    # Save the newest deterministic predictions.
    pred_df.to_csv(
        TRAIN_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )


    torch.save(
        {
            "epoch":
                epoch,

            "stage":
                stage_name,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "train_loss":
                train_loss,

            "train_auc":
                metrics[
                    "auc"
                ],

            "architecture":
                "EchoJEPA ViT-L + equal mean pooling across all cycle tokens + MLP classifier",

            "baseline_definition":
                "all adjacent primary ED-to-ED cycles; QC flags do not exclude cycles; no physiology feature branch or learned MIL",
        },
        CHECKPOINT_LAST,
    )


    print(
        f"Loss={train_loss:.4f} | "
        f"AUC={metrics['auc']:.4f} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Sens={metrics['sensitivity']:.4f} | "
        f"Spec={metrics['specificity']:.4f}"
    )


# ============================================================
# 26. FINAL EVALUATION
# ============================================================

final_metrics, final_predictions = evaluate(
    model,
    eval_loader,
)


final_predictions.to_csv(
    TRAIN_PREDICTIONS_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 27. SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "architecture":
            "EchoJEPA ViT-L + equal mean pooling across all cycle tokens + MLP classifier",

        "base_model":
            "vitl-vmix22m-pt220-c55",

        "num_frames":
            NUM_FRAMES,

        "image_size":
            IMAGE_SIZE,

        "sampling":
            "all adjacent primary ED-to-ED cycles; each resampled to 16 frames",

        "pooling":
            "global mean pooling across all EchoJEPA spatiotemporal tokens from all cycles",

        "threshold":
            0.5,

        "train_metrics":
            final_metrics,
    },
    FINAL_MODEL_PATH,
)


# ============================================================
# 28. SAVE ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    final_predictions[
        "label"
    ].values,
    final_predictions[
        "probability"
    ].values,
)

plt.figure(
    figsize=(
        6,
        6,
    )
)

plt.plot(
    fpr,
    tpr,
    label=(
        f"EchoJEPA baseline "
        f"(AUC={final_metrics['auc']:.3f})"
    ),
)

plt.plot(
    [
        0,
        1,
    ],
    [
        0,
        1,
    ],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Training ROC - EchoJEPA baseline"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "train_roc.png",
    dpi=300,
)

plt.close()


# ============================================================
# 29. SAVE TRAINING CURVES
# ============================================================

history_df = pd.DataFrame(
    history
)


if not history_df.empty:

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.plot(
        history_df[
            "epoch"
        ],
        history_df[
            "train_auc"
        ],
        marker="o",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training AUC"
    )

    plt.title(
        "EchoJEPA baseline training AUC"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "train_auc_curve.png",
        dpi=300,
    )

    plt.close()


    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.plot(
        history_df[
            "epoch"
        ],
        history_df[
            "train_loss"
        ],
        marker="o",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training loss"
    )

    plt.title(
        "EchoJEPA baseline training loss"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "train_loss_curve.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# 30. SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "EchoJEPA BASELINE TRAINING COMPLETED"
)

print(
    "=" * 80
)

print(
    "Train AUC         :",
    f"{final_metrics['auc']:.4f}"
)

print(
    "Accuracy          :",
    f"{final_metrics['accuracy']:.4f}"
)

print(
    "F1                :",
    f"{final_metrics['f1']:.4f}"
)

print(
    "Sensitivity       :",
    f"{final_metrics['sensitivity']:.4f}"
)

print(
    "Specificity       :",
    f"{final_metrics['specificity']:.4f}"
)

print(
    "\nFinal model:"
)

print(
    FINAL_MODEL_PATH
)

print(
    "\nPredictions:"
)

print(
    TRAIN_PREDICTIONS_PATH
)

print(
    "=" * 80
)

