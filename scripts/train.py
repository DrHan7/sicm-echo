


import copy
import gc
import math
import os
import random
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

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from tqdm import tqdm


def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Set the {name} environment variable before running this script."
        )
    return Path(value).expanduser()


VIDEO_ROOT = required_env_path("SICM_VIDEO_ROOT")
LABEL_CSV = required_env_path("SICM_LABEL_CSV")
CHECKPOINT_DIR = Path(
    os.environ.get("SICM_CHECKPOINT_DIR", "checkpoints")
).expanduser()
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
OUTPUT_DIR = Path(
    os.environ.get("SICM_OUTPUT_DIR", "outputs/full_model")
).expanduser()
CYCLE_ROOT = OUTPUT_DIR / "cycles"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CYCLE_ROOT.mkdir(parents=True, exist_ok=True)
CYCLE_MANIFEST_PATH = OUTPUT_DIR / "multi_cycle_manifest.csv"


QUICK_TEST = False


FORCE_REPROCESS_CYCLES = True


if QUICK_TEST:

    SSL_EPOCHS = 1
    STAGE_A_EPOCHS = 2
    STAGE_B_EPOCHS = 3

else:

    SSL_EPOCHS = 10
    STAGE_A_EPOCHS = 10
    STAGE_B_EPOCHS = 20


TOTAL_CLS_EPOCHS = (
    STAGE_A_EPOCHS
    + STAGE_B_EPOCHS
)


SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DEVICE_TYPE = (
    DEVICE.type
)

USE_AMP = True

AMP_ENABLED = (
    USE_AMP
    and DEVICE_TYPE == "cuda"
)


NUM_WORKERS = 0

MANIFEST_FLUSH_EVERY = 50


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


CLS_BATCH_SIZE = 1


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


USE_LVEF_AUXILIARY = False
LVEF_COLUMN = "lvef"
LVEF_AUX_WEIGHT = 0.20


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


print("=" * 80)
print(
    "SICM SINGLE-CYCLE PIPELINE -- "
    "EchoJEPA ViT-L base model"
)
print(
    "Training set only -- "
    "no validation/test split"
)
print("=" * 80)

print(
    "DEVICE       :",
    DEVICE,
)

print(
    "AMP          :",
    AMP_ENABLED,
)

print(
    "QUICK_TEST   :",
    QUICK_TEST,
)

print(
    "Clip geometry:",
    f"{NUM_FRAMES} frames x "
    f"{IMAGE_SIZE}x{IMAGE_SIZE}",
)

print(
    "Token grid   :",
    f"{TEMPORAL_TOKENS} x "
    f"{SPATIAL_SIDE} x "
    f"{SPATIAL_SIDE} "
    f"= {NUM_TOKENS} tokens",
)

print(
    "Feature dim  :",
    FEATURE_DIM,
)

if DEVICE_TYPE == "cuda":

    print(
        "GPU          :",
        torch.cuda.get_device_name(
            0
        ),
    )

    total_mem = (
        torch.cuda
        .get_device_properties(
            0
        )
        .total_memory
        / 1024 ** 3
    )

    print(
        "GPU memory   :",
        f"{total_mem:.2f} GB",
    )

print("=" * 80)


for _path, _description in [
    (
        VIDEO_ROOT,
        "VIDEO_ROOT",
    ),
    (
        LABEL_CSV,
        "LABEL_CSV",
    ),
    (
        ECHOJEPA_REPO_DIR,
        "EchoJEPA repository",
    ),
    (
        ECHOJEPA_CHECKPOINT,
        "vitl-vmix22m-pt220-c55.pt",
    ),
    (
        LV_SEGMENTATION_CHECKPOINT,
        "deeplabv3_resnet50_random.pt",
    ),
]:

    if not _path.exists():

        raise FileNotFoundError(
            f"Cannot find {_description}:\\n"
            f"{_path}"
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

label_df = label_df.drop_duplicates(subset=["id"], keep="first")


def scan_videos(root, label_ids):

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

        sample_id = path.stem
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
print("IDs without matching file   :", n_missing)

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

full_df = full_df.drop(columns=["_merge"])

invalid_label_mask = ~full_df["label"].isin([0, 1])

if invalid_label_mask.any():
    invalid_label_path = OUTPUT_DIR / "ids_with_invalid_labels.csv"
    full_df.loc[
        invalid_label_mask,
        ["id", "video_path", "label"],
    ].to_csv(
        invalid_label_path,
        index=False,
        encoding="utf-8-sig",
    )
    raise RuntimeError(
        "Training stopped because every matched ID must have a valid "
        f"binary label (0/1). Details: {invalid_label_path}"
    )

labeled_df = full_df[full_df["label"].isin([0, 1])].copy()

print("IDs with label 0/1:", len(labeled_df))
print("\nLabel distribution:")
print(
    labeled_df["label"]
    .astype(int)
    .value_counts()
    .sort_index()
)

if labeled_df["label"].nunique() != 2:
    raise RuntimeError(
        "Binary classification requires both label=0 and label=1."
    )


def load_video_tchw(npy_path):
    array = np.asarray(np.load(npy_path, mmap_mode="r"))

    if array.ndim == 3:

        x = torch.from_numpy(array.copy()).float().unsqueeze(1)
    elif array.ndim == 4 and array.shape[-1] in (1, 3):

        x = torch.from_numpy(array.copy()).float().permute(0, 3, 1, 2)
    elif array.ndim == 4 and array.shape[1] in (1, 3):

        x = torch.from_numpy(array.copy()).float()
    else:
        raise ValueError(f"Unsupported video shape {array.shape}: {npy_path}")

    return x


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
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    else:
        gray = x

    vmin = float(gray.min())
    vmax = float(gray.max())

    if vmin >= 0 and vmax <= 1.5:
        gray = gray * 255.0
    elif vmin < 0 or vmax > 255.0:
        if vmax > vmin:
            gray = (gray - vmin) / (vmax - vmin) * 255.0
        else:
            gray = torch.zeros_like(gray)

    return gray.clamp(0, 255)


def estimate_mean_std(dataframe, max_videos, frames_per_video):
    if len(dataframe) == 0:
        raise RuntimeError("No videos available to estimate normalization.")

    if len(dataframe) <= max_videos:
        selected = dataframe
    else:
        selected = dataframe.sample(n=max_videos, random_state=SEED)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    print("\nEstimating dataset normalization...")

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Normalization"):
        x = load_video_tchw(row["video_path"])
        n_frames = x.shape[0]

        if n_frames > frames_per_video:
            indices = (
                np.linspace(0, n_frames - 1, frames_per_video).round().astype(int)
            )
            x = x[indices]

        x = to_three_channel_255(x)
        x = F.interpolate(
            x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False
        )

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

    print("Dataset mean (0-255):", [round(v, 3) for v in mean.tolist()])
    print("Dataset std  (0-255):", [round(v, 3) for v in std.tolist()])

    pd.DataFrame(
        {"channel": ["R", "G", "B"], "mean": mean.numpy(), "std": std.numpy()}
    ).to_csv(
        OUTPUT_DIR / "data_normalization.csv", index=False, encoding="utf-8-sig"
    )

    return mean, std


DATA_MEAN, DATA_STD = estimate_mean_std(
    video_df,
    max_videos=NORMALIZATION_SAMPLE_VIDEOS,
    frames_per_video=NORMALIZATION_FRAMES_PER_VIDEO,
)


def build_lv_segmenter(weight_path):
    print("\nBuilding EchoNet-Dynamic DeepLabV3-ResNet50 LV segmenter...")

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

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        cleaned_state[new_key] = value

    result = model.load_state_dict(cleaned_state, strict=False)

    print("LV segmenter missing keys   :", len(result.missing_keys))
    print("LV segmenter unexpected keys:", len(result.unexpected_keys))

    if result.missing_keys or result.unexpected_keys:
        print("  missing examples   :", result.missing_keys[:10])
        print("  unexpected examples:", result.unexpected_keys[:10])
        raise RuntimeError(
            "LV segmentation checkpoint does not match the model architecture."
        )

    model = model.to(DEVICE)
    model.eval()
    print("LV segmentation model loaded successfully.")
    return model


lv_model = build_lv_segmenter(LV_SEGMENTATION_CHECKPOINT)


@torch.no_grad()
def get_lv_area_curve(npy_path, segmenter):
    original = load_video_tchw(npy_path)

    x = to_three_channel_255(original.clone())
    x = F.interpolate(
        x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False
    )
    x = (x - DATA_MEAN.view(1, 3, 1, 1)) / DATA_STD.view(1, 3, 1, 1)

    areas = []
    for start in range(0, x.shape[0], SEG_BATCH_SIZE):
        batch = x[start:start + SEG_BATCH_SIZE].to(DEVICE, non_blocking=True)
        logits = segmenter(batch)["out"][:, 0]
        masks = logits > 0
        areas.append(masks.sum(dim=(1, 2)).cpu().numpy().astype(np.float32))

    return original, np.concatenate(areas)


def robust_area_range(
    area,
):

    q05 = float(
        np.quantile(
            area,
            0.05,
        )
    )

    q95 = float(
        np.quantile(
            area,
            0.95,
        )
    )

    return max(
        q95 - q05,
        1.0,
    )


def find_ed_peaks(
    smooth,
    area_range,
    distance,
    prominence_fraction,
):

    peaks, _ = find_peaks(
        smooth,
        distance=distance,
        prominence=(
            prominence_fraction
            * area_range
        ),
    )

    return peaks


def build_cycle_candidate(
    smooth,
    area_range,
    ed1,
    ed2,
    method,
):

    ed1 = int(
        ed1
    )

    ed2 = int(
        ed2
    )

    if ed2 < ed1:

        ed1, ed2 = (
            ed2,
            ed1,
        )

    period = (
        ed2 - ed1
    )

    if period < 1:

        return None

    es = (
        ed1
        + int(
            np.argmin(
                smooth[
                    ed1:
                    ed2 + 1
                ]
            )
        )
    )

    mean_ed_area = (
        smooth[
            ed1
        ]
        + smooth[
            ed2
        ]
    ) / 2.0

    es_area = float(
        smooth[
            es
        ]
    )

    excursion_fraction = float(
        (
            mean_ed_area
            - es_area
        )
        / area_range
    )

    ed_area_mismatch_fraction = float(
        abs(
            smooth[
                ed1
            ]
            - smooth[
                ed2
            ]
        )
        / area_range
    )

    quality_score = float(
        excursion_fraction
        - 0.25
        * ed_area_mismatch_fraction
    )

    return {
        "method":
            method,

        "ed1":
            ed1,

        "es":
            es,

        "ed2":
            ed2,

        "cycle_frames":
            int(
                period
            ),

        "excursion_fraction":
            excursion_fraction,

        "ed_area_mismatch_fraction":
            ed_area_mismatch_fraction,

        "quality_score":
            quality_score,

        "qc_excursion_low":
            bool(
                excursion_fraction
                <
                QC_MIN_AREA_EXCURSION_FRACTION
            ),

        "qc_ed_mismatch_high":
            bool(
                ed_area_mismatch_fraction
                >
                QC_MAX_ED_AREA_MISMATCH_FRACTION
            ),

        "qc_cycle_length_outside_preferred":
            bool(
                period
                <
                MIN_CYCLE_FRAMES

                or

                period
                >
                MAX_CYCLE_FRAMES
            ),
    }


def detect_all_cycles(
    raw_area,
):


    raw_area = np.asarray(
        raw_area,
        dtype=np.float32,
    )

    n_frames = len(
        raw_area
    )

    if n_frames < 2:

        raise RuntimeError(
            "Video has fewer than 2 frames; "
            "no temporal interval can be formed."
        )

    smooth = gaussian_filter1d(
        raw_area,
        sigma=SMOOTH_SIGMA,
        mode="nearest",
    )

    area_range = robust_area_range(
        smooth
    )

    ed_candidates = find_ed_peaks(
        smooth,
        area_range,
        MIN_ED_DISTANCE,
        ED_PROMINENCE_FRACTION,
    )

    if len(ed_candidates) < 2:
        raise RuntimeError(
            "Fewer than two ED peaks were detected with the primary settings; "
            "cardiac-cycle detection failed."
        )

    cycles = []

    for i in range(len(ed_candidates) - 1):
        candidate = build_cycle_candidate(
            smooth,
            area_range,
            ed_candidates[i],
            ed_candidates[i + 1],
            "primary_peaks",
        )

        if candidate is not None:
            cycles.append(candidate)

    if not cycles:
        raise RuntimeError(
            "No valid adjacent ED-to-ED cycles were found."
        )

    return {
        "cycles": cycles,
        "smooth_area": smooth,
        "ed_candidates": ed_candidates,
        "detection_method": "primary_peaks",
    }


def resample_cycle(
    original_video,
    ed1,
    ed2,
):


    ed1 = int(
        ed1
    )

    ed2 = int(
        ed2
    )

    if ed1 > ed2:

        ed1, ed2 = (
            ed2,
            ed1,
        )

    cycle = original_video[
        ed1:
        ed2 + 1
    ]

    if (
        cycle.shape[
            0
        ]
        < 2
    ):

        raise RuntimeError(
            "Detected cycle contains fewer than 2 frames."
        )

    cycle = to_grayscale_255(
        cycle
    )

    cycle = (
        cycle
        .permute(
            1,
            0,
            2,
            3,
        )
        .unsqueeze(
            0
        )
    )

    cycle = F.interpolate(
        cycle,
        size=(
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="trilinear",
        align_corners=False,
    )

    cycle = (
        cycle
        .squeeze(
            0
        )
        .squeeze(
            0
        )
        .round()
        .clamp(
            0,
            255,
        )
        .to(
            torch.uint8
        )
        .cpu()
        .numpy()
    )

    return cycle


def build_lv_area_representation(
    smooth_area,
    ed1,
    es,
    ed2,
):


    ed1 = int(
        ed1
    )

    es = int(
        es
    )

    ed2 = int(
        ed2
    )

    if ed1 > ed2:

        ed1, ed2 = (
            ed2,
            ed1,
        )

    segment = np.asarray(
        smooth_area[
            ed1:
            ed2 + 1
        ],
        dtype=np.float32,
    )

    if len(
        segment
    ) < 2:

        raise RuntimeError(
            "LV-area segment contains fewer than 2 points."
        )

    mean_ed = float(
        (
            segment[
                0
            ]
            + segment[
                -1
            ]
        )
        / 2.0
    )

    mean_ed = max(
        mean_ed,
        1e-6,
    )

    normalized = (
        segment
        / mean_ed
    )

    old_x = np.linspace(
        0.0,
        1.0,
        len(
            normalized
        ),
    )

    new_x = np.linspace(
        0.0,
        1.0,
        AREA_CURVE_POINTS,
    )

    curve = np.interp(
        new_x,
        old_x,
        normalized,
    ).astype(
        np.float32
    )

    curve = np.clip(
        curve,
        0.0,
        3.0,
    )

    es_local = int(
        np.argmin(
            segment
        )
    )

    es_area = float(
        segment[
            es_local
        ]
    )

    fac = float(
        (
            mean_ed
            - es_area
        )
        / mean_ed
    )

    systolic_fraction = max(
        es_local
        / max(
            len(
                segment
            )
            - 1,
            1,
        ),
        1e-3,
    )

    diastolic_fraction = max(
        (
            len(
                segment
            )
            - 1
            - es_local
        )
        / max(
            len(
                segment
            )
            - 1,
            1,
        ),
        1e-3,
    )

    systolic_slope = float(
        (
            segment[
                0
            ]
            - es_area
        )
        / mean_ed
        / systolic_fraction
    )

    diastolic_slope = float(
        (
            segment[
                -1
            ]
            - es_area
        )
        / mean_ed
        / diastolic_fraction
    )

    cycle_length_proxy = float(
        min(
            (
                len(
                    segment
                )
                - 1
            )
            / max(
                MAX_CYCLE_FRAMES,
                1,
            ),
            2.0,
        )
    )

    ed_endpoint_mismatch_relative = float(
        abs(
            segment[
                0
            ]
            - segment[
                -1
            ]
        )
        / mean_ed
    )

    physiology = np.asarray(
        [
            fac,
            systolic_slope,
            diastolic_slope,
            cycle_length_proxy,
            ed_endpoint_mismatch_relative,
        ],
        dtype=np.float32,
    )

    return (
        curve,
        physiology,
    )


def save_multicycle_qc(
    study_output,
    raw_area,
    detection,
):

    smooth = detection[
        "smooth_area"
    ]

    pd.DataFrame(
        {
            "frame":
                np.arange(
                    len(
                        raw_area
                    )
                ),

            "lv_area_raw":
                raw_area,

            "lv_area_smooth":
                smooth,
        }
    ).to_csv(
        study_output
        / "lv_area_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not SAVE_QC_PLOTS:

        return

    fig = plt.figure(
        figsize=(
            11,
            5,
        )
    )

    frames = np.arange(
        len(
            raw_area
        )
    )

    plt.plot(
        frames,
        raw_area,
        alpha=0.30,
        label="LV area raw",
    )

    plt.plot(
        frames,
        smooth,
        linewidth=2,
        label="LV area smoothed",
    )

    ed_candidates = np.asarray(
        detection.get(
            "ed_candidates",
            [],
        ),
        dtype=int,
    )

    if len(
        ed_candidates
    ) > 0:

        plt.scatter(
            ed_candidates,
            smooth[
                ed_candidates
            ],
            s=35,
            label="ED candidates",
        )

    cycles = detection[
        "cycles"
    ]

    for index, cycle in enumerate(
        cycles,
        start=1,
    ):

        ed1 = cycle[
            "ed1"
        ]

        ed2 = cycle[
            "ed2"
        ]

        es = cycle[
            "es"
        ]

        plt.axvspan(
            ed1,
            ed2,
            alpha=0.05,
        )

        plt.scatter(
            [
                ed1,
                ed2,
            ],
            smooth[
                [
                    ed1,
                    ed2,
                ]
            ],
            marker="o",
            s=50,
        )

        plt.scatter(
            [
                es
            ],
            smooth[
                [
                    es
                ]
            ],
            marker="x",
            s=55,
        )

        plt.text(
            (
                ed1
                + ed2
            )
            / 2.0,
            max(
                smooth[
                    ed1
                ],
                smooth[
                    ed2
                ],
            ),
            f"C{index}",
            fontsize=8,
        )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "LV segmented area"
    )

    plt.title(
        f"{detection['detection_method']} | "
        f"{len(cycles)} cardiac cycle(s)"
    )

    plt.legend(
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        study_output
        / "lv_area_multicycle_qc.png",
        dpi=180,
    )

    plt.close(
        fig
    )


CYCLE_FILENAME_TEMPLATE = (
    "cycle_{cycle_index:02d}_"
    f"{NUM_FRAMES}f_{IMAGE_SIZE}px.npy"
)

AREA_FILENAME_TEMPLATE = (
    "lv_area_cycle_{cycle_index:02d}_"
    f"{AREA_CURVE_POINTS}p.npy"
)

MANIFEST_COLUMNS = [
    "id",
    "video_path",
    "status",
    "cycle_index",
    "n_cycles_in_study",
    "method",
    "original_frames",
    "ed1",
    "es",
    "ed2",
    "cycle_frames",
    "excursion_fraction",
    "ed_area_mismatch_fraction",
    "quality_score",
    "qc_excursion_low",
    "qc_ed_mismatch_high",
    "qc_cycle_length_outside_preferred",
    "fac",
    "systolic_slope",
    "diastolic_slope",
    "cycle_length_proxy",
    "ed_endpoint_mismatch_relative",
    "cycle_path",
    "area_curve_path",
    "error",
]


def empty_manifest_record(
    sample_id,
    video_path,
):

    record = {
        column:
            np.nan

        for column
        in MANIFEST_COLUMNS
    }

    record[
        "id"
    ] = sample_id

    record[
        "video_path"
    ] = video_path

    record[
        "method"
    ] = ""

    record[
        "cycle_path"
    ] = ""

    record[
        "area_curve_path"
    ] = ""

    record[
        "error"
    ] = ""

    return record


def write_manifest(
    records,
):

    pd.DataFrame(
        records,
        columns=MANIFEST_COLUMNS,
    ).to_csv(
        CYCLE_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def process_all_cycles(
    dataframe,
    segmenter,
):

    previous_groups = {}

    if (
        CYCLE_MANIFEST_PATH.exists()
        and not FORCE_REPROCESS_CYCLES
    ):

        existing = pd.read_csv(
            CYCLE_MANIFEST_PATH,
            dtype={
                "id":
                    str,
            },
        )

        for key, group in existing.groupby(
            "id",
            sort=False,
        ):

            previous_groups[
                str(
                    key
                )
            ] = group.copy()

    records = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LV SEGMENTATION + MULTI-CYCLE EXTRACTION"
    )

    print(
        f"Target visual clip: "
        f"{NUM_FRAMES} frames x "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Target LV-area curve: "
        f"{AREA_CURVE_POINTS} points"
    )

    print(
        "Every adjacent ED-to-ED cycle is retained"
    )

    print(
        "QC metrics are recorded but never used for exclusion"
    )

    print(
        "=" * 80
    )

    for sample_counter, (_, row) in enumerate(
        tqdm(
            dataframe.iterrows(),
            total=len(
                dataframe
            ),
            desc="IDs",
        )
    ):

        sample_id = str(
            row[
                "id"
            ]
        )

        video_path = row[
            "video_path"
        ]

        key = sample_id


        if (
            key in previous_groups
            and not FORCE_REPROCESS_CYCLES
        ):

            old_group = previous_groups[
                key
            ]

            used_rows = old_group[
                old_group[
                    "status"
                ]
                == "USED"
            ]

            cache_valid = (
                len(
                    used_rows
                )
                > 0
                and used_rows[
                    "method"
                ].astype(str).eq("primary_peaks").all()
            )

            if cache_valid:

                for _, old_row in used_rows.iterrows():

                    cycle_path = str(
                        old_row.get(
                            "cycle_path",
                            "",
                        )
                        or ""
                    )

                    area_path = str(
                        old_row.get(
                            "area_curve_path",
                            "",
                        )
                        or ""
                    )

                    if (
                        not cycle_path
                        or not area_path
                        or not Path(
                            cycle_path
                        ).exists()
                        or not Path(
                            area_path
                        ).exists()
                    ):

                        cache_valid = False
                        break

            if cache_valid:

                for _, old_row in old_group.iterrows():

                    records.append(
                        {
                            column:
                                old_row.get(
                                    column,
                                    np.nan,
                                )

                            for column
                            in MANIFEST_COLUMNS
                        }
                    )

                continue

        sample_output = (
            CYCLE_ROOT
            / sample_id
        )

        sample_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            (
                original_video,
                raw_area
            ) = get_lv_area_curve(
                video_path,
                segmenter,
            )

            detection = detect_all_cycles(
                raw_area
            )

            save_multicycle_qc(
                sample_output,
                raw_area,
                detection,
            )

            cycles = detection[
                "cycles"
            ]

            n_cycles = len(
                cycles
            )

            for (
                cycle_index,
                detection_cycle
            ) in enumerate(
                cycles,
                start=1,
            ):

                cycle_array = resample_cycle(
                    original_video,
                    detection_cycle[
                        "ed1"
                    ],
                    detection_cycle[
                        "ed2"
                    ],
                )

                (
                    area_curve,
                    physiology
                ) = build_lv_area_representation(
                    detection[
                        "smooth_area"
                    ],
                    detection_cycle[
                        "ed1"
                    ],
                    detection_cycle[
                        "es"
                    ],
                    detection_cycle[
                        "ed2"
                    ],
                )

                cycle_path = (
                    sample_output
                    / CYCLE_FILENAME_TEMPLATE.format(
                        cycle_index=(
                            cycle_index
                        )
                    )
                )

                area_curve_path = (
                    sample_output
                    / AREA_FILENAME_TEMPLATE.format(
                        cycle_index=(
                            cycle_index
                        )
                    )
                )

                np.save(
                    cycle_path,
                    cycle_array,
                )

                np.save(
                    area_curve_path,
                    area_curve,
                )

                record = empty_manifest_record(
                    sample_id,
                    video_path,
                )

                record.update(
                    {
                        "status":
                            "USED",

                        "cycle_index":
                            cycle_index,

                        "n_cycles_in_study":
                            n_cycles,

                        "method":
                            detection_cycle[
                                "method"
                            ],

                        "original_frames":
                            int(
                                original_video.shape[
                                    0
                                ]
                            ),

                        "ed1":
                            detection_cycle[
                                "ed1"
                            ],

                        "es":
                            detection_cycle[
                                "es"
                            ],

                        "ed2":
                            detection_cycle[
                                "ed2"
                            ],

                        "cycle_frames":
                            detection_cycle[
                                "cycle_frames"
                            ],

                        "excursion_fraction":
                            detection_cycle[
                                "excursion_fraction"
                            ],

                        "ed_area_mismatch_fraction":
                            detection_cycle[
                                "ed_area_mismatch_fraction"
                            ],

                        "quality_score":
                            detection_cycle[
                                "quality_score"
                            ],

                        "qc_excursion_low":
                            detection_cycle[
                                "qc_excursion_low"
                            ],

                        "qc_ed_mismatch_high":
                            detection_cycle[
                                "qc_ed_mismatch_high"
                            ],

                        "qc_cycle_length_outside_preferred":
                            detection_cycle[
                                "qc_cycle_length_outside_preferred"
                            ],

                        "fac":
                            float(
                                physiology[
                                    0
                                ]
                            ),

                        "systolic_slope":
                            float(
                                physiology[
                                    1
                                ]
                            ),

                        "diastolic_slope":
                            float(
                                physiology[
                                    2
                                ]
                            ),

                        "cycle_length_proxy":
                            float(
                                physiology[
                                    3
                                ]
                            ),

                        "ed_endpoint_mismatch_relative":
                            float(
                                physiology[
                                    4
                                ]
                            ),

                        "cycle_path":
                            str(
                                cycle_path
                            ),

                        "area_curve_path":
                            str(
                                area_curve_path
                            ),

                        "error":
                            "",
                    }
                )

                records.append(
                    record
                )

        except Exception as exc:

            record = empty_manifest_record(
                sample_id,
                video_path,
            )

            record[
                "status"
            ] = "ERROR"

            record[
                "cycle_index"
            ] = -1

            record[
                "error"
            ] = repr(
                exc
            )

            records.append(
                record
            )

        if (
            (
                sample_counter
                + 1
            )
            % MANIFEST_FLUSH_EVERY
            == 0
        ):

            write_manifest(
                records
            )

    write_manifest(
        records
    )

    return pd.DataFrame(
        records,
        columns=MANIFEST_COLUMNS,
    )


cycle_manifest = process_all_cycles(
    video_df,
    lv_model,
)


print(
    "\nCycle extraction status:"
)

print(
    cycle_manifest[
        "status"
    ].value_counts(
        dropna=False
    )
)


used_manifest = cycle_manifest[
    cycle_manifest[
        "status"
    ]
    == "USED"
].copy()


n_errors = int(
    (
        cycle_manifest[
            "status"
        ]
        == "ERROR"
    ).sum()
)


if used_manifest.empty:

    raise RuntimeError(
        "No cardiac-cycle instances were generated successfully."
    )


cycles_per_study = (
    used_manifest
    .groupby(
        "id"
    )
    .size()
)


print(
    "\nMulti-cycle summary:"
)

print(
    "Studies with usable cycles :",
    len(
        cycles_per_study
    ),
)

print(
    "Total cardiac cycles       :",
    len(
        used_manifest
    ),
)

print(
    "Mean cycles / study        :",
    f"{cycles_per_study.mean():.2f}",
)

print(
    "Median cycles / study      :",
    f"{cycles_per_study.median():.2f}",
)

print(
    "Maximum cycles / study     :",
    int(
        cycles_per_study.max()
    ),
)


print(
    "\nCycle detection methods:"
)

print(
    used_manifest[
        "method"
    ].value_counts()
)


del lv_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


usable_df = full_df.merge(
    used_manifest[
        [
            "id",
            "cycle_index",
            "n_cycles_in_study",
            "cycle_path",
            "area_curve_path",
            "method",
            "quality_score",
            "fac",
            "systolic_slope",
            "diastolic_slope",
            "cycle_length_proxy",
            "ed_endpoint_mismatch_relative",
            "qc_excursion_low",
            "qc_ed_mismatch_high",
            "qc_cycle_length_outside_preferred",
        ]
    ],
    on="id",
    how="inner",
)


classification_df = usable_df[
    usable_df[
        "label"
    ].isin(
        [
            0,
            1,
        ]
    )
].copy()


classification_df[
    "label"
] = (
    classification_df[
        "label"
    ]
    .astype(
        int
    )
)


ssl_df = (
    classification_df
    .reset_index(
        drop=True
    )
)


if USE_LVEF_AUXILIARY:

    if (
        LVEF_COLUMN
        not in label_df.columns
    ):

        raise RuntimeError(
            f"USE_LVEF_AUXILIARY=True but "
            f"'{LVEF_COLUMN}' is not present in LABEL_CSV."
        )

    lvef_lookup = (
        label_df[
            [
                "id",
                LVEF_COLUMN,
            ]
        ]
        .drop_duplicates(
            subset=[
                "id"
            ]
        )
    )

    classification_df = (
        classification_df.merge(
            lvef_lookup,
            on="id",
            how="left",
        )
    )


study_training_df = (
    classification_df[
        [
            "id",
            "label",
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "DATA USED"
)

print(
    "=" * 80
)

print(
    "Cycles for self-supervised adaptation :",
    len(
        ssl_df
    ),
)

print(
    "Labeled cardiac-cycle instances       :",
    len(
        classification_df
    ),
)

print(
    "Labeled IDs for MIL training           :",
    len(
        study_training_df
    ),
)

print(
    "Unique labeled IDs                     :",
    study_training_df[
        "id"
    ].nunique(),
)

print(
    "Mean labeled cycles / ID               :",
    f"{len(classification_df) / max(len(study_training_df), 1):.2f}",
)

print(
    "\nID-level label distribution:"
)

print(
    study_training_df[
        "label"
    ]
    .value_counts()
    .sort_index()
)

print(
    "=" * 80
)


if (
    study_training_df[
        "label"
    ].nunique()
    != 2
):

    raise RuntimeError(
        "The supervised training set must contain both label=0 and label=1."
    )


classification_df.to_csv(
    OUTPUT_DIR
    / "training_cycle_instances.csv",
    index=False,
    encoding="utf-8-sig",
)


study_training_df.to_csv(
    OUTPUT_DIR
    / "training_ids.csv",
    index=False,
    encoding="utf-8-sig",
)


PHYS_COLUMNS = [
    "fac",
    "systolic_slope",
    "diastolic_slope",
    "cycle_length_proxy",
    "ed_endpoint_mismatch_relative",
]


phys_values = (
    classification_df[
        PHYS_COLUMNS
    ]
    .to_numpy(
        dtype=np.float32
    )
)


if not np.isfinite(
    phys_values
).all():

    raise RuntimeError(
        "Non-finite values were found in the physiological descriptors."
    )


PHYS_MEAN = torch.tensor(
    phys_values.mean(
        axis=0
    ),
    dtype=torch.float32,
)


PHYS_STD = torch.tensor(
    phys_values.std(
        axis=0,
        ddof=0,
    ),
    dtype=torch.float32,
)


PHYS_STD = torch.clamp(
    PHYS_STD,
    min=1e-6,
)


pd.DataFrame(
    {
        "feature":
            PHYS_COLUMNS,

        "training_mean":
            PHYS_MEAN.numpy(),

        "training_std":
            PHYS_STD.numpy(),
    }
).to_csv(
    OUTPUT_DIR
    / "physiology_normalization.csv",
    index=False,
    encoding="utf-8-sig",
)


print(
    "\nTraining-set physiology normalization:"
)


for (
    feature_name,
    feature_mean,
    feature_std
) in zip(
    PHYS_COLUMNS,
    PHYS_MEAN.tolist(),
    PHYS_STD.tolist(),
):

    print(
        f"  {feature_name:34s} "
        f"mean={feature_mean:.6f}  "
        f"std={feature_std:.6f}"
    )


def load_cached_cycle(
    cycle_path,
):

    array = np.load(
        cycle_path
    )

    if array.ndim == 3:

        x = (
            torch.from_numpy(
                array.copy()
            )
            .float()
            .unsqueeze(
                1
            )
            .repeat(
                1,
                3,
                1,
                1,
            )
        )

    elif (
        array.ndim == 4
        and array.shape[-1] == 3
    ):

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
        and array.shape[1] == 3
    ):

        x = torch.from_numpy(
            array.copy()
        ).float()

    else:

        raise ValueError(
            f"Unsupported cached cycle shape: "
            f"{array.shape}"
        )

    if (
        array.dtype == np.uint8
        or float(
            x.max()
        ) > 1.5
    ):

        x = (
            x
            / 255.0
        )

    return x.clamp(
        0.0,
        1.0,
    )


def random_translate(
    cycle,
    max_shift,
):

    dy = random.randint(
        -max_shift,
        max_shift,
    )

    dx = random.randint(
        -max_shift,
        max_shift,
    )

    if (
        dy == 0
        and dx == 0
    ):

        return cycle

    height = cycle.shape[
        -2
    ]

    width = cycle.shape[
        -1
    ]

    padded = F.pad(
        cycle,
        (
            max_shift,
            max_shift,
            max_shift,
            max_shift,
        ),
        value=0.0,
    )

    top = (
        max_shift
        + dy
    )

    left = (
        max_shift
        + dx
    )

    return padded[
        ...,
        top:
        top + height,
        left:
        left + width,
    ]


def augment_cycle(
    cycle,
):


    if (
        AUG_MAX_PHASE_ROLL > 0
        and random.random()
        < AUG_PHASE_ROLL_PROB
    ):

        shift = random.randint(
            -AUG_MAX_PHASE_ROLL,
            AUG_MAX_PHASE_ROLL,
        )

        if shift != 0:

            cycle = torch.roll(
                cycle,
                shifts=shift,
                dims=0,
            )

    if (
        AUG_MAX_TRANSLATE > 0
        and random.random()
        < AUG_TRANSLATE_PROB
    ):

        cycle = random_translate(
            cycle,
            AUG_MAX_TRANSLATE,
        )

    if (
        random.random()
        < AUG_BRIGHTNESS_PROB
    ):

        factor = random.uniform(
            *AUG_BRIGHTNESS_RANGE
        )

        cycle = (
            cycle
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

        mean_value = (
            cycle.mean()
        )

        cycle = (
            (
                cycle
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

        cycle = (
            cycle
            .clamp(
                1e-6,
                1.0,
            )
            .pow(
                gamma
            )
        )

    return cycle


def prepare_echojepa_input(
    cycle,
):

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

    cycle = (
        cycle
        - mean
    ) / std

    return cycle.permute(
        1,
        0,
        2,
        3,
    ).contiguous()


def load_area_curve(
    area_curve_path,
):

    curve = np.asarray(
        np.load(
            area_curve_path
        ),
        dtype=np.float32,
    )

    if curve.shape != (
        AREA_CURVE_POINTS,
    ):

        raise ValueError(
            f"Unexpected LV-area curve shape "
            f"{curve.shape}; expected "
            f"({AREA_CURVE_POINTS},)"
        )

    return torch.from_numpy(
        curve.copy()
    ).float()


class CycleSSLDataset(
    Dataset
):


    def __init__(
        self,
        dataframe,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.id_to_indices = {}

        for sample_id, group in self.df.groupby(
            "id",
            sort=False,
        ):

            self.id_to_indices[
                str(
                    sample_id
                )
            ] = (
                group.index
                .to_list()
            )

        cycles_per_id = (
            self.df
            .groupby(
                "id"
            )
            .size()
        )

        self.n_ids = int(
            len(
                cycles_per_id
            )
        )

        self.n_ids_with_cross_cycle = int(
            (
                cycles_per_id
                >= 2
            ).sum()
        )

        self.n_anchor_cycles_with_cross_cycle = int(
            cycles_per_id[
                cycles_per_id
                >= 2
            ].sum()
        )

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

        anchor_row = self.df.iloc[
            idx
        ]

        sample_id = str(
            anchor_row[
                "id"
            ]
        )

        anchor_cycle = load_cached_cycle(
            anchor_row[
                "cycle_path"
            ]
        )


        view_context = (
            prepare_echojepa_input(
                augment_cycle(
                    anchor_cycle.clone()
                )
            )
        )

        view_token_target = (
            prepare_echojepa_input(
                augment_cycle(
                    anchor_cycle.clone()
                )
            )
        )


        candidate_indices = [
            candidate_idx

            for candidate_idx
            in self.id_to_indices[
                sample_id
            ]

            if candidate_idx
            != idx
        ]

        if len(
            candidate_indices
        ) > 0:

            target_idx = random.choice(
                candidate_indices
            )

            target_cycle = load_cached_cycle(
                self.df.iloc[
                    target_idx
                ][
                    "cycle_path"
                ]
            )

            has_cross_cycle = True

        else:


            target_cycle = anchor_cycle.clone()

            has_cross_cycle = False

        view_cycle_target = (
            prepare_echojepa_input(
                augment_cycle(
                    target_cycle
                )
            )
        )

        return {
            "view_context":
                view_context,

            "view_token_target":
                view_token_target,

            "view_cycle_target":
                view_cycle_target,

            "has_cross_cycle":
                torch.tensor(
                    has_cross_cycle,
                    dtype=torch.bool,
                ),

            "id":
                sample_id,
        }


class SICMMILStudyDataset(
    Dataset
):


    def __init__(
        self,
        cycle_dataframe,
        train=True,
    ):

        self.df = (
            cycle_dataframe
            .reset_index(
                drop=True
            )
        )

        self.train = (
            train
        )

        self.groups = []

        for sample_id, group in self.df.groupby(
            "id",
            sort=False,
        ):

            group = (
                group
                .sort_values(
                    "cycle_index"
                )
                .reset_index(
                    drop=True
                )
            )

            labels = (
                group[
                    "label"
                ]
                .astype(
                    int
                )
                .unique()
            )

            if len(
                labels
            ) != 1:

                raise RuntimeError(
                    f"ID {sample_id} contains inconsistent labels."
                )

            self.groups.append(
                (
                    str(
                        sample_id
                    ),
                    int(
                        labels[
                            0
                        ]
                    ),
                    group,
                )
            )

    def __len__(
        self
    ):

        return len(
            self.groups
        )

    def __getitem__(
        self,
        idx,
    ):

        (
            sample_id,
            label,
            group
        ) = self.groups[
            idx
        ]


        if (
            self.train
            and TRAIN_MAX_CYCLES_PER_STUDY
            > 0
            and len(
                group
            )
            > TRAIN_MAX_CYCLES_PER_STUDY
        ):

            selected_indices = np.random.choice(
                len(
                    group
                ),
                size=(
                    TRAIN_MAX_CYCLES_PER_STUDY
                ),
                replace=False,
            )

            selected_indices = np.sort(
                selected_indices
            )

            selected = (
                group.iloc[
                    selected_indices
                ]
                .reset_index(
                    drop=True
                )
            )

        else:

            selected = group

        cycles = []

        area_curves = []

        physiology = []

        cycle_indices = []

        cycle_paths = []

        methods = []

        for _, row in selected.iterrows():

            cycle = load_cached_cycle(
                row[
                    "cycle_path"
                ]
            )

            if self.train:

                cycle = augment_cycle(
                    cycle
                )

            cycles.append(
                prepare_echojepa_input(
                    cycle
                )
            )

            area_curves.append(
                load_area_curve(
                    row[
                        "area_curve_path"
                    ]
                )
            )

            physiology.append(
                torch.tensor(
                    [
                        float(
                            row[
                                "fac"
                            ]
                        ),
                        float(
                            row[
                                "systolic_slope"
                            ]
                        ),
                        float(
                            row[
                                "diastolic_slope"
                            ]
                        ),
                        float(
                            row[
                                "cycle_length_proxy"
                            ]
                        ),
                        float(
                            row[
                                "ed_endpoint_mismatch_relative"
                            ]
                        ),
                    ],
                    dtype=torch.float32,
                )
            )

            cycle_indices.append(
                int(
                    row[
                        "cycle_index"
                    ]
                )
            )

            cycle_paths.append(
                str(
                    row[
                        "cycle_path"
                    ]
                )
            )

            methods.append(
                str(
                    row[
                        "method"
                    ]
                )
            )

        cycles = torch.stack(
            cycles,
            dim=0,
        )

        area_curves = torch.stack(
            area_curves,
            dim=0,
        )

        physiology = torch.stack(
            physiology,
            dim=0,
        )

        output = {
            "cycles":
                cycles,

            "area_curves":
                area_curves,

            "physiology":
                physiology,

            "label":
                torch.tensor(
                    float(
                        label
                    ),
                    dtype=torch.float32,
                ),

            "id":
                sample_id,

            "cycle_indices":
                cycle_indices,

            "cycle_paths":
                cycle_paths,

            "methods":
                methods,
        }

        if USE_LVEF_AUXILIARY:

            lvef_values = (
                selected[
                    LVEF_COLUMN
                ]
                .dropna()
                .astype(
                    float
                )
                .values
            )

            if len(
                lvef_values
            ) > 0:

                lvef_value = float(
                    lvef_values[
                        0
                    ]
                )

                if lvef_value > 1.5:

                    lvef_value = (
                        lvef_value
                        / 100.0
                    )

                output[
                    "lvef"
                ] = torch.tensor(
                    lvef_value,
                    dtype=torch.float32,
                )

                output[
                    "has_lvef"
                ] = torch.tensor(
                    True
                )

            else:

                output[
                    "lvef"
                ] = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                )

                output[
                    "has_lvef"
                ] = torch.tensor(
                    False
                )

        return output


def mil_collate_fn(
    batch,
):


    if len(
        batch
    ) != 1:

        raise RuntimeError(
            "MIL collate currently requires CLS_BATCH_SIZE=1."
        )

    return batch[
        0
    ]


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
        "Could not identify an encoder state_dict "
        "inside the EchoJEPA checkpoint."
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


def load_echojepa_vitl(
    repo_dir,
    checkpoint_path,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Loading official EchoJEPA ViT-L"
    )

    print(
        "=" * 80
    )


    try:

        from src.models import (
            vision_transformer
            as vit_encoder
        )


    except Exception as exc:

        raise ImportError(
            "Could not import the official EchoJEPA repository.\n"
            f"Repository path: {repo_dir}\n\n"
            "Clone the official EchoJEPA repository into this folder "
            "and install its dependencies, or run `pip install -e .` "
            "inside the repository.\n"
            f"Original import error: {repr(exc)}"
        )


    encoder = (
        vit_encoder.vit_large(
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
    )


    print(
        "Encoder embed_dim:",
        encoder.embed_dim,
    )

    print(
        "Encoder depth    :",
        len(
            encoder.blocks
        ),
    )

    print(
        "Encoder patches  :",
        encoder.num_patches,
    )


    if (
        encoder.embed_dim
        != FEATURE_DIM
    ):

        raise RuntimeError(
            f"Unexpected EchoJEPA feature dim: "
            f"{encoder.embed_dim}; "
            f"expected {FEATURE_DIM}"
        )


    if (
        len(
            encoder.blocks
        )
        != TRANSFORMER_DEPTH
    ):

        raise RuntimeError(
            f"Unexpected EchoJEPA depth: "
            f"{len(encoder.blocks)}; "
            f"expected {TRANSFORMER_DEPTH}"
        )


    if (
        encoder.num_patches
        != NUM_TOKENS
    ):

        raise RuntimeError(
            f"Unexpected token count: "
            f"{encoder.num_patches}; "
            f"expected {NUM_TOKENS}"
        )


    print(
        "\nLoading checkpoint:"
    )

    print(
        checkpoint_path
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


    (
        raw_state,
        selected_key
    ) = _choose_encoder_state_dict(
        checkpoint
    )


    print(
        "Checkpoint encoder key:",
        selected_key,
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


    model_state = (
        encoder.state_dict()
    )


    matched_state = {}


    matched_numel = 0


    total_numel = sum(
        tensor.numel()
        for tensor in model_state.values()
    )


    shape_mismatch = []


    for key, value in (
        cleaned_state.items()
    ):

        if key not in model_state:

            continue


        if (
            model_state[
                key
            ].shape
            != value.shape
        ):

            shape_mismatch.append(
                (
                    key,
                    tuple(
                        value.shape
                    ),
                    tuple(
                        model_state[
                            key
                        ].shape
                    ),
                )
            )

            continue


        matched_state[
            key
        ] = value


        matched_numel += (
            value.numel()
        )


    match_ratio = (
        matched_numel
        / max(
            total_numel,
            1,
        )
    )


    print(
        "Matched parameter ratio:",
        f"{100 * match_ratio:.2f}%",
    )


    if shape_mismatch:

        print(
            "Shape mismatch examples:",
            shape_mismatch[:5],
        )


    if match_ratio < 0.95:

        raise RuntimeError(
            "Less than 95% of EchoJEPA encoder parameters matched. "
            "This usually means the checkpoint variant does not match "
            "the ViT-L V-JEPA 2 architecture used by this script."
        )


    result = encoder.load_state_dict(
        matched_state,
        strict=False,
    )


    print(
        "Missing keys after load   :",
        len(
            result.missing_keys
        ),
    )

    print(
        "Unexpected keys after load:",
        len(
            result.unexpected_keys
        ),
    )


    if result.missing_keys:

        print(
            "Missing examples:",
            result.missing_keys[:10],
        )


    if result.unexpected_keys:

        print(
            "Unexpected examples:",
            result.unexpected_keys[:10],
        )


    print(
        "EchoJEPA ViT-L loaded successfully."
    )


    return encoder


base_backbone = load_echojepa_vitl(
    ECHOJEPA_REPO_DIR,
    ECHOJEPA_CHECKPOINT,
)


def tokens_to_spatial_grid(
    tokens,
):


    batch_size = (
        tokens.shape[0]
    )

    if (
        tokens.shape[1]
        != NUM_TOKENS
    ):

        raise RuntimeError(
            f"Expected {NUM_TOKENS} EchoJEPA tokens, "
            f"got {tokens.shape[1]}"
        )

    tokens = tokens.reshape(
        batch_size,
        TEMPORAL_TOKENS,
        SPATIAL_SIDE
        * SPATIAL_SIDE,
        FEATURE_DIM,
    )

    return tokens


class SpatialAttentionPool(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.input_norm = nn.LayerNorm(
            FEATURE_DIM
        )


        self.temporal_pos = nn.Parameter(
            torch.zeros(
                1,
                TEMPORAL_TOKENS,
                1,
                FEATURE_DIM,
            )
        )

        nn.init.trunc_normal_(
            self.temporal_pos,
            std=0.02,
        )

        self.score = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                128,
            ),
            nn.Tanh(),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        tokens,
    ):


        normalized = self.input_norm(
            tokens
        )


        score_input = (
            normalized
            +
            self.temporal_pos.to(
                dtype=normalized.dtype,
                device=normalized.device,
            )
        )

        scores = self.score(
            score_input
        )

        weights = torch.softmax(
            scores,
            dim=2,
        )


        pooled = torch.sum(
            tokens
            * weights,
            dim=2,
        )

        return (
            pooled,
            weights.squeeze(
                -1
            ),
        )


class TemporalAttentionPool(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.input_norm = nn.LayerNorm(
            FEATURE_DIM
        )


        self.temporal_pos = nn.Parameter(
            torch.zeros(
                1,
                TEMPORAL_TOKENS,
                FEATURE_DIM,
            )
        )

        nn.init.trunc_normal_(
            self.temporal_pos,
            std=0.02,
        )

        self.temporal_attention = (
            nn.MultiheadAttention(
                embed_dim=FEATURE_DIM,
                num_heads=TEMPORAL_HEADS,
                dropout=TEMPORAL_DROPOUT,
                batch_first=True,
            )
        )

        self.norm1 = nn.LayerNorm(
            FEATURE_DIM
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                FEATURE_DIM * 2,
            ),
            nn.GELU(),
            nn.Dropout(
                TEMPORAL_DROPOUT
            ),
            nn.Linear(
                FEATURE_DIM * 2,
                FEATURE_DIM,
            ),
        )

        self.norm2 = nn.LayerNorm(
            FEATURE_DIM
        )

        self.score = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                128,
            ),
            nn.Tanh(),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):


        x = self.input_norm(
            x
        )

        x = (
            x
            +
            self.temporal_pos.to(
                dtype=x.dtype,
                device=x.device,
            )
        )

        attention_output, _ = (
            self.temporal_attention(
                x,
                x,
                x,
                need_weights=False,
            )
        )

        x = self.norm1(
            x
            + attention_output
        )

        x = self.norm2(
            x
            + self.ffn(
                x
            )
        )

        weights = torch.softmax(
            self.score(
                x
            ),
            dim=1,
        )

        pooled = torch.sum(
            x
            * weights,
            dim=1,
        )

        return (
            pooled,
            weights.squeeze(
                -1
            ),
        )


def create_mask(
    batch_size,
    mask_ratio,
    device,
):

    n_mask = int(
        NUM_TOKENS
        * mask_ratio
    )


    noise = torch.rand(
        batch_size,
        NUM_TOKENS,
        device=device,
    )


    order = torch.argsort(
        noise,
        dim=1,
    )


    mask = torch.zeros(
        batch_size,
        NUM_TOKENS,
        dtype=torch.bool,
        device=device,
    )


    mask.scatter_(
        1,
        order[
            :,
            :n_mask,
        ],
        True,
    )


    return mask


def apply_input_mask(
    clip,
    mask,
):


    batch_size = (
        clip.shape[0]
    )


    grid = mask.reshape(
        batch_size,
        1,
        TEMPORAL_TOKENS,
        SPATIAL_SIDE,
        SPATIAL_SIDE,
    ).to(
        clip.dtype
    )


    grid = F.interpolate(
        grid,
        size=(
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="nearest",
    )


    return clip * (
        1.0
        - grid
    )


class LatentPredictor(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.input_projection = nn.Linear(
            FEATURE_DIM,
            PREDICTOR_DIM,
        )


        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                PREDICTOR_DIM,
            )
        )


        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                NUM_TOKENS,
                PREDICTOR_DIM,
            )
        )


        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=PREDICTOR_DIM,
                nhead=PREDICTOR_HEADS,
                dim_feedforward=(
                    PREDICTOR_DIM
                    * 4
                ),
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )


        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=(
                    PREDICTOR_DEPTH
                ),
            )
        )


        self.output_projection = nn.Linear(
            PREDICTOR_DIM,
            FEATURE_DIM,
        )


        nn.init.trunc_normal_(
            self.mask_token,
            std=0.02,
        )


        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )


    def forward(
        self,
        context_tokens,
        mask,
    ):

        x = self.input_projection(
            context_tokens
        )


        mask_token = (
            self.mask_token
            .to(
                dtype=x.dtype,
                device=x.device,
            )
            .expand_as(
                x
            )
        )


        x = torch.where(
            mask.unsqueeze(
                -1
            ),
            mask_token,
            x,
        )


        x = (
            x
            + self.position_embedding.to(
                dtype=x.dtype,
                device=x.device,
            )
        )


        x = self.transformer(
            x
        )


        return self.output_projection(
            x
        )


def configure_ssl_trainable_blocks(
    encoder,
    n_blocks,
):


    for parameter in (
        encoder.parameters()
    ):

        parameter.requires_grad = False


    for block in (
        encoder.blocks[
            -n_blocks:
        ]
    ):

        for parameter in (
            block.parameters()
        ):

            parameter.requires_grad = True


    for parameter in (
        encoder.norm.parameters()
    ):

        parameter.requires_grad = True


class CycleAwareEchoJEPA(
    nn.Module
):

    def __init__(
        self,
        backbone,
    ):

        super().__init__()

        self.context_backbone = (
            backbone
        )

        configure_ssl_trainable_blocks(
            self.context_backbone,
            SSL_UNFREEZE_LAST_N_BLOCKS,
        )


        self.context_spatial_pool = (
            SpatialAttentionPool()
        )

        self.context_temporal_pool = (
            TemporalAttentionPool()
        )


        self.target_backbone = (
            copy.deepcopy(
                backbone
            )
        )

        self.target_spatial_pool = (
            copy.deepcopy(
                self.context_spatial_pool
            )
        )

        self.target_temporal_pool = (
            copy.deepcopy(
                self.context_temporal_pool
            )
        )

        for parameter in (
            self.target_backbone.parameters()
        ):

            parameter.requires_grad = False

        for parameter in (
            self.target_spatial_pool.parameters()
        ):

            parameter.requires_grad = False

        for parameter in (
            self.target_temporal_pool.parameters()
        ):

            parameter.requires_grad = False

        self.predictor = (
            LatentPredictor()
        )

    def spatial_then_temporal(
        self,
        tokens,
        spatial_pool,
        temporal_pool,
    ):

        spatial_grid = (
            tokens_to_spatial_grid(
                tokens
            )
        )

        (
            temporal_features,
            spatial_weights
        ) = spatial_pool(
            spatial_grid
        )

        (
            cycle_feature,
            temporal_weights
        ) = temporal_pool(
            temporal_features
        )

        return (
            cycle_feature,
            spatial_weights,
            temporal_weights,
        )

    def forward(
        self,
        view_context,
        view_token_target,
        view_cycle_target,
        has_cross_cycle,
        mask,
    ):


        masked_view = apply_input_mask(
            view_context,
            mask,
        )

        context_masked_tokens = (
            self.context_backbone(
                masked_view
            )
        )

        if (
            context_masked_tokens.shape[
                1
            ]
            != NUM_TOKENS
        ):

            raise RuntimeError(
                "The EchoJEPA context encoder did not return "
                "the expected full token sequence."
            )

        predicted_tokens = (
            self.predictor(
                context_masked_tokens,
                mask,
            )
        )


        with torch.no_grad():

            token_target_tokens = (
                self.target_backbone(
                    view_token_target
                )
            )


        predicted_masked = F.normalize(
            predicted_tokens[
                mask
            ].float(),
            dim=-1,
        )

        token_target_masked = F.normalize(
            token_target_tokens[
                mask
            ]
            .float()
            .detach(),
            dim=-1,
        )

        prediction_loss = (
            F.smooth_l1_loss(
                predicted_masked,
                token_target_masked,
            )
        )


        has_cross_cycle = (
            has_cross_cycle
            .to(
                device=view_context.device
            )
            .bool()
            .view(
                -1
            )
        )

        valid_cross_cycle_count = int(
            has_cross_cycle
            .sum()
            .item()
        )

        if (
            CYCLE_SSL_LOSS_WEIGHT
            > 0.0
            and valid_cross_cycle_count
            > 0
        ):

            if (
                SSL_USE_UNMASKED_CONTEXT_FOR_CYCLE_LOSS
            ):

                context_cycle_tokens = (
                    self.context_backbone(
                        view_context
                    )
                )

            else:

                context_cycle_tokens = (
                    context_masked_tokens
                )

            (
                context_feature,
                context_spatial_weights,
                context_temporal_weights
            ) = self.spatial_then_temporal(
                context_cycle_tokens,
                self.context_spatial_pool,
                self.context_temporal_pool,
            )

            with torch.no_grad():

                cycle_target_tokens = (
                    self.target_backbone(
                        view_cycle_target
                    )
                )

                (
                    cycle_target_feature,
                    _,
                    _
                ) = self.spatial_then_temporal(
                    cycle_target_tokens,
                    self.target_spatial_pool,
                    self.target_temporal_pool,
                )

            context_feature = F.normalize(
                context_feature.float(),
                dim=-1,
            )

            cycle_target_feature = F.normalize(
                cycle_target_feature.float(),
                dim=-1,
            ).detach()

            cross_cycle_cosine_all = (
                context_feature
                * cycle_target_feature
            ).sum(
                dim=-1
            )

            valid_cross_cycle_cosine = (
                cross_cycle_cosine_all[
                    has_cross_cycle
                ]
            )

            cycle_loss = (
                1.0
                - valid_cross_cycle_cosine
            ).mean()

            cross_cycle_positive_cosine = (
                valid_cross_cycle_cosine
                .mean()
            )

        else:

            cycle_loss = torch.zeros(
                (),
                device=(
                    view_context.device
                ),
            )


            cross_cycle_positive_cosine = (
                torch.zeros(
                    (),
                    device=(
                        view_context.device
                    ),
                )
            )

            context_spatial_weights = torch.zeros(
                view_context.shape[
                    0
                ],
                TEMPORAL_TOKENS,
                SPATIAL_SIDE
                * SPATIAL_SIDE,
                device=(
                    view_context.device
                ),
            )

            context_temporal_weights = torch.zeros(
                view_context.shape[
                    0
                ],
                TEMPORAL_TOKENS,
                device=(
                    view_context.device
                ),
            )

        total_loss = (
            prediction_loss
            +
            CYCLE_SSL_LOSS_WEIGHT
            * cycle_loss
        )

        return (
            total_loss,
            prediction_loss,
            cycle_loss,
            cross_cycle_positive_cosine,
            valid_cross_cycle_count,
            context_spatial_weights,
            context_temporal_weights,
        )

    @torch.no_grad()
    def update_target(
        self,
        momentum,
    ):

        pairs = [
            (
                self.context_backbone,
                self.target_backbone,
            ),
            (
                self.context_spatial_pool,
                self.target_spatial_pool,
            ),
            (
                self.context_temporal_pool,
                self.target_temporal_pool,
            ),
        ]

        for (
            online_module,
            target_module
        ) in pairs:

            for (
                online_parameter,
                target_parameter
            ) in zip(
                online_module.parameters(),
                target_module.parameters(),
            ):

                target_parameter.data.mul_(
                    momentum
                ).add_(
                    online_parameter.data,
                    alpha=(
                        1.0
                        - momentum
                    ),
                )

            for (
                online_buffer,
                target_buffer
            ) in zip(
                online_module.buffers(),
                target_module.buffers(),
            ):

                target_buffer.data.copy_(
                    online_buffer.data
                )


ssl_dataset = (
    CycleSSLDataset(
        ssl_df
    )
)


print(
    "\nCross-cycle SSL availability:"
)


print(
    "  IDs in SSL set                  :",
    ssl_dataset.n_ids,
)


print(
    "  IDs with >=2 detected cycles    :",
    ssl_dataset.n_ids_with_cross_cycle,
)


print(
    "  anchor cycles eligible for "
    "cross-cycle loss                 :",
    ssl_dataset.n_anchor_cycles_with_cross_cycle,
)


print(
    "  total SSL anchor cycles         :",
    len(
        ssl_dataset
    ),
)


ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=SSL_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE_TYPE == "cuda"
    ),
    drop_last=False,
)


ssl_model = (
    CycleAwareEchoJEPA(
        base_backbone
    )
    .to(
        DEVICE
    )
)


ssl_trainable = [
    parameter

    for parameter
    in ssl_model.parameters()

    if parameter.requires_grad
]


print(
    "\nSSL trainable parameters:"
)


print(
    f"{sum(p.numel() for p in ssl_trainable):,}"
)


ssl_optimizer = torch.optim.AdamW(
    ssl_trainable,
    lr=SSL_LR,
    betas=(
        0.9,
        0.95,
    ),
    weight_decay=(
        SSL_WEIGHT_DECAY
    ),
)


steps_per_epoch = max(
    1,
    math.ceil(
        len(
            ssl_loader
        )
        /
        SSL_ACCUMULATION_STEPS
    ),
)


total_ssl_steps = (
    steps_per_epoch
    * SSL_EPOCHS
)


warmup_steps = max(
    int(
        total_ssl_steps
        * 0.05
    ),
    1,
)


def ssl_lr_lambda(
    step,
):

    if step < warmup_steps:

        return (
            step + 1
        ) / warmup_steps


    progress = (
        step
        - warmup_steps
    ) / max(
        total_ssl_steps
        - warmup_steps,
        1,
    )


    progress = min(
        progress,
        1.0,
    )


    return (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


ssl_scheduler = (
    torch.optim.lr_scheduler
    .LambdaLR(
        ssl_optimizer,
        ssl_lr_lambda,
    )
)


ssl_scaler = torch.amp.GradScaler(
    DEVICE_TYPE,
    enabled=AMP_ENABLED,
)


print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 1: THREE-VIEW PHYSIOLOGY-AWARE EchoJEPA SSL"
)

print(
    "Same-cycle masked JEPA + same-ID cross-cycle consistency"
)

print(
    f"Only the final "
    f"{SSL_UNFREEZE_LAST_N_BLOCKS} "
    f"EchoJEPA blocks are adapted"
)

print(
    "=" * 80
)


ssl_history = []


optimizer_step = 0


for epoch in range(
    1,
    SSL_EPOCHS + 1,
):

    ssl_model.train()

    ssl_model.target_backbone.eval()

    ssl_model.target_spatial_pool.eval()

    ssl_model.target_temporal_pool.eval()


    running_total = 0.0

    running_prediction = 0.0

    running_cycle = 0.0

    running_cross_cycle_cosine_sum = 0.0

    running_cross_cycle_pairs = 0

    n_batches = 0


    ssl_optimizer.zero_grad(
        set_to_none=True
    )


    progress_bar = tqdm(
        ssl_loader,
        desc=(
            f"EchoJEPA SSL "
            f"{epoch}/"
            f"{SSL_EPOCHS}"
        ),
    )


    for (
        batch_idx,
        batch
    ) in enumerate(
        progress_bar
    ):

        view_context = (
            batch[
                "view_context"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )


        view_token_target = (
            batch[
                "view_token_target"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )


        view_cycle_target = (
            batch[
                "view_cycle_target"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )


        has_cross_cycle = (
            batch[
                "has_cross_cycle"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )


        mask = create_mask(
            view_context.shape[
                0
            ],
            SSL_MASK_RATIO,
            DEVICE,
        )


        with torch.amp.autocast(
            device_type=DEVICE_TYPE,
            enabled=AMP_ENABLED,
        ):

            (
                total_loss,
                prediction_loss,
                cycle_loss,
                cross_cycle_positive_cosine,
                valid_cross_cycle_count,
                _,
                _
            ) = ssl_model(
                view_context,
                view_token_target,
                view_cycle_target,
                has_cross_cycle,
                mask,
            )


            scaled_loss = (
                total_loss
                /
                SSL_ACCUMULATION_STEPS
            )


        ssl_scaler.scale(
            scaled_loss
        ).backward()


        running_total += float(
            total_loss.item()
        )


        running_prediction += float(
            prediction_loss.item()
        )


        running_cycle += float(
            cycle_loss.item()
        )


        if (
            valid_cross_cycle_count
            > 0
        ):

            running_cross_cycle_cosine_sum += (
                float(
                    cross_cycle_positive_cosine
                    .detach()
                    .float()
                    .item()
                )
                * valid_cross_cycle_count
            )

            running_cross_cycle_pairs += (
                valid_cross_cycle_count
            )


        n_batches += 1


        is_last_batch = (
            batch_idx + 1
            == len(
                ssl_loader
            )
        )


        if (
            (
                batch_idx + 1
            )
            % SSL_ACCUMULATION_STEPS
            == 0
            or is_last_batch
        ):

            ssl_scaler.unscale_(
                ssl_optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                ssl_trainable,
                max_norm=1.0,
            )


            ssl_scaler.step(
                ssl_optimizer
            )


            ssl_scaler.update()


            ssl_optimizer.zero_grad(
                set_to_none=True
            )


            ssl_scheduler.step()


            progress = (
                optimizer_step
                / max(
                    total_ssl_steps
                    - 1,
                    1,
                )
            )


            momentum = (
                EMA_START
                +
                (
                    EMA_END
                    - EMA_START
                )
                * min(
                    progress,
                    1.0,
                )
            )


            ssl_model.update_target(
                momentum
            )


            optimizer_step += 1


        mean_cross_cycle_cosine = (
            running_cross_cycle_cosine_sum
            / max(
                running_cross_cycle_pairs,
                1,
            )
        )


        progress_bar.set_postfix(
            {
                "total":
                    f"{running_total / n_batches:.5f}",

                "pred":
                    f"{running_prediction / n_batches:.5f}",

                "cross":
                    f"{running_cycle / n_batches:.5f}",

                "sameIDcos":
                    (
                        f"{mean_cross_cycle_cosine:.4f}"
                        if running_cross_cycle_pairs
                        > 0
                        else "NA"
                    ),

                "lr":
                    f"{ssl_optimizer.param_groups[0]['lr']:.2e}",
            }
        )


    denominator = max(
        n_batches,
        1,
    )


    epoch_cross_cycle_cosine = (
        running_cross_cycle_cosine_sum
        / max(
            running_cross_cycle_pairs,
            1,
        )
        if running_cross_cycle_pairs
        > 0
        else np.nan
    )


    ssl_history.append(
        {
            "epoch":
                epoch,

            "total_ssl_loss":
                running_total
                / denominator,

            "same_cycle_latent_prediction_loss":
                running_prediction
                / denominator,

            "same_id_cross_same_id_cross_cycle_consistency_loss":
                running_cycle
                / denominator,

            "same_id_cross_cycle_cosine":
                epoch_cross_cycle_cosine,

            "cross_cycle_pairs":
                running_cross_cycle_pairs,
        }
    )


    pd.DataFrame(
        ssl_history
    ).to_csv(
        OUTPUT_DIR
        / "ssl_history.csv",
        index=False,
        encoding="utf-8-sig",
    )


    print(
        f"[SSL epoch "
        f"{epoch}/"
        f"{SSL_EPOCHS}] "
        f"total="
        f"{running_total / denominator:.5f}  "
        f"same-cycle-pred="
        f"{running_prediction / denominator:.5f}  "
        f"cross-cycle="
        f"{running_cycle / denominator:.5f}  "
        f"same-ID-cos="
        f"{epoch_cross_cycle_cosine:.4f}  "
        f"pairs="
        f"{running_cross_cycle_pairs}"
    )


SSL_CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "echojepa_cycle_ssl_pretrained.pt"
)


torch.save(
    {
        "backbone_state_dict":
            ssl_model
            .context_backbone
            .state_dict(),

        "spatial_pool_state_dict":
            ssl_model
            .context_spatial_pool
            .state_dict(),

        "temporal_pool_state_dict":
            ssl_model
            .context_temporal_pool
            .state_dict(),

        "ssl_epochs":
            SSL_EPOCHS,

        "ssl_unfreeze_last_n_blocks":
            SSL_UNFREEZE_LAST_N_BLOCKS,

        "cycle_ssl_loss_weight":
            CYCLE_SSL_LOSS_WEIGHT,

        "ssl_token_target":
            "independent augmentation of the same cardiac cycle",

        "ssl_cycle_target":
            "different cardiac cycle from the same ID",

        "ssl_cross_cycle_only_when_multiple_cycles":
            True,

        "ssl_use_unmasked_context_for_cross_cycle_loss":
            SSL_USE_UNMASKED_CONTEXT_FOR_CYCLE_LOSS,

        "base_model":
            "EchoJEPA ViT-L V-JEPA2 "
            "vitl-vmix22m-pt220-c55",

        "num_frames":
            NUM_FRAMES,

        "image_size":
            IMAGE_SIZE,

        "patch_size":
            PATCH_SIZE,

        "tubelet_size":
            TUBELET_SIZE,
    },
    SSL_CHECKPOINT_PATH,
)


classification_backbone = (
    ssl_model
    .context_backbone
    .cpu()
)


classification_spatial_pool = (
    ssl_model
    .context_spatial_pool
    .cpu()
)


classification_temporal_pool = (
    ssl_model
    .context_temporal_pool
    .cpu()
)


ssl_model.context_backbone = (
    nn.Identity()
)


ssl_model.context_spatial_pool = (
    nn.Identity()
)


ssl_model.context_temporal_pool = (
    nn.Identity()
)


del ssl_model
del ssl_optimizer
del ssl_scheduler
del ssl_scaler
del ssl_trainable
del base_backbone


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


class LVAreaEncoder(
    nn.Module
):

    def __init__(
        self,
        phys_mean,
        phys_std,
    ):

        super().__init__()


        self.register_buffer(
            "phys_mean",
            phys_mean.clone()
        )

        self.register_buffer(
            "phys_std",
            phys_std.clone()
        )


        self.curve_encoder = nn.Sequential(
            nn.Linear(
                AREA_CURVE_POINTS,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                128,
                96,
            ),
            nn.GELU(),
        )


        self.physiology_encoder = nn.Sequential(
            nn.Linear(
                5,
                32,
            ),
            nn.GELU(),
        )


        self.fusion = nn.Sequential(
            nn.Linear(
                96 + 32,
                AREA_EMBED_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                AREA_EMBED_DIM
            ),
        )

    def forward(
        self,
        area_curves,
        physiology,
    ):

        curve_feature = (
            self.curve_encoder(
                area_curves
            )
        )

        physiology = (
            physiology
            - self.phys_mean
        ) / (
            self.phys_std
            + 1e-6
        )

        physiology_feature = (
            self.physiology_encoder(
                physiology
            )
        )

        return self.fusion(
            torch.cat(
                [
                    curve_feature,
                    physiology_feature,
                ],
                dim=-1,
            )
        )


class CycleMILAttention(
    nn.Module
):

    def __init__(
        self,
        dim,
    ):

        super().__init__()

        self.score = nn.Sequential(
            nn.LayerNorm(
                dim
            ),
            nn.Linear(
                dim,
                MIL_ATTENTION_DIM,
            ),
            nn.Tanh(),
            nn.Linear(
                MIL_ATTENTION_DIM,
                1,
            ),
        )

    def forward(
        self,
        cycle_features,
    ):


        logits = (
            self.score(
                cycle_features
            )
            .squeeze(
                -1
            )
        )

        weights = torch.softmax(
            logits,
            dim=0,
        )

        study_feature = torch.sum(
            cycle_features
            * weights.unsqueeze(
                -1
            ),
            dim=0,
        )

        return (
            study_feature,
            weights,
        )


class SICMMultiCycleMILClassifier(
    nn.Module
):

    def __init__(
        self,
        backbone,
        spatial_pool=None,
        temporal_pool=None,
    ):

        super().__init__()

        self.backbone = (
            backbone
        )

        self.spatial_pool = (
            SpatialAttentionPool()
            if spatial_pool is None
            else spatial_pool
        )

        self.temporal_pool = (
            TemporalAttentionPool()
            if temporal_pool is None
            else temporal_pool
        )

        self.area_encoder = (
            LVAreaEncoder(
                phys_mean=PHYS_MEAN,
                phys_std=PHYS_STD,
            )
        )

        self.cycle_fusion = nn.Sequential(
            nn.LayerNorm(
                FEATURE_DIM
                + AREA_EMBED_DIM
            ),
            nn.Linear(
                FEATURE_DIM
                + AREA_EMBED_DIM,
                CYCLE_FUSION_DIM,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),
            nn.LayerNorm(
                CYCLE_FUSION_DIM
            ),
        )

        self.mil_attention = (
            CycleMILAttention(
                CYCLE_FUSION_DIM
            )
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                0.20
            ),
            nn.Linear(
                CYCLE_FUSION_DIM,
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


        if USE_LVEF_AUXILIARY:

            self.lvef_head = nn.Sequential(
                nn.Linear(
                    CYCLE_FUSION_DIM,
                    128,
                ),
                nn.GELU(),
                nn.Linear(
                    128,
                    1,
                ),
                nn.Sigmoid(),
            )

        else:

            self.lvef_head = None

    def backbone_is_trainable(
        self,
    ):

        return any(
            parameter.requires_grad

            for parameter
            in self.backbone.parameters()
        )

    def encode_visual_cycles(
        self,
        cycles,
    ):


        visual_features = []

        spatial_weights_all = []

        temporal_weights_all = []

        backbone_trainable = (
            self.backbone_is_trainable()
        )

        for start in range(
            0,
            cycles.shape[
                0
            ],
            CYCLE_FORWARD_CHUNK_SIZE,
        ):

            chunk = cycles[
                start:
                start
                + CYCLE_FORWARD_CHUNK_SIZE
            ]

            if backbone_trainable:

                tokens = self.backbone(
                    chunk
                )

            else:

                with torch.no_grad():

                    tokens = self.backbone(
                        chunk
                    )

            spatial_grid = (
                tokens_to_spatial_grid(
                    tokens
                )
            )

            (
                temporal_features,
                spatial_weights
            ) = self.spatial_pool(
                spatial_grid
            )

            (
                pooled_visual,
                temporal_weights
            ) = self.temporal_pool(
                temporal_features
            )

            visual_features.append(
                pooled_visual
            )

            spatial_weights_all.append(
                spatial_weights
            )

            temporal_weights_all.append(
                temporal_weights
            )

        return (
            torch.cat(
                visual_features,
                dim=0,
            ),
            torch.cat(
                spatial_weights_all,
                dim=0,
            ),
            torch.cat(
                temporal_weights_all,
                dim=0,
            ),
        )

    def forward(
        self,
        cycles,
        area_curves,
        physiology,
    ):

        (
            visual_feature,
            spatial_weights,
            temporal_weights
        ) = self.encode_visual_cycles(
            cycles
        )

        area_feature = (
            self.area_encoder(
                area_curves,
                physiology,
            )
        )

        fused_cycle_feature = (
            self.cycle_fusion(
                torch.cat(
                    [
                        visual_feature,
                        area_feature,
                    ],
                    dim=-1,
                )
            )
        )

        (
            study_feature,
            cycle_attention
        ) = (
            self.mil_attention(
                fused_cycle_feature
            )
        )

        study_logit = (
            self.classifier(
                study_feature
            )
            .squeeze(
                -1
            )
        )


        if self.lvef_head is not None:

            lvef_prediction = (
                self.lvef_head(
                    study_feature
                )
                .squeeze(
                    -1
                )
            )

        else:

            lvef_prediction = None

        return {
            "study_logit":
                study_logit,

            "cycle_attention":
                cycle_attention,

            "spatial_attention":
                spatial_weights,

            "temporal_attention":
                temporal_weights,

            "study_feature":
                study_feature,

            "lvef_prediction":
                lvef_prediction,
        }


classifier = (
    SICMMultiCycleMILClassifier(
        classification_backbone,
        spatial_pool=(
            classification_spatial_pool
        ),
        temporal_pool=(
            classification_temporal_pool
        ),
    )
    .to(
        DEVICE
    )
)


if (
    CLS_BATCH_SIZE
    != 1
):

    raise RuntimeError(
        "This variable-length MIL implementation requires "
        "CLS_BATCH_SIZE=1."
    )


train_dataset = (
    SICMMILStudyDataset(
        classification_df,
        train=True,
    )
)


train_eval_dataset = (
    SICMMILStudyDataset(
        classification_df,
        train=False,
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE_TYPE == "cuda"
    ),
    drop_last=False,
    collate_fn=mil_collate_fn,
)


train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE_TYPE == "cuda"
    ),
    drop_last=False,
    collate_fn=mil_collate_fn,
)


if USE_CLASS_POS_WEIGHT:

    n_positive = int(
        (
            study_training_df[
                "label"
            ]
            == 1
        ).sum()
    )

    n_negative = int(
        (
            study_training_df[
                "label"
            ]
            == 0
        ).sum()
    )

    pos_weight_value = (
        n_negative
        / max(
            n_positive,
            1,
        )
    )

    print(
        f"\nStudy-level class balance: "
        f"{n_negative} negative / "
        f"{n_positive} positive "
        f"-> pos_weight="
        f"{pos_weight_value:.3f}"
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [
                    pos_weight_value
                ],
                dtype=torch.float32,
                device=DEVICE,
            )
        )
    )

else:

    criterion = (
        nn.BCEWithLogitsLoss()
    )


lvef_criterion = (
    nn.SmoothL1Loss()
)


def freeze_backbone(
    model,
):

    for parameter in (
        model.backbone.parameters()
    ):

        parameter.requires_grad = False


def unfreeze_last_blocks(
    model,
    n_blocks,
):

    for parameter in (
        model.backbone.parameters()
    ):

        parameter.requires_grad = False

    for block in (
        model
        .backbone
        .blocks[
            -n_blocks:
        ]
    ):

        for parameter in (
            block.parameters()
        ):

            parameter.requires_grad = True

    for parameter in (
        model
        .backbone
        .norm
        .parameters()
    ):

        parameter.requires_grad = True


def print_trainable_parameters(
    model,
):

    total = sum(
        parameter.numel()

        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable:,}"
    )

    print(
        f"Trainable percentage : "
        f"{100 * trainable / max(total, 1):.2f}%"
    )


@torch.no_grad()
def evaluate_training_set(
    model,
    loader,
):

    model.eval()

    study_records = []

    cycle_records = []

    spatial_attention_all = []

    for batch in tqdm(
        loader,
        desc="Training MIL metrics",
        leave=False,
    ):

        cycles = (
            batch[
                "cycles"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        area_curves = (
            batch[
                "area_curves"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        physiology = (
            batch[
                "physiology"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        with torch.amp.autocast(
            device_type=DEVICE_TYPE,
            enabled=AMP_ENABLED,
        ):

            outputs = model(
                cycles,
                area_curves,
                physiology,
            )

        study_probability = float(
            torch.sigmoid(
                outputs[
                    "study_logit"
                ].float()
            )
            .cpu()
            .item()
        )


        cycle_attention = (
            outputs[
                "cycle_attention"
            ]
            .float()
            .cpu()
            .numpy()
        )

        spatial_attention = (
            outputs[
                "spatial_attention"
            ]
            .float()
            .cpu()
            .numpy()
        )

        temporal_attention = (
            outputs[
                "temporal_attention"
            ]
            .float()
            .cpu()
            .numpy()
        )

        spatial_attention_all.append(
            spatial_attention
        )

        label = int(
            batch[
                "label"
            ].item()
        )

        study_records.append(
            {
                "id":
                    batch[
                        "id"
                    ],

                "label":
                    label,

                "probability":
                    study_probability,

                "n_cycles":
                    len(
                        batch[
                            "cycle_indices"
                        ]
                    ),
            }
        )

        for cycle_position, (
            cycle_index,
            cycle_path,
            method
        ) in enumerate(
            zip(
                batch[
                    "cycle_indices"
                ],
                batch[
                    "cycle_paths"
                ],
                batch[
                    "methods"
                ],
            )
        ):

            row = {
                "id":
                    batch[
                        "id"
                    ],

                "label":
                    label,

                "cycle_index":
                    int(
                        cycle_index
                    ),

                "method":
                    method,

                "cycle_path":
                    cycle_path,


                "mil_attention":
                    float(
                        cycle_attention[
                            cycle_position
                        ]
                    ),
            }


            cycle_spatial = (
                spatial_attention[
                    cycle_position
                ]
            )

            flat_peak = int(
                np.argmax(
                    cycle_spatial
                )
            )

            peak_temporal = (
                flat_peak
                // (
                    SPATIAL_SIDE
                    * SPATIAL_SIDE
                )
            )

            peak_spatial = (
                flat_peak
                % (
                    SPATIAL_SIDE
                    * SPATIAL_SIDE
                )
            )

            peak_row = (
                peak_spatial
                // SPATIAL_SIDE
            )

            peak_col = (
                peak_spatial
                % SPATIAL_SIDE
            )

            row[
                "spatial_peak_temporal_index"
            ] = int(
                peak_temporal
                + 1
            )

            row[
                "spatial_peak_row"
            ] = int(
                peak_row
            )

            row[
                "spatial_peak_col"
            ] = int(
                peak_col
            )

            row[
                "spatial_peak_weight"
            ] = float(
                cycle_spatial[
                    peak_temporal,
                    peak_spatial,
                ]
            )

            for temporal_index in range(
                TEMPORAL_TOKENS
            ):

                row[
                    f"temporal_attention_"
                    f"{temporal_index + 1}"
                ] = float(
                    temporal_attention[
                        cycle_position,
                        temporal_index,
                    ]
                )

            cycle_records.append(
                row
            )

    study_df = pd.DataFrame(
        study_records
    )

    cycle_df = pd.DataFrame(
        cycle_records
    )

    y_true = (
        study_df[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    y_prob = (
        study_df[
            "probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    y_pred = (
        y_prob
        >= 0.5
    ).astype(
        int
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    (
        tn,
        fp,
        fn,
        tp
    ) = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    sensitivity = (
        tp
        / max(
            tp
            + fn,
            1,
        )
    )

    specificity = (
        tn
        / max(
            tn
            + fp,
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

        "predicted_0":
            int(
                (
                    y_pred
                    == 0
                ).sum()
            ),

        "predicted_1":
            int(
                (
                    y_pred
                    == 1
                ).sum()
            ),

        "study_df":
            study_df,

        "cycle_df":
            cycle_df,

        "y_true":
            y_true,

        "y_prob":
            y_prob,

        "spatial_attention":
            np.concatenate(
                spatial_attention_all,
                axis=0,
            ),
    }


freeze_backbone(
    classifier
)


print(
    "\nStage A initialization:"
)


print_trainable_parameters(
    classifier
)


def downstream_parameters(
    model,
):

    params = (
        list(
            model
            .spatial_pool
            .parameters()
        )
        +
        list(
            model
            .temporal_pool
            .parameters()
        )
        +
        list(
            model
            .area_encoder
            .parameters()
        )
        +
        list(
            model
            .cycle_fusion
            .parameters()
        )
        +
        list(
            model
            .mil_attention
            .parameters()
        )
        +
        list(
            model
            .classifier
            .parameters()
        )
    )

    if (
        model.lvef_head
        is not None
    ):

        params += list(
            model
            .lvef_head
            .parameters()
        )

    return params


optimizer_stage_a = (
    torch.optim.AdamW(
        [
            {
                "params":
                    list(
                        classifier
                        .spatial_pool
                        .parameters()
                    )
                    +
                    list(
                        classifier
                        .temporal_pool
                        .parameters()
                    ),

                "lr":
                    TEMPORAL_LR,
            },
            {
                "params":
                    list(
                        classifier
                        .area_encoder
                        .parameters()
                    )
                    +
                    list(
                        classifier
                        .cycle_fusion
                        .parameters()
                    )
                    +
                    list(
                        classifier
                        .mil_attention
                        .parameters()
                    )
                    +
                    list(
                        classifier
                        .classifier
                        .parameters()
                    )
                    +
                    (
                        list(
                            classifier
                            .lvef_head
                            .parameters()
                        )
                        if classifier
                        .lvef_head
                        is not None
                        else []
                    ),

                "lr":
                    HEAD_LR,
            },
        ],
        weight_decay=(
            CLS_WEIGHT_DECAY
        ),
    )
)


scheduler_stage_a = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer_stage_a,
        T_max=max(
            STAGE_A_EPOCHS,
            1,
        ),
        eta_min=1e-6,
    )
)


def create_stage_b_optimizer(
    model,
):

    return torch.optim.AdamW(
        [
            {
                "params": [
                    parameter

                    for parameter
                    in model
                    .backbone
                    .parameters()

                    if parameter
                    .requires_grad
                ],

                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    list(
                        model
                        .spatial_pool
                        .parameters()
                    )
                    +
                    list(
                        model
                        .temporal_pool
                        .parameters()
                    ),

                "lr":
                    TEMPORAL_LR,
            },
            {
                "params":
                    list(
                        model
                        .area_encoder
                        .parameters()
                    )
                    +
                    list(
                        model
                        .cycle_fusion
                        .parameters()
                    )
                    +
                    list(
                        model
                        .mil_attention
                        .parameters()
                    )
                    +
                    list(
                        model
                        .classifier
                        .parameters()
                    )
                    +
                    (
                        list(
                            model
                            .lvef_head
                            .parameters()
                        )
                        if model
                        .lvef_head
                        is not None
                        else []
                    ),

                "lr":
                    HEAD_LR,
            },
        ],
        weight_decay=(
            CLS_WEIGHT_DECAY
        ),
    )


current_optimizer = (
    optimizer_stage_a
)


current_scheduler = (
    scheduler_stage_a
)


cls_scaler = torch.amp.GradScaler(
    DEVICE_TYPE,
    enabled=AMP_ENABLED,
)


CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "checkpoint_last.pt"
)


FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "final_model.pt"
)


history = []


print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 2: SICM MULTI-CYCLE MIL FINE-TUNING"
)

print(
    "EchoJEPA visual feature + LV-area physiology + cycle-level MIL"
)

print(
    "All labeled studies are used for training -- reported AUC is IN-SAMPLE"
)

print(
    "=" * 80
)


for epoch in range(
    1,
    TOTAL_CLS_EPOCHS
    + 1,
):

    if (
        epoch
        == STAGE_A_EPOCHS
        + 1
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"STAGE B: unfreezing the last "
            f"{UNFREEZE_LAST_N_BLOCKS} "
            f"EchoJEPA Transformer blocks"
        )

        print(
            "=" * 80
        )

        unfreeze_last_blocks(
            classifier,
            n_blocks=(
                UNFREEZE_LAST_N_BLOCKS
            ),
        )

        print_trainable_parameters(
            classifier
        )

        current_optimizer = (
            create_stage_b_optimizer(
                classifier
            )
        )

        current_scheduler = (
            torch.optim.lr_scheduler
            .CosineAnnealingLR(
                current_optimizer,
                T_max=max(
                    STAGE_B_EPOCHS,
                    1,
                ),
                eta_min=1e-7,
            )
        )

    in_stage_a = (
        epoch
        <= STAGE_A_EPOCHS
    )

    stage_name = (
        "Frozen EchoJEPA + MIL"
        if in_stage_a
        else
        "EchoJEPA last-block + MIL fine-tuning"
    )

    classifier.train()

    if in_stage_a:

        classifier.backbone.eval()

    running_loss = 0.0

    running_sicm_loss = 0.0


    running_lvef_loss = 0.0

    n_studies = 0

    current_optimizer.zero_grad(
        set_to_none=True
    )

    progress_bar = tqdm(
        train_loader,
        desc=(
            f"MIL "
            f"{epoch}/"
            f"{TOTAL_CLS_EPOCHS} "
            f"[{stage_name}]"
        ),
    )

    for (
        study_index,
        batch
    ) in enumerate(
        progress_bar
    ):

        cycles = (
            batch[
                "cycles"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        area_curves = (
            batch[
                "area_curves"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        physiology = (
            batch[
                "physiology"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        label = (
            batch[
                "label"
            ]
            .to(
                DEVICE
            )
        )

        with torch.amp.autocast(
            device_type=DEVICE_TYPE,
            enabled=AMP_ENABLED,
        ):

            outputs = classifier(
                cycles,
                area_curves,
                physiology,
            )

            sicm_loss = criterion(
                outputs[
                    "study_logit"
                ].view(
                    1
                ),
                label.view(
                    1
                ),
            )

            total_loss = (
                sicm_loss
            )


            lvef_loss = torch.zeros(
                (),
                device=DEVICE,
            )

            if (
                USE_LVEF_AUXILIARY
                and bool(
                    batch[
                        "has_lvef"
                    ].item()
                )
            ):

                lvef_target = (
                    batch[
                        "lvef"
                    ]
                    .to(
                        DEVICE
                    )
                )

                lvef_loss = (
                    lvef_criterion(
                        outputs[
                            "lvef_prediction"
                        ].view(
                            1
                        ),
                        lvef_target.view(
                            1
                        ),
                    )
                )

                total_loss = (
                    total_loss
                    +
                    LVEF_AUX_WEIGHT
                    * lvef_loss
                )

            scaled_loss = (
                total_loss
                /
                CLS_ACCUMULATION_STEPS
            )

        cls_scaler.scale(
            scaled_loss
        ).backward()

        running_loss += float(
            total_loss.item()
        )

        running_sicm_loss += float(
            sicm_loss.item()
        )


        running_lvef_loss += float(
            lvef_loss.item()
        )

        n_studies += 1

        is_last_study = (
            study_index
            + 1
            == len(
                train_loader
            )
        )

        if (
            (
                study_index
                + 1
            )
            % CLS_ACCUMULATION_STEPS
            == 0

            or

            is_last_study
        ):

            cls_scaler.unscale_(
                current_optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter

                    for parameter
                    in classifier.parameters()

                    if parameter.requires_grad
                ],
                max_norm=1.0,
            )

            cls_scaler.step(
                current_optimizer
            )

            cls_scaler.update()

            current_optimizer.zero_grad(
                set_to_none=True
            )

        progress_bar.set_postfix(
            {
                "loss":
                    f"{running_loss / n_studies:.4f}",

                "cycles":
                    cycles.shape[
                        0
                    ],
            }
        )

    current_scheduler.step()

    train_loss = (
        running_loss
        / max(
            n_studies,
            1,
        )
    )

    metrics = evaluate_training_set(
        classifier,
        train_eval_loader,
    )

    param_groups = (
        current_optimizer
        .param_groups
    )

    if in_stage_a:

        backbone_lr = 0.0

        temporal_lr = (
            param_groups[
                0
            ][
                "lr"
            ]
        )

        head_lr = (
            param_groups[
                1
            ][
                "lr"
            ]
        )

    else:

        backbone_lr = (
            param_groups[
                0
            ][
                "lr"
            ]
        )

        temporal_lr = (
            param_groups[
                1
            ][
                "lr"
            ]
        )

        head_lr = (
            param_groups[
                2
            ][
                "lr"
            ]
        )

    history.append(
        {
            "epoch":
                epoch,

            "stage":
                stage_name,

            "train_loss":
                train_loss,

            "train_sicm_loss":
                running_sicm_loss
                / max(
                    n_studies,
                    1,
                ),


            "train_lvef_aux_loss":
                running_lvef_loss
                / max(
                    n_studies,
                    1,
                ),

            "train_auc":
                metrics[
                    "auc"
                ],

            "train_accuracy":
                metrics[
                    "accuracy"
                ],

            "train_f1":
                metrics[
                    "f1"
                ],

            "train_sensitivity":
                metrics[
                    "sensitivity"
                ],

            "train_specificity":
                metrics[
                    "specificity"
                ],

            "predicted_0":
                metrics[
                    "predicted_0"
                ],

            "predicted_1":
                metrics[
                    "predicted_1"
                ],

            "backbone_lr":
                backbone_lr,

            "temporal_lr":
                temporal_lr,

            "head_lr":
                head_lr,
        }
    )

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIR
        / "classification_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nEpoch "
        f"{epoch}/"
        f"{TOTAL_CLS_EPOCHS}  "
        f"[{stage_name}]"
    )

    print(
        f"  Train loss        = "
        f"{train_loss:.4f}"
    )

    print(
        f"  Train AUC         = "
        f"{metrics['auc']:.4f}"
    )

    print(
        f"  Train accuracy    = "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Train F1          = "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  Train sensitivity = "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"  Train specificity = "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"  Predicted 0/1     = "
        f"{metrics['predicted_0']} / "
        f"{metrics['predicted_1']}"
    )

    torch.save(
        {
            "epoch":
                epoch,

            "stage":
                stage_name,

            "model_state_dict":
                classifier
                .state_dict(),

            "optimizer_state_dict":
                current_optimizer
                .state_dict(),

            "scheduler_state_dict":
                current_scheduler
                .state_dict(),

            "scaler_state_dict":
                cls_scaler
                .state_dict(),

            "train_auc":
                metrics[
                    "auc"
                ],

            "train_loss":
                train_loss,
        },
        CHECKPOINT_PATH,
    )


torch.save(
    {
        "model_state_dict":
            classifier
            .state_dict(),

        "base_model":
            "EchoJEPA ViT-L V-JEPA2 "
            "vitl-vmix22m-pt220-c55",

        "architecture":
            "EchoJEPA + phase-aware Spatial Attention + phase-aware Temporal Attention + LV-area physiology + multi-cycle MIL",

        "num_frames":
            NUM_FRAMES,

        "image_size":
            IMAGE_SIZE,

        "patch_size":
            PATCH_SIZE,

        "tubelet_size":
            TUBELET_SIZE,

        "feature_dim":
            FEATURE_DIM,

        "temporal_tokens":
            TEMPORAL_TOKENS,

        "area_curve_points":
            AREA_CURVE_POINTS,

        "area_embed_dim":
            AREA_EMBED_DIM,

        "cycle_fusion_dim":
            CYCLE_FUSION_DIM,

        "echojepa_mean":
            ECHOJEPA_MEAN.tolist(),

        "echojepa_std":
            ECHOJEPA_STD.tolist(),

        "classes":
            {
                0:
                    "non-SICM",

                1:
                    "SICM",
            },

        "cycle_definition":
            "all adjacent LV-area ED-to-ED cycles",

        "mil_aggregation":
            "attention",

        "physiology_features":
            [
                "normalized_lv_area_curve",
                "fractional_area_change",
                "systolic_slope",
                "diastolic_slope",
                "cycle_length_proxy",
                "ed_endpoint_mismatch_relative",
            ],

        "physiology_normalization":
            "training-set feature-wise z-score",

        "area_curve_normalization":
            "mean ED area normalization only; no per-cycle LayerNorm",

        "cycle_classifier":
            False,

        "temporal_phase_embedding":
            True,

        "ssl_design":
            "same-cycle masked JEPA + same-ID different-cycle consistency",

        "ssl_cross_cycle_loss_weight":
            CYCLE_SSL_LOSS_WEIGHT,

        "use_lvef_auxiliary":
            USE_LVEF_AUXILIARY,
    },
    FINAL_MODEL_PATH,
)


print(
    f"\nFinal model saved to: "
    f"{FINAL_MODEL_PATH}"
)


final_metrics = (
    evaluate_training_set(
        classifier,
        train_eval_loader,
    )
)


study_predictions = (
    final_metrics[
        "study_df"
    ].copy()
)


study_predictions[
    "prediction"
] = (
    study_predictions[
        "probability"
    ]
    >= 0.5
).astype(
    int
)


study_predictions.to_csv(
    OUTPUT_DIR
    / "train_id_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


final_metrics[
    "cycle_df"
].to_csv(
    OUTPUT_DIR
    / "train_cycle_attention.csv",
    index=False,
    encoding="utf-8-sig",
)


np.save(
    OUTPUT_DIR
    / "train_spatial_attention.npy",
    final_metrics[
        "spatial_attention"
    ],
)


fpr, tpr, _ = roc_curve(
    final_metrics[
        "y_true"
    ],
    final_metrics[
        "y_prob"
    ],
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
        f"Train AUC = "
        f"{final_metrics['auc']:.3f}"
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
    "SICM study-level MIL training ROC "
    "(in-sample)"
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


print(
    "\n"
    + "=" * 80
)

print(
    "PIPELINE COMPLETE"
)

print(
    "=" * 80
)


print(
    "Base model                       : "
    "EchoJEPA ViT-L"
)


print(
    "Architecture                     : "
    "Spatial Attention + Temporal Attention + Multi-cycle MIL + LV-area physiology"
)


print(
    "Original videos found            :",
    len(
        video_df
    ),
)


print(
    "IDs with usable cycles           :",
    len(
        cycles_per_study
    ),
)


print(
    "Labeled cycles used for SSL      :",
    len(
        ssl_df
    ),
)


print(
    "Labeled IDs used for MIL         :",
    len(
        study_training_df
    ),
)


print(
    "Labeled cycle instances          :",
    len(
        classification_df
    ),
)


print(
    "Mean cycles per labeled ID       :",
    f"{len(classification_df) / max(len(study_training_df), 1):.2f}",
)


print(
    "Processing failures (ERROR)      :",
    n_errors,
)


print()


print(
    f"FINAL TRAIN AUC         = "
    f"{final_metrics['auc']:.4f}"
)


print(
    f"FINAL TRAIN ACCURACY    = "
    f"{final_metrics['accuracy']:.4f}"
)


print(
    f"FINAL TRAIN F1          = "
    f"{final_metrics['f1']:.4f}"
)


print(
    f"FINAL TRAIN SENSITIVITY = "
    f"{final_metrics['sensitivity']:.4f}"
)


print(
    f"FINAL TRAIN SPECIFICITY = "
    f"{final_metrics['specificity']:.4f}"
)


print(
    f"PREDICTED 0 / 1         = "
    f"{final_metrics['predicted_0']} / "
    f"{final_metrics['predicted_1']}"
)


print(
    "\nThese are IN-SAMPLE study-level metrics. "
    "They show whether the MIL model can fit the internal data, "
    "not whether it generalizes."
)


print(
    "\nOutputs"
)


print(
    "  multi-cycle manifest   :",
    CYCLE_MANIFEST_PATH,
)


print(
    "  training cycle instances:",
    OUTPUT_DIR
    / "training_cycle_instances.csv",
)


print(
    "  training IDs           :",
    OUTPUT_DIR
    / "training_ids.csv",
)


print(
    "  physiology normalization:",
    OUTPUT_DIR
    / "physiology_normalization.csv",
)


print(
    "  SSL history            :",
    OUTPUT_DIR
    / "ssl_history.csv",
)


print(
    "  SSL checkpoint         :",
    SSL_CHECKPOINT_PATH,
)


print(
    "  classification history :",
    OUTPUT_DIR
    / "classification_history.csv",
)


print(
    "  final model            :",
    FINAL_MODEL_PATH,
)


print(
    "  ID predictions         :",
    OUTPUT_DIR
    / "train_id_predictions.csv",
)


print(
    "  cycle attention        :",
    OUTPUT_DIR
    / "train_cycle_attention.csv",
)


print(
    "  spatial attention      :",
    OUTPUT_DIR
    / "train_spatial_attention.npy",
)


print(
    "  train ROC              :",
    OUTPUT_DIR
    / "train_roc.png",
)


print(
    "=" * 80
)

