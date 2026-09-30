"""
External test inference for the trained full physiology-informed SICM model.

Pipeline
--------
Raw echocardiographic NPY
    -> EchoNet-Dynamic DeepLabV3-ResNet50 LV segmentation
    -> LV area-time curve
    -> ED / ES detection
    -> all adjacent ED-to-ED cardiac cycles
    -> 16-frame, 224 x 224 resampling
    -> EchoJEPA ViT-L
    -> phase-aware spatial attention
    -> phase-aware temporal attention
    -> LV-area physiological representation
    -> multi-cycle MIL
    -> one SICM probability per ID

Important
---------
- No label file is read.
- No AUC/accuracy is calculated.
- No random augmentation is applied.
- No self-supervised adaptation or fine-tuning is performed.
- LV-segmentation normalization is loaded from the TRAINING outcome folder.
- Physiology normalization is stored inside the trained final_model.pt.
"""


import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


# ============================================================
# 0. PATHS
# ============================================================

def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return Path(value).expanduser()


TEST_ROOT = required_env_path("SICM_EXTERNAL_VIDEO_ROOT")
CHECKPOINT_DIR = Path(os.environ.get("SICM_CHECKPOINT_DIR", "checkpoints")).expanduser()
ECHOJEPA_REPO_DIR = Path(
    os.environ.get("ECHOJEPA_REPO_DIR", str(CHECKPOINT_DIR / "EchoJEPA"))
).expanduser()
LV_SEGMENTATION_CHECKPOINT = Path(
    os.environ.get(
        "LV_SEGMENTATION_CHECKPOINT",
        str(CHECKPOINT_DIR / "deeplabv3_resnet50_random.pt"),
    )
).expanduser()
MODEL_PATH = required_env_path("SICM_FINAL_MODEL")
SEGMENTATION_NORMALIZATION_CSV = Path(
    os.environ.get(
        "SICM_SEGMENTATION_NORMALIZATION_CSV",
        str(MODEL_PATH.parent / "data_normalization.csv"),
    )
).expanduser()
OUTPUT_CSV = Path(
    os.environ.get(
        "SICM_EXTERNAL_OUTPUT_CSV",
        "outputs/external/physiology_informed_test_predictions.csv",
    )
).expanduser()
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. GENERAL SETTINGS
# ============================================================

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


# ============================================================
# 2. LV SEGMENTATION / CYCLE SETTINGS
#
# These values match the training pipeline.
# ============================================================

SEG_SIZE = 112
SEG_BATCH_SIZE = 64

SMOOTH_SIGMA = 1.2

MIN_ED_DISTANCE = 6
ED_PROMINENCE_FRACTION = 0.05

MIN_CYCLE_FRAMES = 6
MAX_CYCLE_FRAMES = 90

QC_MIN_AREA_EXCURSION_FRACTION = 0.06
QC_MAX_ED_AREA_MISMATCH_FRACTION = 0.60


# ============================================================
# 3. EchoJEPA / MIL SETTINGS
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

TEMPORAL_HEADS = 8
TEMPORAL_DROPOUT = 0.10

AREA_CURVE_POINTS = 64
AREA_EMBED_DIM = 128
CYCLE_FUSION_DIM = 512
MIL_ATTENTION_DIM = 128

CYCLE_FORWARD_CHUNK_SIZE = 1

USE_LVEF_AUXILIARY = False

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


# ============================================================
# 4. STARTUP CHECKS
# ============================================================

for path, description in [
    (
        TEST_ROOT,
        "test NPY folder",
    ),
    (
        ECHOJEPA_REPO_DIR,
        "EchoJEPA repository",
    ),
    (
        LV_SEGMENTATION_CHECKPOINT,
        "LV segmentation checkpoint",
    ),
    (
        MODEL_PATH,
        "trained final_model.pt",
    ),
    (
        SEGMENTATION_NORMALIZATION_CSV,
        "training data_normalization.csv",
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
    "SICM EXTERNAL TEST INFERENCE"
)

print(
    "=" * 80
)

print(
    "Device          :",
    DEVICE,
)

print(
    "Test root :",
    TEST_ROOT,
)

print(
    "Model           :",
    MODEL_PATH,
)

print(
    "Output          :",
    OUTPUT_CSV,
)

print(
    "=" * 80
)


# ============================================================
# 5. LOAD TRAINING-SET NORMALIZATION FOR LV SEGMENTATION
# ============================================================

normalization_df = pd.read_csv(
    SEGMENTATION_NORMALIZATION_CSV
)

if not {
    "mean",
    "std",
}.issubset(
    normalization_df.columns
):

    raise RuntimeError(
        "data_normalization.csv must contain columns: mean, std"
    )


DATA_MEAN = torch.tensor(
    normalization_df[
        "mean"
    ].to_numpy(
        dtype=np.float32
    ),
    dtype=torch.float32,
)


DATA_STD = torch.tensor(
    normalization_df[
        "std"
    ].to_numpy(
        dtype=np.float32
    ),
    dtype=torch.float32,
)


if DATA_MEAN.numel() != 3 or DATA_STD.numel() != 3:

    raise RuntimeError(
        "Expected three-channel LV segmentation normalization."
    )


print(
    "\nTraining LV-segmentation mean:",
    DATA_MEAN.tolist(),
)

print(
    "Training LV-segmentation std :",
    DATA_STD.tolist(),
)


# ============================================================
# 6. PREPROCESSING FUNCTIONS
# ============================================================


def load_video_tchw(npy_path):
    array = np.asarray(np.load(npy_path, mmap_mode="r"))

    if array.ndim == 3:
        # T, H, W
        x = torch.from_numpy(array.copy()).float().unsqueeze(1)
    elif array.ndim == 4 and array.shape[-1] in (1, 3):
        # T, H, W, C
        x = torch.from_numpy(array.copy()).float().permute(0, 3, 1, 2)
    elif array.ndim == 4 and array.shape[1] in (1, 3):
        # T, C, H, W
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

    """
    Return all adjacent ED-to-ED cycles found by primary ED-peak detection.

    If fewer than two ED peaks are detected, raise an error. No alternate
    detector or synthetic fallback interval is used.
    """

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


# ============================================================
# 17. RESAMPLE VIDEO CYCLE + LV-AREA PHYSIOLOGY
# ============================================================

def resample_cycle(
    original_video,
    ed1,
    ed2,
):

    """
    Crop ED1..ED2 and resample to:

        NUM_FRAMES x IMAGE_SIZE x IMAGE_SIZE

    Output:
        uint8 T,H,W
    """

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

    """
    Create a fixed-length normalized LV-area curve plus explicit
    physiological descriptors.

    The curve is divided by the mean ED area, so ED is approximately 1.
    This emphasizes relative contraction / filling rather than image zoom.

    Explicit descriptors:
        0. fractional area change
        1. normalized systolic contraction slope
        2. normalized diastolic filling slope
        3. cycle-length proxy
        4. relative ED endpoint mismatch

    The cycle-length value is frame-based because the current NPY files do not
    provide a per-study frame rate. If true FPS becomes available, replace this
    proxy with duration in seconds.
    """

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


def tokens_to_spatial_grid(
    tokens,
):

    """
    Convert the EchoJEPA token sequence:

        B, 1568, 1024

    to:

        B, 8, 196, 1024

    where:

        8   = temporal tubelet positions
        196 = 14 x 14 spatial tokens

    No spatial averaging is performed here.
    """

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


# ============================================================
# 7. EchoJEPA INPUT PREPARATION
# ============================================================

def prepare_echojepa_cycle(
    cycle_array,
):

    """
    cycle_array:
        uint8, shape T,H,W

    return:
        float32, shape C,T,H,W
    """

    cycle = (
        torch.from_numpy(
            cycle_array.copy()
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
        / 255.0
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


# ============================================================
# 8. BUILD EMPTY EchoJEPA ViT-L ARCHITECTURE
#
# final_model.pt already contains the trained backbone weights.
# Therefore the official pretrained checkpoint does NOT need to be
# loaded again here; the local EchoJEPA source code is only used to
# construct the architecture.
# ============================================================

def build_echojepa_vitl():

    from src.models import (
        vision_transformer
        as vit_encoder
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
        use_activation_checkpointing=False,
        handle_nonsquare_inputs=True,
    )

    if encoder.embed_dim != FEATURE_DIM:

        raise RuntimeError(
            f"Unexpected EchoJEPA embed_dim: "
            f"{encoder.embed_dim}"
        )

    if len(
        encoder.blocks
    ) != TRANSFORMER_DEPTH:

        raise RuntimeError(
            f"Unexpected EchoJEPA depth: "
            f"{len(encoder.blocks)}"
        )

    if encoder.num_patches != NUM_TOKENS:

        raise RuntimeError(
            f"Unexpected EchoJEPA token count: "
            f"{encoder.num_patches}"
        )

    return encoder


# ============================================================
# 9. SPATIAL ATTENTION
#
# phase_aware=False:
#     matches the earlier model that produced the ~0.981 train AUC.
#
# phase_aware=True:
#     matches the newer model with explicit cardiac-phase embeddings.
# ============================================================

class SpatialAttentionPool(
    nn.Module
):

    def __init__(
        self,
        phase_aware,
    ):

        super().__init__()

        self.phase_aware = (
            phase_aware
        )

        self.input_norm = nn.LayerNorm(
            FEATURE_DIM
        )

        if self.phase_aware:

            self.temporal_pos = nn.Parameter(
                torch.zeros(
                    1,
                    TEMPORAL_TOKENS,
                    1,
                    FEATURE_DIM,
                )
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

        if self.phase_aware:

            normalized = (
                normalized
                + self.temporal_pos.to(
                    dtype=normalized.dtype,
                    device=normalized.device,
                )
            )

        scores = self.score(
            normalized
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


# ============================================================
# 10. TEMPORAL ATTENTION
# ============================================================

class TemporalAttentionPool(
    nn.Module
):

    def __init__(
        self,
        phase_aware,
    ):

        super().__init__()

        self.phase_aware = (
            phase_aware
        )

        self.input_norm = nn.LayerNorm(
            FEATURE_DIM
        )

        if self.phase_aware:

            self.temporal_pos = nn.Parameter(
                torch.zeros(
                    1,
                    TEMPORAL_TOKENS,
                    FEATURE_DIM,
                )
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

        if self.phase_aware:

            x = (
                x
                + self.temporal_pos.to(
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


# ============================================================
# 11. LV-AREA ENCODERS
# ============================================================

class LVAreaEncoderLegacy(
    nn.Module
):

    """
    Matches the earlier trained Spatial-Attention model.
    """

    def __init__(
        self,
    ):

        super().__init__()

        self.curve_encoder = nn.Sequential(
            nn.LayerNorm(
                AREA_CURVE_POINTS
            ),
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
            nn.LayerNorm(
                5
            ),
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

        curve_feature = self.curve_encoder(
            area_curves
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


class LVAreaEncoderDatasetNorm(
    nn.Module
):

    """
    Matches the newer phase-aware model.
    """

    def __init__(
        self,
    ):

        super().__init__()

        # Temporary initialization.
        # The trained values are loaded from model_state_dict.
        self.register_buffer(
            "phys_mean",
            torch.zeros(
                5,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "phys_std",
            torch.ones(
                5,
                dtype=torch.float32,
            )
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

        curve_feature = self.curve_encoder(
            area_curves
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


# ============================================================
# 12. MIL ATTENTION
# ============================================================

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


# ============================================================
# 13. INFERENCE MODEL
# ============================================================

class SICMInferenceModel(
    nn.Module
):

    def __init__(
        self,
        backbone,
        phase_aware,
        dataset_phys_norm,
        has_lvef_head,
    ):

        super().__init__()

        self.backbone = (
            backbone
        )

        self.spatial_pool = (
            SpatialAttentionPool(
                phase_aware=(
                    phase_aware
                )
            )
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                phase_aware=(
                    phase_aware
                )
            )
        )

        if dataset_phys_norm:

            self.area_encoder = (
                LVAreaEncoderDatasetNorm()
            )

        else:

            self.area_encoder = (
                LVAreaEncoderLegacy()
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

        if has_lvef_head:

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

    @torch.no_grad()
    def encode_visual_cycles(
        self,
        cycles,
    ):

        visual_features = []

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
                _
            ) = self.spatial_pool(
                spatial_grid
            )

            (
                pooled_visual,
                _
            ) = self.temporal_pool(
                temporal_features
            )

            visual_features.append(
                pooled_visual
            )

        return torch.cat(
            visual_features,
            dim=0,
        )

    @torch.no_grad()
    def forward(
        self,
        cycles,
        area_curves,
        physiology,
    ):

        visual_feature = (
            self.encode_visual_cycles(
                cycles
            )
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
        ) = self.mil_attention(
            fused_cycle_feature
        )

        study_logit = (
            self.classifier(
                study_feature
            )
            .squeeze(
                -1
            )
        )

        return (
            study_logit,
            cycle_attention,
        )


# ============================================================
# 14. LOAD TRAINED FINAL MODEL
# ============================================================

def load_trained_model(
    model_path,
):

    try:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    phase_aware = (
        "spatial_pool.temporal_pos"
        in state_dict
        or
        "temporal_pool.temporal_pos"
        in state_dict
    )

    dataset_phys_norm = (
        "area_encoder.phys_mean"
        in state_dict
    )

    has_lvef_head = any(
        key.startswith(
            "lvef_head."
        )
        for key in state_dict.keys()
    )

    legacy_cycle_classifier = any(
        key.startswith(
            "cycle_classifier."
        )
        for key in state_dict.keys()
    )

    print(
        "\nDetected checkpoint architecture:"
    )

    print(
        "  phase-aware pooling       :",
        phase_aware,
    )

    print(
        "  dataset physiology z-score:",
        dataset_phys_norm,
    )

    print(
        "  legacy cycle classifier   :",
        legacy_cycle_classifier,
    )

    backbone = build_echojepa_vitl()

    model = SICMInferenceModel(
        backbone=backbone,
        phase_aware=phase_aware,
        dataset_phys_norm=(
            dataset_phys_norm
        ),
        has_lvef_head=(
            has_lvef_head
        ),
    )

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    allowed_unexpected = [
        key
        for key in result.unexpected_keys
        if key.startswith(
            "cycle_classifier."
        )
    ]

    other_unexpected = [
        key
        for key in result.unexpected_keys
        if key not in allowed_unexpected
    ]

    if result.missing_keys:

        raise RuntimeError(
            "Missing model keys:\n"
            + "\n".join(
                result.missing_keys[
                    :30
                ]
            )
        )

    if other_unexpected:

        raise RuntimeError(
            "Unexpected model keys:\n"
            + "\n".join(
                other_unexpected[
                    :30
                ]
            )
        )

    if allowed_unexpected:

        print(
            "Ignoring legacy cycle-classifier weights "
            "because test output is study-level only."
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    return model


# ============================================================
# 15. PREPARE ONE NPY STUDY IN MEMORY
# ============================================================

def prepare_study(
    npy_path,
    segmenter,
):

    (
        original_video,
        raw_area
    ) = get_lv_area_curve(
        npy_path,
        segmenter,
    )

    detection = detect_all_cycles(
        raw_area
    )

    cycle_tensors = []

    area_curves = []

    physiology_values = []

    for cycle_info in detection[
        "cycles"
    ]:

        cycle_array = resample_cycle(
            original_video,
            cycle_info[
                "ed1"
            ],
            cycle_info[
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
            cycle_info[
                "ed1"
            ],
            cycle_info[
                "es"
            ],
            cycle_info[
                "ed2"
            ],
        )

        cycle_tensors.append(
            prepare_echojepa_cycle(
                cycle_array
            )
        )

        area_curves.append(
            torch.from_numpy(
                area_curve.copy()
            ).float()
        )

        physiology_values.append(
            torch.from_numpy(
                physiology.copy()
            ).float()
        )

    if len(
        cycle_tensors
    ) == 0:

        raise RuntimeError(
            "No cardiac cycles were generated."
        )

    cycles = torch.stack(
        cycle_tensors,
        dim=0,
    )

    area_curves = torch.stack(
        area_curves,
        dim=0,
    )

    physiology = torch.stack(
        physiology_values,
        dim=0,
    )

    return (
        cycles,
        area_curves,
        physiology,
        detection,
    )


# ============================================================
# 16. FIND TEST NPY FILES
#
# The parent folder name of each NPY is used as the output ID.
# Nested folders are supported.
# ============================================================

npy_paths = sorted(
    [
        path
        for path in TEST_ROOT.rglob("*.npy")
        if "outcome" not in {
            part.lower()
            for part in path.parts
        }
        and "cycles" not in {
            part.lower()
            for part in path.parts
        }
    ]
)


if len(
    npy_paths
) == 0:

    raise RuntimeError(
        f"No .npy files found under:\n"
        f"{TEST_ROOT}"
    )


print(
    f"\nFound {len(npy_paths)} test NPY videos."
)


# ============================================================
# 17. LOAD MODELS
# ============================================================

lv_segmenter = build_lv_segmenter(
    LV_SEGMENTATION_CHECKPOINT
)


model = load_trained_model(
    MODEL_PATH
)


# ============================================================
# 18. TEST INFERENCE
# ============================================================

records = []


for npy_path in tqdm(
    npy_paths,
    desc="Full-model test inference",
):

    sample_id = str(
        npy_path.parent.name
    )

    try:

        (
            cycles,
            area_curves,
            physiology,
            detection
        ) = prepare_study(
            npy_path,
            lv_segmenter,
        )

        cycles = cycles.to(
            DEVICE,
            non_blocking=True,
        )

        area_curves = area_curves.to(
            DEVICE,
            non_blocking=True,
        )

        physiology = physiology.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.no_grad():

            with torch.amp.autocast(
                device_type=DEVICE_TYPE,
                enabled=AMP_ENABLED,
            ):

                (
                    study_logit,
                    cycle_attention
                ) = model(
                    cycles,
                    area_curves,
                    physiology,
                )

        probability = float(
            torch.sigmoid(
                study_logit.float()
            )
            .cpu()
            .item()
        )

        prediction = int(
            probability
            >= 0.5
        )

        records.append(
            {
                "id":
                    sample_id,

                "npy_file":
                    npy_path.name,

                "relative_path":
                    str(
                        npy_path.relative_to(
                            TEST_ROOT
                        )
                    ),

                "n_cycles":
                    int(
                        cycles.shape[
                            0
                        ]
                    ),

                "sicm_probability":
                    probability,

                "prediction":
                    prediction,

                "prediction_name":
                    (
                        "SICM"
                        if prediction == 1
                        else "sepsis-only"
                    ),

                "status":
                    "success",

                "error":
                    "",
            }
        )

        del cycles
        del area_curves
        del physiology
        del cycle_attention

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    except Exception as exc:

        records.append(
            {
                "id":
                    sample_id,

                "npy_file":
                    npy_path.name,

                "relative_path":
                    str(
                        npy_path.relative_to(
                            TEST_ROOT
                        )
                    ),

                "n_cycles":
                    np.nan,

                "sicm_probability":
                    np.nan,

                "prediction":
                    np.nan,

                "prediction_name":
                    "",

                "status":
                    "failed",

                "error":
                    repr(
                        exc
                    ),
            }
        )


# ============================================================
# 19. SAVE TEST MODEL RESULTS ONLY
# ============================================================

result_df = pd.DataFrame(
    records
)


result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


success_count = int(
    (
        result_df[
            "status"
        ]
        == "success"
    ).sum()
)


failed_count = int(
    (
        result_df[
            "status"
        ]
        == "failed"
    ).sum()
)


print(
    "\n"
    + "=" * 80
)

print(
    "VALIDATION INFERENCE COMPLETED"
)

print(
    "=" * 80
)

print(
    "Total   :",
    len(
        result_df
    ),
)

print(
    "Success :",
    success_count,
)

print(
    "Failed  :",
    failed_count,
)

print(
    "\nPredictions saved to:"
)

print(
    OUTPUT_CSV
)

print(
    "\nNo test labels were loaded or compared."
)

print(
    "=" * 80
)

