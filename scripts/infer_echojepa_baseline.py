"""
External test inference for the cycle-preprocessed PURE EchoJEPA baseline.

Pipeline
--------
Raw echocardiographic NPY
    -> EchoNet-Dynamic LV segmentation for cycle localization only
    -> primary ED-peak detection on the LV area-time curve
    -> retain every adjacent ED-to-ED cycle; QC flags do not exclude cycles
    -> resample each cycle to 16 frames at 224 x 224
    -> EchoJEPA ViT-L
    -> equal mean pooling over all cycle features
    -> MLP classifier
    -> one SICM probability per ID

No fallback interval is generated when fewer than two primary ED peaks are
found. The LV-area representation is not passed to the baseline classifier.
No labels, random augmentation, adaptation, or fine-tuning are used.
"""

import os
import sys
from pathlib import Path

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
MODEL_PATH = required_env_path("SICM_BASELINE_FINAL_MODEL")
LV_SEGMENTATION_CHECKPOINT = Path(
    os.environ.get(
        "LV_SEGMENTATION_CHECKPOINT",
        str(CHECKPOINT_DIR / "deeplabv3_resnet50_random.pt"),
    )
).expanduser()
SEGMENTATION_NORMALIZATION_CSV = Path(
    os.environ.get(
        "SICM_BASELINE_SEGMENTATION_NORMALIZATION_CSV",
        str(MODEL_PATH.parent / "data_normalization.csv"),
    )
).expanduser()
OUTPUT_CSV = Path(
    os.environ.get(
        "SICM_EXTERNAL_BASELINE_PREDICTIONS_CSV",
        "outputs/external/echojepa_baseline_test_predictions.csv",
    )
).expanduser()
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SETTINGS
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

SEG_SIZE = 112
SEG_BATCH_SIZE = 64
SMOOTH_SIGMA = 1.2
MIN_ED_DISTANCE = 6
ED_PROMINENCE_FRACTION = 0.05
MIN_CYCLE_FRAMES = 6
MAX_CYCLE_FRAMES = 90
QC_MIN_AREA_EXCURSION_FRACTION = 0.06
QC_MAX_ED_AREA_MISMATCH_FRACTION = 0.60

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
# 2. STARTUP CHECKS
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
        MODEL_PATH,
        "trained EchoJEPA baseline final_model.pt",
    ),
    (
        LV_SEGMENTATION_CHECKPOINT,
        "EchoNet-Dynamic LV segmentation checkpoint",
    ),
    (
        SEGMENTATION_NORMALIZATION_CSV,
        "training-set LV-segmentation normalization",
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

_normalization = pd.read_csv(SEGMENTATION_NORMALIZATION_CSV)
if not {"mean", "std"}.issubset(_normalization.columns) or len(_normalization) != 3:
    raise RuntimeError(
        "Expected three RGB mean/std rows in the training normalization CSV."
    )
DATA_MEAN = torch.tensor(_normalization["mean"].to_numpy(), dtype=torch.float32)
DATA_STD = torch.tensor(_normalization["std"].to_numpy(), dtype=torch.float32)
if torch.any(DATA_STD <= 0):
    raise RuntimeError("Training-set segmentation standard deviations must be positive.")


# ============================================================
# 3. CARDIAC-CYCLE PREPROCESSING
# ============================================================

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

    vmin, vmax = float(x.min()), float(x.max())
    if vmin >= 0 and vmax <= 1.5:
        x = x * 255.0
    elif vmin < 0 or vmax > 255.0:
        x = (x - vmin) / (vmax - vmin) * 255.0 if vmax > vmin else torch.zeros_like(x)
    return x


def to_grayscale_255(x):
    if x.shape[1] == 3:
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    elif x.shape[1] != 1:
        raise ValueError(f"Channel count must be 1 or 3; got {x.shape[1]}")

    vmin, vmax = float(x.min()), float(x.max())
    if vmin >= 0 and vmax <= 1.5:
        x = x * 255.0
    elif vmin < 0 or vmax > 255.0:
        x = (x - vmin) / (vmax - vmin) * 255.0 if vmax > vmin else torch.zeros_like(x)
    return x.clamp(0, 255)


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
    for parameter in model.parameters():
        parameter.requires_grad = False
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


def robust_area_range(area):
    q05, q95 = np.quantile(area, [0.05, 0.95])
    return max(float(q95 - q05), 1.0)


def find_ed_peaks(smooth, area_range, distance, prominence_fraction):
    peaks, _ = find_peaks(
        smooth,
        distance=distance,
        prominence=prominence_fraction * area_range,
    )
    return peaks


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
    raw_area = np.asarray(raw_area, dtype=np.float32)
    if len(raw_area) < 2:
        raise RuntimeError("Video has fewer than 2 frames; no temporal interval can be formed.")

    smooth = gaussian_filter1d(raw_area, sigma=SMOOTH_SIGMA, mode="nearest")
    area_range = robust_area_range(smooth)
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
    for index in range(len(ed_candidates) - 1):
        candidate = build_cycle_candidate(
            smooth, area_range, ed_candidates[index], ed_candidates[index + 1]
        )
        if candidate is not None:
            cycles.append(candidate)
    if not cycles:
        raise RuntimeError("No valid adjacent ED-to-ED cycles were found.")

    return {
        "cycles": cycles,
        "smooth_area": smooth,
        "ed_candidates": ed_candidates,
        "detection_method": "primary_peaks",
    }


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


def prepare_echojepa_cycle(cycle_array):
    video = (
        torch.from_numpy(cycle_array.copy()).float().unsqueeze(1).repeat(1, 3, 1, 1)
    )
    if cycle_array.dtype == np.uint8 or float(video.max()) > 1.5:
        video = video / 255.0
    video = video.clamp(0.0, 1.0)
    mean = ECHOJEPA_MEAN.view(1, 3, 1, 1)
    std = ECHOJEPA_STD.view(1, 3, 1, 1)
    return (video - mean).div(std).permute(1, 0, 2, 3).contiguous()


def prepare_input(npy_path, segmenter):
    original_video, raw_area = get_lv_area_curve(npy_path, segmenter)
    detection = detect_all_cycles(raw_area)
    cycle_tensors = [
        prepare_echojepa_cycle(
            resample_cycle(original_video, cycle["ed1"], cycle["ed2"])
        )
        for cycle in detection["cycles"]
    ]
    if not cycle_tensors:
        raise RuntimeError("No cardiac cycles were detected.")
    return torch.stack(cycle_tensors, dim=0), detection


# ============================================================
# 4. BUILD EchoJEPA ViT-L ARCHITECTURE
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
            f"Unexpected EchoJEPA embed_dim: {encoder.embed_dim}"
        )

    if len(
        encoder.blocks
    ) != TRANSFORMER_DEPTH:

        raise RuntimeError(
            f"Unexpected EchoJEPA depth: {len(encoder.blocks)}"
        )

    if encoder.num_patches != NUM_TOKENS:

        raise RuntimeError(
            f"Unexpected EchoJEPA token count: {encoder.num_patches}"
        )

    return encoder


# ============================================================
# 5. PURE BASELINE MODEL
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

    def forward(self, video):
        # video: N_cycles,C,T,H,W (or C,T,H,W for a single cycle)
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise RuntimeError(f"Expected (N,C,T,H,W) cycles; got {video.shape}")

        cycle_features = []
        for cycle in video:
            tokens = self.backbone(cycle.unsqueeze(0))
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
            if tokens.ndim != 3:
                raise RuntimeError(f"Unexpected EchoJEPA output shape: {tokens.shape}")
            cycle_features.append(tokens.mean(dim=1).squeeze(0))

        feature = torch.stack(cycle_features, dim=0).mean(dim=0, keepdim=True)
        return self.classifier(feature).squeeze(-1)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
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


model = EchoJEPABaselineClassifier(
    build_echojepa_vitl()
)

load_result = model.load_state_dict(
    state_dict,
    strict=True,
)

model = model.to(
    DEVICE
)

model.eval()

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# 7. FIND TEST NPY FILES
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
        f"No .npy files found under:\n{TEST_ROOT}"
    )


print(
    "=" * 80
)

print(
    "PURE EchoJEPA BASELINE TEST INFERENCE"
)

print(
    "=" * 80
)

print(
    "Device    :",
    DEVICE,
)

print(
    "Test root :",
    TEST_ROOT,
)

print(
    "Model     :",
    MODEL_PATH,
)

print(
    "Videos    :",
    len(
        npy_paths
    ),
)

print(
    "=" * 80
)

lv_segmenter = build_lv_segmenter(LV_SEGMENTATION_CHECKPOINT)


# ============================================================
# 8. TEST INFERENCE
# ============================================================

records = []


for npy_path in tqdm(
    npy_paths,
    desc="EchoJEPA baseline test inference",
):

    sample_id = str(
        npy_path.parent.name
    )

    try:

        video, detection = prepare_input(
            npy_path,
            lv_segmenter,
        )
        video = video.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.no_grad():

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

                "n_cycles":
                    int(video.shape[0]),

                "relative_path":
                    str(
                        npy_path.relative_to(
                            TEST_ROOT
                        )
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

        del video

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    except Exception as exc:

        records.append(
            {
                "id":
                    sample_id,

                "npy_file":
                    npy_path.name,

                "n_cycles":
                    np.nan,

                "relative_path":
                    str(
                        npy_path.relative_to(
                            TEST_ROOT
                        )
                    ),

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
# 9. SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(
    records
)

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\n"
    + "=" * 80
)

print(
    "TEST INFERENCE COMPLETED"
)

print(
    "=" * 80
)

print(
    "Successful:",
    int(
        (
            result_df[
                "status"
            ]
            == "success"
        ).sum()
    ),
)

print(
    "Failed    :",
    int(
        (
            result_df[
                "status"
            ]
            == "failed"
        ).sum()
    ),
)

print(
    "Output    :",
    OUTPUT_CSV,
)

print(
    "=" * 80
)

