
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def required_env_path(name):

    value = os.environ.get(name)

    if not value:

        raise RuntimeError(
            f"Set the {name} environment variable before running this script."
        )

    return Path(value).expanduser()


TEST_ROOT = required_env_path(
    "SICM_EXTERNAL_VIDEO_ROOT"
)

CHECKPOINT_DIR = Path(
    os.environ.get(
        "SICM_CHECKPOINT_DIR",
        "checkpoints",
    )
).expanduser()

ECHOJEPA_REPO_DIR = Path(
    os.environ.get(
        "ECHOJEPA_REPO_DIR",
        CHECKPOINT_DIR / "EchoJEPA",
    )
).expanduser()

MODEL_PATH = Path(
    os.environ.get(
        "SICM_BASELINE_FINAL_MODEL",
        "outputs/echojepa_baseline/final_model.pt",
    )
).expanduser()

OUTPUT_CSV = Path(
    os.environ.get(
        "SICM_EXTERNAL_BASELINE_PREDICTIONS_CSV",
        "outputs/external/echojepa_baseline_test_predictions.csv",
    )
).expanduser()

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
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


def to_grayscale_01(
    x,
):

    if x.shape[
        1
    ] == 3:

        x = (
            0.2989
            * x[
                :,
                0:1
            ]
            +
            0.5870
            * x[
                :,
                1:2
            ]
            +
            0.1140
            * x[
                :,
                2:3
            ]
        )

    elif x.shape[
        1
    ] != 1:

        raise ValueError(
            f"Expected 1 or 3 channels; got {x.shape[1]}"
        )

    vmin = float(
        x.min()
    )

    vmax = float(
        x.max()
    )

    if (
        vmin >= 0
        and vmax <= 1.5
    ):

        x = x.clamp(
            0.0,
            1.0,
        )

    elif (
        vmin >= 0
        and vmax <= 255.0
    ):

        x = (
            x
            / 255.0
        ).clamp(
            0.0,
            1.0,
        )

    else:

        if vmax > vmin:

            x = (
                x
                - vmin
            ) / (
                vmax
                - vmin
            )

        else:

            x = torch.zeros_like(
                x
            )

    return x.float()


def uniform_sample_video(
    video,
):

    total_frames = int(
        video.shape[
            0
        ]
    )

    if total_frames < 1:

        raise RuntimeError(
            "Video contains no frames."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
    )

    indices = np.round(
        indices
    ).astype(
        np.int64
    )

    indices = np.clip(
        indices,
        0,
        total_frames - 1,
    )

    indices = torch.from_numpy(
        indices
    ).long()

    return video[
        indices
    ]


def prepare_input(
    npy_path,
):

    video = load_video_tchw(
        npy_path
    )

    video = to_grayscale_01(
        video
    )

    video = uniform_sample_video(
        video
    )

    video = F.interpolate(
        video,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )


    video = video.repeat(
        1,
        3,
        1,
        1,
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


    return (
        video
        .permute(
            1,
            0,
            2,
            3,
        )
        .contiguous()
        .unsqueeze(
            0
        )
    )


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

    def forward(
        self,
        video,
    ):

        tokens = self.backbone(
            video
        )

        if isinstance(
            tokens,
            (
                tuple,
                list,
            ),
        ):

            tokens = tokens[
                0
            ]

        if tokens.ndim != 3:

            raise RuntimeError(
                f"Unexpected EchoJEPA output shape: {tokens.shape}"
            )

        feature = tokens.mean(
            dim=1
        )

        return (
            self.classifier(
                feature
            )
            .squeeze(
                -1
            )
        )


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

if isinstance(checkpoint, dict):

    expected_metadata = {
        "architecture":
            "EchoJEPA ViT-L + global mean pooling + MLP classifier",

        "base_model":
            "vitl-vmix22m-pt220-c55",

        "num_frames":
            NUM_FRAMES,

        "image_size":
            IMAGE_SIZE,

        "sampling":
            "uniform sampling from complete cine loop",

        "pooling":
            "global mean pooling over all EchoJEPA spatiotemporal tokens",
    }

    for key, expected_value in expected_metadata.items():

        if (
            key in checkpoint
            and checkpoint[key] != expected_value
        ):

            raise RuntimeError(
                "Baseline checkpoint metadata does not match the locked "
                f"training pipeline: {key}={checkpoint[key]!r}, "
                f"expected {expected_value!r}."
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

model.load_state_dict(
    state_dict,
    strict=True,
)

model = model.to(
    DEVICE
)

model.eval()

for parameter in model.parameters():

    parameter.requires_grad = False


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


paths_by_id = {}

for path in npy_paths:

    sample_id = str(
        path.parent.name
    ).strip()

    paths_by_id.setdefault(
        sample_id,
        [],
    ).append(path)

duplicate_ids = {
    sample_id: paths
    for sample_id, paths in paths_by_id.items()
    if len(paths) != 1
}

if duplicate_ids:

    details = {
        sample_id: [
            str(path.relative_to(TEST_ROOT))
            for path in paths
        ]
        for sample_id, paths in duplicate_ids.items()
    }

    raise RuntimeError(
        "Each external pseudonymous ID must have exactly one selected A4C NPY. "
        f"Duplicate IDs/files: {details}"
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


records = []


for npy_path in tqdm(
    npy_paths,
    desc="EchoJEPA baseline test inference",
):

    sample_id = str(
        npy_path.parent.name
    )

    try:

        video = prepare_input(
            npy_path
        ).to(
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