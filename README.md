# SICM-Echo

Research code for the physiology-informed echocardiography workflow accompanying the Critical Care manuscript on early identification of sepsis-induced cardiomyopathy (SICM).

This repository contains the supplied training, patient-level cross-validation, external inference, and external evaluation scripts. The scripts expect a cohort table and de-identified video arrays prepared outside this repository. They do not retrieve clinical records, select the index echocardiogram, or derive the clinical labels.

## Repository map

~~~text
.
|-- scripts/
|   |-- train.py
|   |-- cross_validate_patient_5fold.py
|   |-- infer_external.py
|   `-- evaluate_external_test_auc.py
|-- docs/
|   |-- data_format.md
|   |-- methodology_audit.md
|   `-- third_party.md
|-- requirements.txt
|-- .gitignore
`-- LICENSE
~~~

The four scripts were reorganized from the supplied source files. Only local path configuration and output destinations were made environment-configurable. Model architecture, preprocessing, hyperparameters, cycle-detection logic, and evaluation logic were retained. See the methodology audit before interpreting a run: it records differences between the manuscript and the supplied implementations that need author review.

## Workflow

1. **Train the final development-cohort model** with scripts/train.py. It uses every valid labeled development ID, performs LV segmentation and cardiac-cycle extraction, runs self-supervised adaptation, then supervised two-stage patient-level MIL training. Its reported training metrics are in-sample.
2. **Run internal validation** with scripts/cross_validate_patient_5fold.py. It assigns unique patient IDs to stratified folds and writes pooled out-of-fold (OOF) predictions and fold-level artifacts. The current implementation estimates LV-segmentation input normalization before assigning folds; this is documented for review.
3. **Run external inference** with scripts/infer_external.py. It reads the trained final model and training-derived normalization, does not read labels, and writes one probability per input ID.
4. **Evaluate external predictions** with scripts/evaluate_external_test_auc.py. It joins a prepared binary label file to full-model and EchoJEPA baseline prediction files. One label definition is evaluated per invocation; prepare separate label files for the primary and sensitivity definitions.

Clinical cohort construction, Sepsis-3 identification, cardiac exclusion rules, time-window selection, and SICM reference-label derivation are outside the supplied scripts. Prepare those inputs under the applicable data-access and governance rules. Do not place data or labels in this repository.

## Data layout and privacy

See docs/data_format.md for CSV columns, NumPy video shapes, ID matching rules, and the distinct path convention used by training versus external inference.

**Never commit patient-level input, label tables, raw echocardiograms, derived cycles, prediction tables, model checkpoints, or local run outputs.** .gitignore excludes the standard data, checkpoint, and output locations and common medical-video/model formats. The ignore rules are a safeguard, not a substitute for checking git status before committing.

## Dependencies and model assets

Install the direct Python requirements from requirements.txt, then install the EchoJEPA source dependency described in docs/third_party.md. Install a compatible PyTorch/torchvision pair for the available CUDA runtime before running the scripts.

The scripts expect the EchoJEPA source tree and two checkpoint files to be supplied locally:

- vitl-vmix22m-pt220-c55.pt: EchoJEPA ViT-L initialization checkpoint.
- deeplabv3_resnet50_random.pt: EchoNet-Dynamic DeepLabV3-ResNet50 LV-segmentation checkpoint.

They are not included in this repository. The source files supplied for this release did not include an environment lockfile, upstream EchoJEPA commit ID, or checkpoint binaries, so the exact original runtime and binary identities could not be independently recorded. Record those versions and checksums alongside any reproduction run.

## Setup

Use Python 3.12 as a practical starting point for the current EchoJEPA upstream setup. The exact Python, PyTorch, torchvision, CUDA, and package versions used for the study were not present in the supplied files; see the audit document.

~~~bash
python -m venv .venv
source .venv/bin/activate

# Install a torch/torchvision pair matching your CUDA runtime first.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

git clone https://github.com/bowang-lab/EchoJEPA.git checkpoints/EchoJEPA
python -m pip install -e checkpoints/EchoJEPA
git -C checkpoints/EchoJEPA rev-parse HEAD
~~~

Obtain the required pretrained checkpoints through their respective upstream sources and place them under checkpoints/, or set explicit environment variables. Do not commit them.

For PowerShell, activate the environment with .venv\Scripts\Activate.ps1; set variables as $env:NAME = "value".

## Full-model training

The label CSV must contain exactly two columns named id and label; every ID must have one binary label (0 = sepsis-only control, 1 = SICM) and one matching <id>.npy video file under the video root. See docs/data_format.md.

~~~bash
export SICM_VIDEO_ROOT="/secure/path/development/videos"
export SICM_LABEL_CSV="/secure/path/development/id_label.csv"
export SICM_CHECKPOINT_DIR="checkpoints"
export ECHOJEPA_REPO_DIR="checkpoints/EchoJEPA"
export ECHOJEPA_CHECKPOINT="checkpoints/vitl-vmix22m-pt220-c55.pt"
export LV_SEGMENTATION_CHECKPOINT="checkpoints/deeplabv3_resnet50_random.pt"
export SICM_OUTPUT_DIR="outputs/full_model"

python scripts/train.py
~~~

The source settings use 10 self-supervised epochs, 10 frozen-backbone Stage A epochs, and 20 Stage B fine-tuning epochs. The code uses all detected primary-peak cycles by default and records, but does not use, its cycle QC flags to exclude cycles. It also uses a positive-class weight of n_negative / n_positive in the supervised loss.

The final model is written to outputs/full_model/final_model.pt; the segmentation normalization used for inference is written beside it as data_normalization.csv. The script also writes local manifests, histories, and patient-level predictions. These are run outputs and must remain outside version control.

## Patient-level fivefold cross-validation

Use the same development video and label inputs and checkpoint configuration as for training:

~~~bash
export SICM_VIDEO_ROOT="/secure/path/development/videos"
export SICM_LABEL_CSV="/secure/path/development/id_label.csv"
export SICM_CHECKPOINT_DIR="checkpoints"
export ECHOJEPA_REPO_DIR="checkpoints/EchoJEPA"
export ECHOJEPA_CHECKPOINT="checkpoints/vitl-vmix22m-pt220-c55.pt"
export LV_SEGMENTATION_CHECKPOINT="checkpoints/deeplabv3_resnet50_random.pt"
export SICM_CV_OUTPUT_DIR="outputs/patient_level_5fold"

python scripts/cross_validate_patient_5fold.py
~~~

Each fold starts from the supplied EchoJEPA checkpoint and trains on four folds. All cycles belonging to an ID stay in that ID's fold. The script writes fold assignments, per-fold checkpoints, predictions, metrics, and pooled OOF metrics. Do not treat training-fold metrics as validation results. The full-cohort segmentation-normalization step before fold assignment is disclosed in the audit document.

## External inference

External video arrays use the directory convention described in docs/data_format.md. This script does not access labels or compute evaluation metrics.

~~~bash
export SICM_EXTERNAL_VIDEO_ROOT="/secure/path/external/videos"
export SICM_CHECKPOINT_DIR="checkpoints"
export ECHOJEPA_REPO_DIR="checkpoints/EchoJEPA"
export LV_SEGMENTATION_CHECKPOINT="checkpoints/deeplabv3_resnet50_random.pt"
export SICM_FINAL_MODEL="outputs/full_model/final_model.pt"
export SICM_SEGMENTATION_NORMALIZATION_CSV="outputs/full_model/data_normalization.csv"
export SICM_EXTERNAL_OUTPUT_CSV="outputs/external/physiology_informed_test_predictions.csv"

python scripts/infer_external.py
~~~

The prediction CSV includes the pseudonymous ID, input filename/path relative to the configured external-video root, number of cycles, SICM probability, a 0.5-threshold prediction, and a success/error status. It is patient-level output and must not be committed.

## External evaluation

Supply the prepared binary reference-label CSV and both prediction files:

~~~bash
export SICM_EXTERNAL_LABELS_CSV="/secure/path/external/id_label_primary.csv"
export SICM_EXTERNAL_FULL_PREDICTIONS_CSV="outputs/external/physiology_informed_test_predictions.csv"
export SICM_EXTERNAL_BASELINE_PREDICTIONS_CSV="outputs/external/echojepa_baseline_test_predictions.csv"
export SICM_EXTERNAL_EVALUATION_DIR="outputs/external/evaluation"

python scripts/evaluate_external_test_auc.py
~~~

The current evaluator calculates AUROC with a percentile bootstrap 95% confidence interval (2,000 resamples, seed 42), plus accuracy, sensitivity, specificity, and F1 at probability threshold 0.5. It writes merged ID/label/prediction tables and a ROC comparison. Run it separately with each prespecified sensitivity-analysis label file, using the same model probabilities. It does not derive labels from cTnT values.

The supplied source set did not include the script that generates the EchoJEPA baseline prediction CSV required above. Until that source is added, baseline inference is an external prerequisite. The manuscript analyses not implemented by these scripts are listed in the audit.

## Research use

This software is provided for research and reproducibility. It is not a medical device and must not be used to diagnose or guide care. Apply the terms governing each dataset, checkpoint, and upstream package independently.

