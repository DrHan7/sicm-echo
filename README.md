# SICM-Echo

Research code for the physiology-informed echocardiography workflow accompanying the Critical Care manuscript on early identification of sepsis-induced cardiomyopathy (SICM).

This repository contains the supplied training, patient-level cross-validation, external inference, and external evaluation scripts. The scripts expect a cohort table and de-identified video arrays prepared outside this repository. They do not retrieve clinical records, select the index echocardiogram, or derive the clinical labels.

## Repository map

~~~text
.
|-- scripts/
|   |-- train.py
|   |-- train_echojepa_baseline.py
|   |-- cross_validate_patient_5fold.py
|   |-- infer_external.py
|   |-- infer_echojepa_baseline.py
|   `-- evaluate_external_test_auc.py
|-- docs/
|   |-- data_format.md
|   `-- third_party.md
|-- requirements.txt
|-- .gitignore
`-- LICENSE
~~~

The six scripts were reorganized from the supplied source files. Local paths and output destinations were made environment-configurable. The patient-level CV source fits segmentation normalization within each training fold, and external inference uses the same primary ED-peak detector as development.

## Workflow

1. **Train the physiology-informed model** with scripts/train.py. It performs self-supervised adaptation followed by supervised two-stage patient-level MIL training. Its training metrics are in-sample.
2. **Train the direct EchoJEPA baseline** with scripts/train_echojepa_baseline.py. It uses the same cycle extraction, then averages EchoJEPA cycle features before an MLP classifier. Its training metrics are also in-sample.
3. **Run internal validation** with scripts/cross_validate_patient_5fold.py. It assigns IDs to stratified patient folds, fits segmentation normalization on each training fold, and writes OOF predictions plus preprocessing-failure records.
4. **Run both external inference scripts** with scripts/infer_external.py and scripts/infer_echojepa_baseline.py. Each loads its locked model and training-derived segmentation normalization, reads no labels, and writes probabilities by pseudonymous ID.
5. **Evaluate external predictions** with scripts/evaluate_external_test_auc.py. It joins prepared binary labels to the full-model and baseline predictions. One label definition is evaluated per invocation; prepare separate label files for the primary and sensitivity definitions.

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

Use Python 3.12 as a practical starting point for the current EchoJEPA upstream setup. The exact Python, PyTorch, torchvision, CUDA, and package versions used for the study were not present in the supplied files; record the environment used for each reproduction run.

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

## Direct EchoJEPA baseline training

Use the same development videos, binary labels, EchoJEPA initialization, and LV-segmentation checkpoint as the full model:

~~~bash
export SICM_VIDEO_ROOT="/secure/path/development/videos"
export SICM_LABEL_CSV="/secure/path/development/id_label.csv"
export SICM_CHECKPOINT_DIR="checkpoints"
export ECHOJEPA_REPO_DIR="checkpoints/EchoJEPA"
export ECHOJEPA_CHECKPOINT="checkpoints/vitl-vmix22m-pt220-c55.pt"
export LV_SEGMENTATION_CHECKPOINT="checkpoints/deeplabv3_resnet50_random.pt"
export SICM_BASELINE_OUTPUT_DIR="outputs/echojepa_baseline"

python scripts/train_echojepa_baseline.py
~~~

This script extracts adjacent primary-peak ED-to-ED cycles, uses EchoJEPA ViT-L features with mean pooling across tokens and cycles, and trains an MLP classifier. It does not use LV-area physiological features or self-supervised adaptation. Training uses 10 frozen-backbone Stage A epochs and 20 Stage B epochs, with the last two backbone blocks unfrozen in Stage B. Its supervised loss uses the positive-class weight described above. Training metrics are in-sample, not an independent validation estimate. The final checkpoint and training-derived segmentation normalization are written under the configured output directory.

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

Each fold starts from the supplied EchoJEPA checkpoint and trains on four folds. All cycles belonging to an ID stay in that ID's fold. The script fits segmentation normalization using only the four training folds and applies it to that fold's training and validation videos. It writes fold assignments, per-fold checkpoints, predictions, metrics, and preprocessing-failure records. Only successfully preprocessed patients enter that fold's training or OOF metrics. The source defaults to reprocessing cycles for the CV run, which can take substantial time. Do not treat training-fold metrics as validation results.

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

The full-model prediction CSV includes the pseudonymous ID, input filename/path relative to the configured external-video root, number of cycles, SICM probability, a 0.5-threshold prediction, and a success/error status. Cycle extraction uses the primary ED-peak rule and does not add relaxed or fallback detection. The predictions are patient-level output and must not be committed.

### EchoJEPA baseline inference

Run baseline inference against the same external video layout. Point it to the baseline checkpoint and its training-set segmentation normalization:

~~~bash
export SICM_EXTERNAL_VIDEO_ROOT="/secure/path/external/videos"
export SICM_CHECKPOINT_DIR="checkpoints"
export ECHOJEPA_REPO_DIR="checkpoints/EchoJEPA"
export LV_SEGMENTATION_CHECKPOINT="checkpoints/deeplabv3_resnet50_random.pt"
export SICM_BASELINE_FINAL_MODEL="outputs/echojepa_baseline/final_model.pt"
export SICM_BASELINE_SEGMENTATION_NORMALIZATION_CSV="outputs/echojepa_baseline/data_normalization.csv"
export SICM_EXTERNAL_BASELINE_PREDICTIONS_CSV="outputs/external/echojepa_baseline_test_predictions.csv"

python scripts/infer_echojepa_baseline.py
~~~

This baseline also uses primary ED peaks and all adjacent cycles, then averages the visual cycle features. It does not use the LV-area physiological representation, labels, or test-time training. Its prediction CSV has the same `id` and `sicm_probability` fields expected by the evaluator.

## External evaluation

The external evaluator reproduces the manuscript-level discrimination, threshold-based classification, calibration/Brier, paired model comparison, decision-curve, and cTnT sensitivity-analysis workflow. Primary model comparison is performed only when the physiology-informed model and EchoJEPA baseline have successful predictions for the same labeled patient IDs.

### Primary analysis

Supply the prepared primary binary labels (cTnT >0.10 ng/mL), plus both prediction files:

~~~bash
export SICM_EXTERNAL_LABELS_CSV="/secure/path/external/id_label_primary.csv"
export SICM_EXTERNAL_FULL_PREDICTIONS_CSV="outputs/external/physiology_informed_test_predictions.csv"
export SICM_EXTERNAL_BASELINE_PREDICTIONS_CSV="outputs/external/echojepa_baseline_test_predictions.csv"
export SICM_EXTERNAL_EVALUATION_DIR="outputs/external/evaluation"

python scripts/evaluate_external_test_auc.py
~~~

The script calculates:

- AUROC with percentile-bootstrap 95% CI (2,000 resamples; seed 42 by default).
- Accuracy, sensitivity, specificity, F1 score, and confusion-matrix counts at the prespecified 0.5 probability threshold.
- Brier score for the physiology-informed model and the EchoJEPA baseline.
- A calibration curve for the physiology-informed model with pointwise 95% confidence intervals.
- Paired DeLong comparison of the two AUROCs, including the AUROC difference, 95% CI, and two-sided P value.
- Decision-curve net benefit for the physiology-informed model, EchoJEPA baseline, treat-all, and treat-none strategies.
- A paired primary-analysis table containing the identical patient IDs used by both models.

Key outputs include external_model_performance.csv, paired_delong_test.csv, paired_primary_external_predictions.csv, external_calibration_curve.csv/.png/.pdf, external_decision_curve.csv/.png/.pdf, and external_roc_comparison.png/.pdf.

### Sensitivity analyses

The manuscript uses the same locked model probabilities and changes only the cTnT-based reference definition:

1. Primary: episode cTnT >0.10 ng/mL.
2. Sensitivity analysis 1: episode cTnT >0.20 ng/mL.
3. Sensitivity analysis 2: exclude patients whose episode-level cTnT is 0.08-0.12 ng/mL, then retain the >0.10 ng/mL definition in the remaining cohort.

There are two supported ways to reproduce these analyses.

**Option A: derive all cTnT label sets from a secure cTnT file.** Provide one or more cTnT measurements per pseudonymous ID; the evaluator uses the maximum value during the septic episode, which is equivalent to the manuscript rule of classifying a patient as positive when at least one value exceeds the cutoff.

~~~bash
export SICM_EXTERNAL_CTNT_CSV="/secure/path/external/ctnt_during_septic_episode.csv"
export SICM_CTNT_ID_COLUMN="id"
export SICM_CTNT_VALUE_COLUMN="ctnt_ng_ml"

python scripts/evaluate_external_test_auc.py
~~~

**Option B: provide precomputed sensitivity-label files.**

~~~bash
export SICM_EXTERNAL_LABELS_020_CSV="/secure/path/external/id_label_ctnt_gt_020.csv"
export SICM_EXTERNAL_LABELS_GRAY_EXCLUDED_CSV="/secure/path/external/id_label_exclude_008_012.csv"

python scripts/evaluate_external_test_auc.py
~~~

If raw cTnT data and prepared label files are both supplied, the evaluator cross-checks their labels and stops on disagreement. The same full-model probability table is reused for all sensitivity analyses.

The manuscript specifies a calibration curve with pointwise 95% confidence intervals and decision-curve analysis, but it does not state the calibration binning rule, the method used to construct the pointwise calibration intervals, or the exact DCA threshold grid. Repository defaults are 10 quantile calibration bins, pointwise percentile-bootstrap intervals using the same bootstrap iteration count/seed, and threshold probabilities 0.01-0.99 in 0.01 increments. These presentation settings can be changed with SICM_CALIBRATION_BINS, SICM_CALIBRATION_STRATEGY, SICM_DCA_MIN_THRESHOLD, SICM_DCA_MAX_THRESHOLD, and SICM_DCA_STEP without changing model predictions.

## Research use

This software is provided for research and reproducibility. It is not a medical device and must not be used to diagnose or guide care. Apply the terms governing each dataset, checkpoint, and upstream package independently.

