# Input data format

This repository contains no clinical records, labels, echo videos, or model weights. Keep all inputs and generated files in access-controlled storage outside the Git checkout. Use pseudonymous IDs; do not put names, MRNs, dates of birth, or other direct identifiers in IDs or file names.

## Development labels

scripts/train.py, scripts/train_echojepa_baseline.py, and scripts/cross_validate_patient_5fold.py expect a UTF-8 CSV whose columns are `id` and `label` in this order. Use exactly these two columns:

~~~csv
id,label
<study_id_001>,0
<study_id_002>,1
~~~

- id: one stable pseudonymous patient ID. The modeling code assumes one index study per ID.
- label: integer 0 (sepsis-only control) or 1 (SICM).
- Each labeled ID must match a `.npy` file whose filename stem is the ID. Full-model training, baseline training, and CV search the configured video root recursively. CV expects one raw video per ID.
- The scripts do not derive these labels or enforce the manuscript's cohort-selection, clinical exclusion, or time-window criteria.

The full-model training script and the direct EchoJEPA baseline both validate exact `id,label` columns, binary values, conflicting labels, and one matching `<id>.npy` file for every labeled ID. CV validates the same ID/video relationship before assigning folds. Do not add a real label CSV to this repository; use an external secure path.

## NumPy cine-loop arrays

One file represents one selected A4C cine loop. The loaders accept these layouts:

- T x H x W grayscale
- T x H x W x C, where C is 1 or 3
- T x C x H x W, where C is 1 or 3

The values should consistently encode the frame intensities, ordinarily as 8-bit values (0 to 255) or normalized floating-point values (0 to 1). The physiology-informed pipeline converts supported arrays for LV-segmentation/cycle preprocessing and EchoJEPA encoding. The direct EchoJEPA baseline instead converts the complete cine loop to grayscale, uniformly samples 16 frames, resizes them to 224 x 224, replicates them to three channels, and applies EchoJEPA normalization. Unsupported dimensions/channel layouts fail at load time.

Development file layout (used by both model-training scripts and CV):

~~~text
<development_video_root>/
|-- <study_id_001>.npy
|-- nested/
|   `-- <study_id_002>.npy
`-- ...
~~~

Any subdirectory component named outcome is excluded by the source scanners. Do not place generated cycle caches inside the raw input tree.

The direct EchoJEPA baseline uses the same development ID/file matching convention as the full model, but it does **not** perform LV segmentation, ED/ES detection, cardiac-cycle extraction, physiological feature construction, attention pooling, or MIL.

## External inference video layout

Both external inference scripts recursively find `.npy` files and use the **immediate parent directory name** as the prediction ID. They skip paths containing a directory component named `outcome` or `cycles`.

~~~text
<external_video_root>/
|-- <study_id_101>/
|   `-- selected_a4c.npy
|-- <study_id_102>/
|   `-- selected_a4c.npy
`-- ...
~~~

For reliable evaluation, supply exactly one selected loop per pseudonymous ID and ensure that the parent-folder ID matches the ID in the external label CSV. The training loader instead uses the video filename stem as the ID; preserve the distinct conventions when preparing the two cohorts.

## External label and prediction files

The external evaluator uses **three precomputed binary label CSVs**, each with columns named `id,label`:

1. Primary external analysis: cTnT >0.10 ng/mL.
2. Sensitivity analysis 1: cTnT >0.20 ng/mL.
3. Sensitivity analysis 2: patients with cTnT 0.08-0.12 ng/mL are excluded, and the >0.10 ng/mL definition is retained in the remaining cohort.

Example format:

~~~csv
id,label
<study_id_101>,1
<study_id_102>,0
~~~

The evaluator reads prediction CSVs with at least:

- id
- sicm_probability (or probability)

If a status column is present, rows whose status is not success are excluded. Duplicate successful prediction IDs are rejected.

The primary full-model analysis merges the primary label file with the locked full-model predictions. Model-versus-baseline ROC comparison, paired DeLong testing, and decision-curve analysis use the subset with successful predictions from both models. The two sensitivity analyses reuse the same locked full-model probability table and change only the precomputed label CSV.

The repository does not derive these external labels from raw cTnT values. Clinical label derivation is performed upstream, outside this repository.

The source scripts write files containing IDs, labels, paths, or predictions. Keep all real patient-level label files, manifests, fold assignments, merged evaluation tables, and prediction CSVs outside version control.

