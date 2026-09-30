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

The full-model training script validates column names and binary values, rejects conflicting labels, and requires a matching video for every ID. The baseline training script reads the first two columns positionally as `id,label`, keeps binary-labeled matched videos, and requires both classes. CV validates the labels and ID/video matches before assigning folds. Do not add a real label CSV to this repository; use an external secure path.

## NumPy cine-loop arrays

One file represents one selected A4C cine loop. The loaders accept these layouts:

- T x H x W grayscale
- T x H x W x C, where C is 1 or 3
- T x C x H x W, where C is 1 or 3

The values should consistently encode the frame intensities, ordinarily as 8-bit values (0 to 255) or normalized floating-point values (0 to 1). The source code converts supported arrays to grayscale for cycle preparation and constructs three-channel input for the EchoNet-Dynamic segmenter and EchoJEPA encoder. Unsupported dimensions/channel layouts fail at load time.

Development file layout (used by both model-training scripts and CV):

~~~text
<development_video_root>/
|-- <study_id_001>.npy
|-- nested/
|   `-- <study_id_002>.npy
`-- ...
~~~

Any subdirectory component named outcome is excluded by the source scanners. Do not place generated cycle caches inside the raw input tree.

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

The evaluator expects a binary label CSV; provide columns named id,label even though its current loader treats the first two columns positionally. IDs must match the inference outputs. The evaluator reads prediction CSVs with at least:

- id
- sicm_probability

If present, rows with status != success are discarded. Duplicate prediction IDs are rejected. Both classes must remain after matching. The same prediction table can be evaluated against separately prepared primary and sensitivity-analysis labels.

The source scripts write files containing IDs, labels, paths, or predictions. Keep these files out of Git, including manifests, fold assignments, merged evaluation tables, and prediction CSVs.

