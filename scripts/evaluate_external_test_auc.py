
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Set the {name} environment variable before running this script."
        )
    return Path(value).expanduser()


LABEL_CSV = required_env_path("SICM_EXTERNAL_LABELS_CSV")
FULL_PRED_CSV = Path(
    os.environ.get(
        "SICM_EXTERNAL_FULL_PREDICTIONS_CSV",
        "outputs/external/physiology_informed_test_predictions.csv",
    )
).expanduser()
BASELINE_PRED_CSV = Path(
    os.environ.get(
        "SICM_EXTERNAL_BASELINE_PREDICTIONS_CSV",
        "outputs/external/echojepa_baseline_test_predictions.csv",
    )
).expanduser()
EVALUATION_DIR = Path(
    os.environ.get("SICM_EXTERNAL_EVALUATION_DIR", "outputs/external/evaluation")
).expanduser()
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = EVALUATION_DIR / "external_model_performance.csv"
FULL_MERGED_CSV = EVALUATION_DIR / "full_model_predictions_with_labels.csv"
BASELINE_MERGED_CSV = EVALUATION_DIR / "echojepa_baseline_predictions_with_labels.csv"
ROC_FIGURE = EVALUATION_DIR / "external_roc_comparison.png"

THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42


def load_labels(path):
    df = pd.read_csv(path, dtype=str)

    if df.shape[1] < 2:
        raise RuntimeError(
            "label.csv must contain at least two columns."
        )

    df = df.iloc[:, :2].copy()
    df.columns = ["id", "label"]

    df["id"] = (
        df["id"]
        .astype(str)
        .str.strip()
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    df = df[
        df["label"].isin([0, 1])
    ].copy()

    df["label"] = df["label"].astype(int)

    conflict = (
        df.groupby("id")["label"]
        .nunique()
    )

    conflict = conflict[
        conflict > 1
    ]

    if not conflict.empty:
        raise RuntimeError(
            "Conflicting labels found for IDs: "
            + ", ".join(conflict.index.astype(str))
        )

    return df.drop_duplicates(
        subset=["id"],
        keep="first",
    )


def bootstrap_auc_ci(labels, probs):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probs = np.asarray(
        probs,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(labels)
    aucs = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        y = labels[idx]
        p = probs[idx]

        if np.unique(y).size < 2:
            continue

        aucs.append(
            roc_auc_score(
                y,
                p,
            )
        )

    if not aucs:
        return np.nan, np.nan

    return (
        float(
            np.percentile(
                aucs,
                2.5,
            )
        ),
        float(
            np.percentile(
                aucs,
                97.5,
            )
        ),
    )


def evaluate_one(
    label_df,
    pred_path,
    model_name,
    merged_path,
):
    pred_df = pd.read_csv(
        pred_path,
        dtype={"id": str},
    )

    required = {
        "id",
        "sicm_probability",
    }

    missing = required - set(
        pred_df.columns
    )

    if missing:
        raise RuntimeError(
            f"{pred_path.name} is missing columns: {missing}"
        )

    pred_df["id"] = (
        pred_df["id"]
        .astype(str)
        .str.strip()
    )

    if "status" in pred_df.columns:
        pred_df = pred_df[
            pred_df["status"] == "success"
        ].copy()

    pred_df["sicm_probability"] = pd.to_numeric(
        pred_df["sicm_probability"],
        errors="coerce",
    )

    pred_df = pred_df.dropna(
        subset=["sicm_probability"]
    )

    if pred_df["id"].duplicated().any():
        dup = pred_df.loc[
            pred_df["id"].duplicated(
                keep=False
            ),
            "id",
        ].unique()

        raise RuntimeError(
            f"{model_name}: duplicated prediction IDs found: "
            f"{dup[:20].tolist()}"
        )

    merged = label_df.merge(
        pred_df,
        on="id",
        how="inner",
    )

    if merged.empty:
        raise RuntimeError(
            f"{model_name}: no IDs matched."
        )

    labels = merged["label"].to_numpy(
        dtype=np.int64
    )

    probs = merged["sicm_probability"].to_numpy(
        dtype=np.float64
    )

    if np.unique(labels).size != 2:
        raise RuntimeError(
            f"{model_name}: both classes are required for AUC."
        )

    preds = (
        probs >= THRESHOLD
    ).astype(np.int64)

    auc = roc_auc_score(
        labels,
        probs,
    )

    ci_low, ci_high = bootstrap_auc_ci(
        labels,
        probs,
    )

    acc = accuracy_score(
        labels,
        preds,
    )

    f1 = f1_score(
        labels,
        preds,
        zero_division=0,
    )

    sens = recall_score(
        labels,
        preds,
        pos_label=1,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    ).ravel()

    spec = (
        tn / max(tn + fp, 1)
    )

    merged["prediction_at_0.5"] = preds

    merged.to_csv(
        merged_path,
        index=False,
        encoding="utf-8-sig",
    )

    fpr, tpr, _ = roc_curve(
        labels,
        probs,
    )

    metrics = {
        "model": model_name,
        "n": int(len(merged)),
        "n_SICM": int((labels == 1).sum()),
        "n_sepsis_only": int((labels == 0).sum()),
        "auc": float(auc),
        "auc_95ci_lower": ci_low,
        "auc_95ci_upper": ci_high,
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "threshold": THRESHOLD,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return metrics, fpr, tpr


for path in [
    LABEL_CSV,
    FULL_PRED_CSV,
    BASELINE_PRED_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{path}"
        )


label_df = load_labels(
    LABEL_CSV
)


full_metrics, full_fpr, full_tpr = evaluate_one(
    label_df,
    FULL_PRED_CSV,
    "Physiology-informed model",
    FULL_MERGED_CSV,
)


baseline_metrics, baseline_fpr, baseline_tpr = evaluate_one(
    label_df,
    BASELINE_PRED_CSV,
    "EchoJEPA baseline",
    BASELINE_MERGED_CSV,
)


summary_df = pd.DataFrame(
    [
        full_metrics,
        baseline_metrics,
    ]
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
    encoding="utf-8-sig",
)


plt.figure(
    figsize=(6, 6)
)

plt.plot(
    full_fpr,
    full_tpr,
    label=(
        "Physiology-informed model "
        f"(AUC={full_metrics['auc']:.3f})"
    ),
)

plt.plot(
    baseline_fpr,
    baseline_tpr,
    label=(
        "EchoJEPA baseline "
        f"(AUC={baseline_metrics['auc']:.3f})"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "External validation ROC"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    ROC_FIGURE,
    dpi=600,
)

plt.close()


print(
    "\n"
    + "=" * 80
)

print(
    "EXTERNAL VALIDATION RESULTS"
)

print(
    "=" * 80
)

for m in [
    full_metrics,
    baseline_metrics,
]:
    print(
        f"\n{m['model']}"
    )
    print(
        f"N           : {m['n']}"
    )
    print(
        f"AUC         : {m['auc']:.4f} "
        f"(95% CI {m['auc_95ci_lower']:.4f}-"
        f"{m['auc_95ci_upper']:.4f})"
    )
    print(
        f"Accuracy    : {m['accuracy']:.4f}"
    )
    print(
        f"Sensitivity : {m['sensitivity']:.4f}"
    )
    print(
        f"Specificity : {m['specificity']:.4f}"
    )
    print(
        f"F1          : {m['f1']:.4f}"
    )
    print(
        f"TP/TN/FP/FN : "
        f"{m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}"
    )

print(
    "\nSaved:"
)
print(
    SUMMARY_CSV
)
print(
    FULL_MERGED_CSV
)
print(
    BASELINE_MERGED_CSV
)
print(
    ROC_FIGURE
)

