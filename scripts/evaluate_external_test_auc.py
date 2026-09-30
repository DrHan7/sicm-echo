import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return Path(value).expanduser()


def optional_env_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


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

CTNT_CSV = optional_env_path("SICM_EXTERNAL_CTNT_CSV")
LABEL_020_CSV = optional_env_path("SICM_EXTERNAL_LABELS_020_CSV")
LABEL_GRAY_EXCLUDED_CSV = optional_env_path(
    "SICM_EXTERNAL_LABELS_GRAY_EXCLUDED_CSV"
)
CTNT_VALUE_COLUMN = os.environ.get("SICM_CTNT_VALUE_COLUMN", "ctnt_ng_ml")
CTNT_ID_COLUMN = os.environ.get("SICM_CTNT_ID_COLUMN", "id")

THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = int(os.environ.get("SICM_BOOTSTRAP_ITERATIONS", "2000"))
BOOTSTRAP_SEED = int(os.environ.get("SICM_BOOTSTRAP_SEED", "42"))
CALIBRATION_BINS = int(os.environ.get("SICM_CALIBRATION_BINS", "10"))
CALIBRATION_STRATEGY = os.environ.get(
    "SICM_CALIBRATION_STRATEGY", "quantile"
).strip().lower()
DCA_MIN_THRESHOLD = float(os.environ.get("SICM_DCA_MIN_THRESHOLD", "0.01"))
DCA_MAX_THRESHOLD = float(os.environ.get("SICM_DCA_MAX_THRESHOLD", "0.99"))
DCA_STEP = float(os.environ.get("SICM_DCA_STEP", "0.01"))

if CALIBRATION_STRATEGY not in {"uniform", "quantile"}:
    raise RuntimeError("SICM_CALIBRATION_STRATEGY must be 'uniform' or 'quantile'.")
if not (0 < DCA_MIN_THRESHOLD < DCA_MAX_THRESHOLD < 1):
    raise RuntimeError("DCA thresholds must satisfy 0 < min < max < 1.")
if DCA_STEP <= 0:
    raise RuntimeError("SICM_DCA_STEP must be positive.")


def load_labels(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "id" not in normalized or "label" not in normalized:
        raise RuntimeError(f"{path.name} must contain columns named id and label.")

    out = df[[normalized["id"], normalized["label"]]].copy()
    out.columns = ["id", "label"]
    out["id"] = out["id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")

    if out["id"].eq("").any():
        raise RuntimeError(f"{path.name} contains an empty ID.")
    if not out["label"].isin([0, 1]).all():
        raise RuntimeError(f"{path.name} contains labels other than 0/1.")

    conflicts = out.groupby("id")["label"].nunique()
    if (conflicts > 1).any():
        ids = conflicts[conflicts > 1].index.astype(str).tolist()
        raise RuntimeError(
            f"{path.name} contains conflicting labels for IDs: {ids[:20]}"
        )

    out = out.drop_duplicates("id").copy()
    out["label"] = out["label"].astype(int)
    return out


def load_predictions(path, probability_name):
    df = pd.read_csv(path, dtype={"id": str}, encoding="utf-8-sig")
    required = {"id", "sicm_probability"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {sorted(missing)}")

    out = df.copy()
    out["id"] = out["id"].astype(str).str.strip()
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "success"].copy()

    out["sicm_probability"] = pd.to_numeric(
        out["sicm_probability"], errors="coerce"
    )
    out = out.dropna(subset=["id", "sicm_probability"]).copy()

    if not out["sicm_probability"].between(0, 1, inclusive="both").all():
        raise RuntimeError(f"{path.name} contains probabilities outside [0, 1].")

    if out["id"].duplicated().any():
        ids = out.loc[out["id"].duplicated(keep=False), "id"].unique().tolist()
        raise RuntimeError(
            f"{path.name} contains duplicated successful prediction IDs: {ids[:20]}"
        )

    return out[["id", "sicm_probability"]].rename(
        columns={"sicm_probability": probability_name}
    )


def bootstrap_auc_ci(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aucs = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, len(y), size=len(y))
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))

    if not aucs:
        return np.nan, np.nan

    return (
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def classification_metrics(labels, probabilities, model_name, analysis_name):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)

    if np.unique(y).size != 2:
        raise RuntimeError(
            f"{analysis_name}/{model_name}: both classes are required for AUROC."
        )

    pred = (p >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p))
    ci_low, ci_high = bootstrap_auc_ci(y, p)

    return {
        "analysis": analysis_name,
        "model": model_name,
        "n": int(len(y)),
        "n_SICM": int((y == 1).sum()),
        "n_sepsis_only": int((y == 0).sum()),
        "auc": auc,
        "auc_95ci_lower": ci_low,
        "auc_95ci_upper": ci_high,
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": THRESHOLD,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def make_primary_paired_dataset(label_df, full_df, baseline_df):
    label_ids = set(label_df["id"])
    full_ids = set(full_df["id"]) & label_ids
    baseline_ids = set(baseline_df["id"]) & label_ids

    missing_full = sorted(label_ids - full_ids)
    missing_baseline = sorted(label_ids - baseline_ids)
    if missing_full or missing_baseline:
        raise RuntimeError(
            "Every primary external-analysis ID must have a successful prediction "
            "from both models.\n"
            f"Missing full-model predictions ({len(missing_full)}): {missing_full[:20]}\n"
            f"Missing baseline predictions ({len(missing_baseline)}): {missing_baseline[:20]}"
        )

    if full_ids != baseline_ids:
        only_full = sorted(full_ids - baseline_ids)
        only_baseline = sorted(baseline_ids - full_ids)
        raise RuntimeError(
            "The physiology-informed model and EchoJEPA baseline do not have "
            "successful predictions for the same labeled IDs. Paired DeLong "
            "comparison requires an identical cohort.\n"
            f"Only full model ({len(only_full)}): {only_full[:20]}\n"
            f"Only baseline ({len(only_baseline)}): {only_baseline[:20]}"
        )

    paired = (
        label_df[label_df["id"].isin(full_ids)]
        .merge(full_df, on="id", how="inner")
        .merge(baseline_df, on="id", how="inner")
        .sort_values("id")
        .reset_index(drop=True)
    )

    if len(paired) != len(full_ids):
        raise RuntimeError("Paired external cohort construction failed.")

    if paired["label"].nunique() != 2:
        raise RuntimeError("Both classes are required in the paired external cohort.")

    return paired


def delong_placement_values(labels, predictions):
    y = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(predictions, dtype=np.float64)

    if preds.ndim == 1:
        preds = preds[None, :]
    if preds.shape[1] != len(y):
        raise ValueError("Prediction matrix does not match label length.")

    positive = preds[:, y == 1]
    negative = preds[:, y == 0]
    m = positive.shape[1]
    n = negative.shape[1]

    if m < 2 or n < 2:
        raise RuntimeError("Paired DeLong test requires at least two cases per class.")

    v10 = np.empty((preds.shape[0], m), dtype=np.float64)
    v01 = np.empty((preds.shape[0], n), dtype=np.float64)

    for k in range(preds.shape[0]):
        pos = positive[k]
        neg = negative[k]
        comparison = (
            (pos[:, None] > neg[None, :]).astype(np.float64)
            + 0.5 * (pos[:, None] == neg[None, :]).astype(np.float64)
        )
        v10[k] = comparison.mean(axis=1)
        v01[k] = comparison.mean(axis=0)

    aucs = v10.mean(axis=1)
    sx = np.cov(v10, bias=False)
    sy = np.cov(v01, bias=False)

    if preds.shape[0] == 1:
        sx = np.asarray([[float(sx)]])
        sy = np.asarray([[float(sy)]])

    covariance = sx / m + sy / n
    return aucs, covariance


def paired_delong_test(labels, full_prob, baseline_prob):
    predictions = np.vstack(
        [
            np.asarray(full_prob, dtype=np.float64),
            np.asarray(baseline_prob, dtype=np.float64),
        ]
    )
    aucs, covariance = delong_placement_values(labels, predictions)

    difference = float(aucs[0] - aucs[1])
    variance = float(
        covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    )
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance))

    if se == 0:
        z = np.inf if difference != 0 else 0.0
        p_value = 0.0 if difference != 0 else 1.0
        ci_low = difference
        ci_high = difference
    else:
        z = difference / se
        p_value = float(2 * norm.sf(abs(z)))
        critical = float(norm.ppf(0.975))
        ci_low = difference - critical * se
        ci_high = difference + critical * se

    return {
        "full_model_auc": float(aucs[0]),
        "baseline_auc": float(aucs[1]),
        "auc_difference_full_minus_baseline": difference,
        "difference_95ci_lower": float(ci_low),
        "difference_95ci_upper": float(ci_high),
        "standard_error": se,
        "z": float(z),
        "p_value_two_sided": p_value,
        "n": int(len(labels)),
    }


def save_roc_plot(paired, full_metrics, baseline_metrics):
    y = paired["label"].to_numpy(dtype=np.int64)
    full_prob = paired["full_probability"].to_numpy(dtype=np.float64)
    baseline_prob = paired["baseline_probability"].to_numpy(dtype=np.float64)

    full_fpr, full_tpr, _ = roc_curve(y, full_prob)
    baseline_fpr, baseline_tpr, _ = roc_curve(y, baseline_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        full_fpr,
        full_tpr,
        linewidth=1.5,
        label=f"Physiology-informed model (AUC={full_metrics['auc']:.3f})",
    )
    ax.plot(
        baseline_fpr,
        baseline_tpr,
        linewidth=1.5,
        label=f"EchoJEPA baseline (AUC={baseline_metrics['auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("External validation ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "external_roc_comparison.png", dpi=600)
    fig.savefig(EVALUATION_DIR / "external_roc_comparison.pdf")
    plt.close(fig)


def save_calibration(paired):
    y = paired["label"].to_numpy(dtype=np.int64)
    p = paired["full_probability"].to_numpy(dtype=np.float64)

    fraction_positive, mean_predicted = calibration_curve(
        y,
        p,
        n_bins=CALIBRATION_BINS,
        strategy=CALIBRATION_STRATEGY,
    )

    calibration_df = pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted,
            "observed_event_fraction": fraction_positive,
        }
    )
    calibration_df.to_csv(
        EVALUATION_DIR / "external_calibration_curve.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        linewidth=1.5,
        label="Physiology-informed model",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event fraction")
    ax.set_title("External calibration")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "external_calibration_curve.png", dpi=600)
    fig.savefig(EVALUATION_DIR / "external_calibration_curve.pdf")
    plt.close(fig)

    return float(brier_score_loss(y, p))


def decision_curve_table(labels, full_prob, baseline_prob):
    y = np.asarray(labels, dtype=np.int64)
    full = np.asarray(full_prob, dtype=np.float64)
    baseline = np.asarray(baseline_prob, dtype=np.float64)

    thresholds = np.arange(
        DCA_MIN_THRESHOLD,
        DCA_MAX_THRESHOLD + DCA_STEP / 2,
        DCA_STEP,
        dtype=np.float64,
    )
    thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]
    n = len(y)
    prevalence = float(y.mean())

    rows = []
    for pt in thresholds:
        weight = pt / (1.0 - pt)

        def net_benefit(prob):
            pred = prob >= pt
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            return tp / n - fp / n * weight

        rows.append(
            {
                "threshold_probability": float(pt),
                "physiology_informed_net_benefit": float(net_benefit(full)),
                "echojepa_baseline_net_benefit": float(net_benefit(baseline)),
                "treat_all_net_benefit": float(
                    prevalence - (1.0 - prevalence) * weight
                ),
                "treat_none_net_benefit": 0.0,
            }
        )

    return pd.DataFrame(rows)


def save_decision_curve(paired):
    dca = decision_curve_table(
        paired["label"],
        paired["full_probability"],
        paired["baseline_probability"],
    )
    dca.to_csv(EVALUATION_DIR / "external_decision_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        dca["threshold_probability"],
        dca["physiology_informed_net_benefit"],
        label="Physiology-informed model",
    )
    ax.plot(
        dca["threshold_probability"],
        dca["echojepa_baseline_net_benefit"],
        label="EchoJEPA baseline",
    )
    ax.plot(
        dca["threshold_probability"],
        dca["treat_all_net_benefit"],
        linestyle="--",
        label="Treat all",
    )
    ax.plot(
        dca["threshold_probability"],
        dca["treat_none_net_benefit"],
        linestyle=":",
        label="Treat none",
    )
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("External decision curve analysis")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "external_decision_curve.png", dpi=600)
    fig.savefig(EVALUATION_DIR / "external_decision_curve.pdf")
    plt.close(fig)


def load_ctnt_labels(path):
    df = pd.read_csv(path, dtype={CTNT_ID_COLUMN: str}, encoding="utf-8-sig")
    if CTNT_ID_COLUMN not in df.columns or CTNT_VALUE_COLUMN not in df.columns:
        raise RuntimeError(
            f"{path.name} must contain '{CTNT_ID_COLUMN}' and "
            f"'{CTNT_VALUE_COLUMN}' columns."
        )

    x = df[[CTNT_ID_COLUMN, CTNT_VALUE_COLUMN]].copy()
    x.columns = ["id", "ctnt_ng_ml"]
    x["id"] = x["id"].astype(str).str.strip()
    x["ctnt_ng_ml"] = pd.to_numeric(x["ctnt_ng_ml"], errors="coerce")
    x = x.dropna(subset=["id", "ctnt_ng_ml"])
    if x["id"].eq("").any():
        raise RuntimeError(f"{path.name} contains an empty ID.")
    if (x["ctnt_ng_ml"] < 0).any():
        raise RuntimeError(f"{path.name} contains negative cTnT values.")

    episode = (
        x.groupby("id", as_index=False)["ctnt_ng_ml"]
        .max()
        .rename(columns={"ctnt_ng_ml": "episode_max_ctnt_ng_ml"})
    )

    primary = episode[["id"]].copy()
    primary["label"] = (episode["episode_max_ctnt_ng_ml"] > 0.10).astype(int)

    high = episode[["id"]].copy()
    high["label"] = (episode["episode_max_ctnt_ng_ml"] > 0.20).astype(int)

    gray_mask = episode["episode_max_ctnt_ng_ml"].between(
        0.08, 0.12, inclusive="both"
    )
    gray = episode.loc[~gray_mask, ["id"]].copy()
    gray["label"] = (
        episode.loc[~gray_mask, "episode_max_ctnt_ng_ml"] > 0.10
    ).astype(int).to_numpy()

    episode.to_csv(
        EVALUATION_DIR / "episode_max_ctnt_ng_ml.csv",
        index=False,
    )
    return primary, high, gray


def compare_label_sets(reference, derived, name):
    missing = sorted(set(reference["id"]) - set(derived["id"]))
    if missing:
        raise RuntimeError(
            f"{name}: cTnT-derived labels are missing {len(missing)} required IDs: "
            f"{missing[:20]}"
        )

    merged = reference.merge(
        derived,
        on="id",
        how="inner",
        suffixes=("_reference", "_derived"),
    )
    mismatched = merged[merged["label_reference"] != merged["label_derived"]]
    if not mismatched.empty:
        raise RuntimeError(
            f"{name}: prepared labels disagree with cTnT-derived labels for "
            f"{len(mismatched)} IDs."
        )


def sensitivity_metrics(label_df, full_df, analysis_name):
    missing = sorted(set(label_df["id"]) - set(full_df["id"]))
    if missing:
        raise RuntimeError(
            f"{analysis_name}: {len(missing)} labeled IDs have no successful "
            f"full-model prediction: {missing[:20]}"
        )

    merged = (
        label_df.merge(full_df, on="id", how="inner")
        .sort_values("id")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise RuntimeError(f"{analysis_name}: no prediction IDs matched labels.")

    metrics = classification_metrics(
        merged["label"],
        merged["full_probability"],
        "Physiology-informed model",
        analysis_name,
    )
    return merged, metrics


for path in [LABEL_CSV, FULL_PRED_CSV, BASELINE_PRED_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find:\n{path}")

primary_labels = load_labels(LABEL_CSV)
full_predictions = load_predictions(FULL_PRED_CSV, "full_probability")
baseline_predictions = load_predictions(BASELINE_PRED_CSV, "baseline_probability")

paired = make_primary_paired_dataset(
    primary_labels,
    full_predictions,
    baseline_predictions,
)
paired.to_csv(
    EVALUATION_DIR / "paired_primary_external_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

full_metrics = classification_metrics(
    paired["label"],
    paired["full_probability"],
    "Physiology-informed model",
    "primary_ctnt_gt_0.10",
)
baseline_metrics = classification_metrics(
    paired["label"],
    paired["baseline_probability"],
    "EchoJEPA baseline",
    "primary_ctnt_gt_0.10",
)

delong = paired_delong_test(
    paired["label"],
    paired["full_probability"],
    paired["baseline_probability"],
)
pd.DataFrame([delong]).to_csv(
    EVALUATION_DIR / "paired_delong_test.csv",
    index=False,
)

save_roc_plot(paired, full_metrics, baseline_metrics)
brier = save_calibration(paired)
save_decision_curve(paired)

if not np.isclose(brier, full_metrics["brier"], atol=1e-12, rtol=0):
    raise RuntimeError("Brier-score consistency check failed.")

metric_rows = [full_metrics, baseline_metrics]

derived_primary = derived_020 = derived_gray = None
if CTNT_CSV is not None:
    if not CTNT_CSV.exists():
        raise FileNotFoundError(f"Cannot find:\n{CTNT_CSV}")
    derived_primary, derived_020, derived_gray = load_ctnt_labels(CTNT_CSV)
    compare_label_sets(
        primary_labels,
        derived_primary,
        "Primary cTnT >0.10 ng/mL analysis",
    )

if LABEL_020_CSV is not None:
    if not LABEL_020_CSV.exists():
        raise FileNotFoundError(f"Cannot find:\n{LABEL_020_CSV}")
    labels_020 = load_labels(LABEL_020_CSV)
    if derived_020 is not None:
        compare_label_sets(
            labels_020,
            derived_020,
            "Sensitivity cTnT >0.20 ng/mL analysis",
        )
elif derived_020 is not None:
    labels_020 = derived_020
else:
    labels_020 = None

if LABEL_GRAY_EXCLUDED_CSV is not None:
    if not LABEL_GRAY_EXCLUDED_CSV.exists():
        raise FileNotFoundError(f"Cannot find:\n{LABEL_GRAY_EXCLUDED_CSV}")
    labels_gray = load_labels(LABEL_GRAY_EXCLUDED_CSV)
    if derived_gray is not None:
        compare_label_sets(
            labels_gray,
            derived_gray,
            "Sensitivity exclusion 0.08-0.12 ng/mL analysis",
        )
elif derived_gray is not None:
    labels_gray = derived_gray
else:
    labels_gray = None

if labels_020 is not None:
    merged_020, metrics_020 = sensitivity_metrics(
        labels_020,
        full_predictions,
        "sensitivity_ctnt_gt_0.20",
    )
    merged_020.to_csv(
        EVALUATION_DIR / "sensitivity_ctnt_gt_0.20_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_rows.append(metrics_020)

if labels_gray is not None:
    merged_gray, metrics_gray = sensitivity_metrics(
        labels_gray,
        full_predictions,
        "sensitivity_exclude_0.08_to_0.12",
    )
    merged_gray.to_csv(
        EVALUATION_DIR / "sensitivity_exclude_0.08_to_0.12_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_rows.append(metrics_gray)

performance_df = pd.DataFrame(metric_rows)
performance_df.to_csv(
    EVALUATION_DIR / "external_model_performance.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 80)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 80)

for m in metric_rows:
    print(f"\n{m['analysis']} | {m['model']}")
    print(f"N           : {m['n']}")
    print(f"SICM/control: {m['n_SICM']}/{m['n_sepsis_only']}")
    print(
        f"AUC         : {m['auc']:.4f} "
        f"(95% CI {m['auc_95ci_lower']:.4f}-"
        f"{m['auc_95ci_upper']:.4f})"
    )
    print(f"Brier       : {m['brier']:.4f}")
    print(f"Accuracy    : {m['accuracy']:.4f}")
    print(f"Sensitivity : {m['sensitivity']:.4f}")
    print(f"Specificity : {m['specificity']:.4f}")
    print(f"F1          : {m['f1']:.4f}")
    print(f"TP/TN/FP/FN : {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

print("\nPaired DeLong comparison")
print(
    f"Delta AUC   : {delong['auc_difference_full_minus_baseline']:.4f} "
    f"(95% CI {delong['difference_95ci_lower']:.4f}-"
    f"{delong['difference_95ci_upper']:.4f})"
)
print(f"P (2-sided) : {delong['p_value_two_sided']:.6g}")
print("\nOutputs:", EVALUATION_DIR)
