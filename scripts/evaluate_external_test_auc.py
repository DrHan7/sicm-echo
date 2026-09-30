import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def required_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return Path(value).expanduser()


PRIMARY_LABEL_CSV = required_env_path("SICM_EXTERNAL_LABELS_CSV")
LABEL_020_CSV = required_env_path("SICM_EXTERNAL_LABELS_020_CSV")
LABEL_GRAY_EXCLUDED_CSV = required_env_path(
    "SICM_EXTERNAL_LABELS_GRAY_EXCLUDED_CSV"
)

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

THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42


CALIBRATION_BINS = 10


DCA_THRESHOLD_MIN = 0.05
DCA_THRESHOLD_MAX = 0.60
DCA_POINTS = 200


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
    if "id" not in df.columns:
        raise RuntimeError(f"{path.name} is missing column: id")

    if "sicm_probability" in df.columns:
        probability_column = "sicm_probability"
    elif "probability" in df.columns:
        probability_column = "probability"
    else:
        raise RuntimeError(
            f"{path.name} must contain sicm_probability or probability."
        )

    out = df.copy()
    out["id"] = out["id"].astype(str).str.strip()

    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "success"].copy()

    out[probability_name] = pd.to_numeric(
        out[probability_column],
        errors="coerce",
    )
    out = out.dropna(subset=["id", probability_name]).copy()

    if not out[probability_name].between(0, 1, inclusive="both").all():
        raise RuntimeError(f"{path.name} contains probabilities outside [0, 1].")

    if out["id"].duplicated().any():
        ids = out.loc[out["id"].duplicated(keep=False), "id"].unique().tolist()
        raise RuntimeError(
            f"{path.name} contains duplicated successful prediction IDs: {ids[:20]}"
        )

    return out[["id", probability_name]]


def bootstrap_auc_ci(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_values = []
    n = len(y_true)

    for _ in range(BOOTSTRAP_ITERATIONS):
        index = rng.integers(0, n, size=n)
        y_sample = y_true[index]
        p_sample = y_prob[index]
        if np.unique(y_sample).size < 2:
            continue
        auc_values.append(roc_auc_score(y_sample, p_sample))

    if not auc_values:
        return np.nan, np.nan

    return (
        float(np.percentile(auc_values, 2.5)),
        float(np.percentile(auc_values, 97.5)),
    )


def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def fast_delong(predictions_sorted_transposed, positive_count):
    m = int(positive_count)
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, covariance


def paired_delong_test(y_true, probability_1, probability_2):
    y_true = np.asarray(y_true, dtype=int)
    probability_1 = np.asarray(probability_1, dtype=float)
    probability_2 = np.asarray(probability_2, dtype=float)

    if not (
        len(y_true) == len(probability_1) == len(probability_2)
    ):
        raise ValueError(
            "DeLong requires paired predictions from identical patients."
        )

    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError("DeLong requires both classes.")

    order = np.argsort(-y_true)
    predictions = np.vstack([probability_1, probability_2])[:, order]
    aucs, covariance = fast_delong(predictions, positive_count)

    delta_auc = aucs[0] - aucs[1]
    variance_delta = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2 * covariance[0, 1]
    )
    variance_delta = max(float(variance_delta), 0.0)
    se_delta = math.sqrt(variance_delta)

    if se_delta == 0:
        p_value = 1.0 if delta_auc == 0 else 0.0
        delta_ci_lower = delta_auc
        delta_ci_upper = delta_auc
    else:
        z_value = abs(delta_auc) / se_delta
        p_value = math.erfc(z_value / math.sqrt(2.0))
        delta_ci_lower = delta_auc - 1.96 * se_delta
        delta_ci_upper = delta_auc + 1.96 * se_delta

    return {
        "auc_full": float(aucs[0]),
        "auc_baseline": float(aucs[1]),
        "delta_auc": float(delta_auc),
        "delta_ci_lower": float(delta_ci_lower),
        "delta_ci_upper": float(delta_ci_upper),
        "p_value": float(p_value),
    }


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1.0 + z ** 2 / total
    centre = (
        proportion + z ** 2 / (2.0 * total)
    ) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z ** 2 / (4.0 * total ** 2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def calibration_with_ci(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if len(edges) < 2:
        raise RuntimeError("Calibration probabilities do not define usable bins.")

    bin_index = np.digitize(
        y_prob,
        edges[1:-1],
        right=True,
    )

    mean_pred = []
    observed = []
    lower_ci = []
    upper_ci = []
    bin_n = []

    for i in range(len(edges) - 1):
        mask = bin_index == i
        n = int(mask.sum())
        if n == 0:
            continue

        mean_probability = float(np.mean(y_prob[mask]))
        successes = int(np.sum(y_true[mask]))
        observed_rate = successes / n
        low, high = wilson_interval(successes, n)

        mean_pred.append(mean_probability)
        observed.append(observed_rate)
        lower_ci.append(low)
        upper_ci.append(high)
        bin_n.append(n)

    return (
        np.asarray(mean_pred),
        np.asarray(observed),
        np.asarray(lower_ci),
        np.asarray(upper_ci),
        np.asarray(bin_n),
    )


def calibration_intercept_slope(
    y_true,
    y_prob,
    max_iter=100,
    tolerance=1e-9,
):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    eps = 1e-6
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    logit_probability = np.log(y_prob / (1.0 - y_prob))
    X = np.column_stack(
        [
            np.ones_like(logit_probability),
            logit_probability,
        ]
    )

    beta = np.array([0.0, 1.0], dtype=float)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        predicted = 1.0 / (1.0 + np.exp(-eta))
        weights = predicted * (1.0 - predicted)
        gradient = X.T @ (y_true - predicted)
        information = X.T @ (X * weights[:, None])

        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient

        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tolerance:
            beta = beta_new
            break
        beta = beta_new

    eta = np.clip(X @ beta, -30, 30)
    predicted = 1.0 / (1.0 + np.exp(-eta))
    weights = predicted * (1.0 - predicted)
    information = X.T @ (X * weights[:, None])
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.diag(covariance))

    intercept = float(beta[0])
    slope = float(beta[1])
    intercept_se = float(standard_errors[0])
    slope_se = float(standard_errors[1])

    return {
        "intercept": intercept,
        "intercept_ci_lower": intercept - 1.96 * intercept_se,
        "intercept_ci_upper": intercept + 1.96 * intercept_se,
        "slope": slope,
        "slope_ci_lower": slope - 1.96 * slope_se,
        "slope_ci_upper": slope + 1.96 * slope_se,
    }


def calculate_net_benefit(y_true, probabilities, thresholds):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(y_true)
    net_benefit = []

    for threshold in thresholds:
        predicted_positive = probabilities >= threshold
        tp = int(np.sum(predicted_positive & (y_true == 1)))
        fp = int(np.sum(predicted_positive & (y_true == 0)))
        odds = threshold / (1.0 - threshold)
        net_benefit.append(tp / n - fp / n * odds)

    return np.asarray(net_benefit, dtype=float)


def evaluate_full_model(label_df, full_pred_df, analysis_name):
    merged = (
        label_df
        .merge(full_pred_df, on="id", how="inner")
        .sort_values("id")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise RuntimeError(f"{analysis_name}: no prediction IDs matched labels.")

    y = merged["label"].to_numpy(dtype=int)
    p = merged["full_probability"].to_numpy(dtype=float)
    if np.unique(y).size != 2:
        raise RuntimeError(f"{analysis_name}: both classes are required.")

    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p))
    ci_low, ci_high = bootstrap_auc_ci(y, p)

    metrics = {
        "analysis": analysis_name,
        "model": "Physiology-informed model",
        "n": int(len(y)),
        "n_SICM": int((y == 1).sum()),
        "n_sepsis_only": int((y == 0).sum()),
        "auc": auc,
        "auc_95ci_lower": ci_low,
        "auc_95ci_upper": ci_high,
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": THRESHOLD,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return merged, metrics


for path in [
    PRIMARY_LABEL_CSV,
    LABEL_020_CSV,
    LABEL_GRAY_EXCLUDED_CSV,
    FULL_PRED_CSV,
    BASELINE_PRED_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find:\n{path}")

primary_labels = load_labels(PRIMARY_LABEL_CSV)
labels_020 = load_labels(LABEL_020_CSV)
labels_gray = load_labels(LABEL_GRAY_EXCLUDED_CSV)
full_predictions = load_predictions(FULL_PRED_CSV, "full_probability")
baseline_predictions = load_predictions(
    BASELINE_PRED_CSV,
    "baseline_probability",
)


full_external, full_metrics = evaluate_full_model(
    primary_labels,
    full_predictions,
    "primary_ctnt_gt_0.10",
)
full_external.to_csv(
    EVALUATION_DIR / "full_model_predictions_with_labels.csv",
    index=False,
    encoding="utf-8-sig",
)

full_y = full_external["label"].to_numpy(dtype=int)
full_prob = full_external["full_probability"].to_numpy(dtype=float)


paired_external = (
    primary_labels
    .merge(full_predictions, on="id", how="inner")
    .merge(baseline_predictions, on="id", how="inner")
    .sort_values("id")
    .reset_index(drop=True)
)
if paired_external.empty:
    raise RuntimeError("No paired external predictions are available.")
if paired_external["label"].nunique() != 2:
    raise RuntimeError("Both classes are required in the paired external cohort.")

paired_external.to_csv(
    EVALUATION_DIR / "paired_primary_external_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

if len(full_external) != len(paired_external):
    print(
        "WARNING: full-model and paired external cohorts differ in size. "
        "Primary full-model metrics/calibration use the full-model cohort; "
        "DeLong/DCA use only paired cases."
    )

paired_y = paired_external["label"].to_numpy(dtype=int)
paired_full_prob = paired_external["full_probability"].to_numpy(dtype=float)
paired_baseline_prob = paired_external["baseline_probability"].to_numpy(dtype=float)

paired_full_auc = float(roc_auc_score(paired_y, paired_full_prob))
paired_baseline_auc = float(roc_auc_score(paired_y, paired_baseline_prob))
paired_full_ci = bootstrap_auc_ci(paired_y, paired_full_prob)
paired_baseline_ci = bootstrap_auc_ci(paired_y, paired_baseline_prob)

delong_result = paired_delong_test(
    paired_y,
    paired_full_prob,
    paired_baseline_prob,
)

pd.DataFrame(
    [
        {
            "Full model AUROC": paired_full_auc,
            "EchoJEPA baseline AUROC": paired_baseline_auc,
            "Delta AUROC": delong_result["delta_auc"],
            "Delta AUROC 95% CI lower": delong_result["delta_ci_lower"],
            "Delta AUROC 95% CI upper": delong_result["delta_ci_upper"],
            "DeLong P value": delong_result["p_value"],
            "Paired N": len(paired_y),
        }
    ]
).to_csv(
    EVALUATION_DIR / "Figure4_DeLong_results.csv",
    index=False,
    encoding="utf-8-sig",
)


(
    calibration_pred,
    calibration_obs,
    calibration_low,
    calibration_high,
    calibration_n,
) = calibration_with_ci(
    full_y,
    full_prob,
    n_bins=CALIBRATION_BINS,
)

pd.DataFrame(
    {
        "mean_predicted_probability": calibration_pred,
        "observed_event_fraction": calibration_obs,
        "observed_fraction_95ci_lower": calibration_low,
        "observed_fraction_95ci_upper": calibration_high,
        "n": calibration_n,
    }
).to_csv(
    EVALUATION_DIR / "external_calibration_curve.csv",
    index=False,
    encoding="utf-8-sig",
)

calibration_stats = calibration_intercept_slope(
    full_y,
    full_prob,
)

pd.DataFrame(
    [
        {
            "Metric": "Brier score",
            "Estimate": full_metrics["brier"],
            "95% CI lower": np.nan,
            "95% CI upper": np.nan,
        },
        {
            "Metric": "Calibration intercept",
            "Estimate": calibration_stats["intercept"],
            "95% CI lower": calibration_stats["intercept_ci_lower"],
            "95% CI upper": calibration_stats["intercept_ci_upper"],
        },
        {
            "Metric": "Calibration slope",
            "Estimate": calibration_stats["slope"],
            "95% CI lower": calibration_stats["slope_ci_lower"],
            "95% CI upper": calibration_stats["slope_ci_upper"],
        },
    ]
).to_csv(
    EVALUATION_DIR / "Supplementary_calibration_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
ax.fill_between(
    calibration_pred,
    calibration_low,
    calibration_high,
    alpha=0.20,
    linewidth=0,
    label="Pointwise 95% Wilson CI",
)
ax.plot(
    calibration_pred,
    calibration_obs,
    marker="o",
    linewidth=1.5,
    label=f"Physiology-informed model (Brier={full_metrics['brier']:.3f})",
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed proportion")
ax.set_title("External calibration")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(EVALUATION_DIR / "external_calibration_curve.png", dpi=600)
fig.savefig(EVALUATION_DIR / "external_calibration_curve.pdf")
plt.close(fig)


thresholds = np.linspace(
    DCA_THRESHOLD_MIN,
    DCA_THRESHOLD_MAX,
    DCA_POINTS,
)
full_net_benefit = calculate_net_benefit(
    paired_y,
    paired_full_prob,
    thresholds,
)
baseline_net_benefit = calculate_net_benefit(
    paired_y,
    paired_baseline_prob,
    thresholds,
)
prevalence = float(np.mean(paired_y))
treat_all = (
    prevalence
    - (1.0 - prevalence)
    * thresholds
    / (1.0 - thresholds)
)
treat_none = np.zeros_like(thresholds)

pd.DataFrame(
    {
        "threshold_probability": thresholds,
        "physiology_informed_net_benefit": full_net_benefit,
        "echojepa_baseline_net_benefit": baseline_net_benefit,
        "treat_all_net_benefit": treat_all,
        "treat_none_net_benefit": treat_none,
    }
).to_csv(
    EVALUATION_DIR / "external_decision_curve.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(thresholds, full_net_benefit, label="Physiology-informed model")
ax.plot(thresholds, baseline_net_benefit, label="EchoJEPA baseline")
ax.plot(thresholds, treat_all, linestyle="--", label="Treat all")
ax.plot(thresholds, treat_none, linestyle=":", label="Treat none")
ax.set_xlim(DCA_THRESHOLD_MIN, DCA_THRESHOLD_MAX)
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_title("External decision curve analysis")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(EVALUATION_DIR / "external_decision_curve.png", dpi=600)
fig.savefig(EVALUATION_DIR / "external_decision_curve.pdf")
plt.close(fig)


full_fpr, full_tpr, _ = roc_curve(paired_y, paired_full_prob)
baseline_fpr, baseline_tpr, _ = roc_curve(paired_y, paired_baseline_prob)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(
    full_fpr,
    full_tpr,
    linewidth=1.5,
    label=f"Physiology-informed model (AUC={paired_full_auc:.3f})",
)
ax.plot(
    baseline_fpr,
    baseline_tpr,
    linewidth=1.5,
    label=f"EchoJEPA baseline (AUC={paired_baseline_auc:.3f})",
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


sensitivity_020, metrics_020 = evaluate_full_model(
    labels_020,
    full_predictions,
    "sensitivity_ctnt_gt_0.20",
)
sensitivity_gray, metrics_gray = evaluate_full_model(
    labels_gray,
    full_predictions,
    "sensitivity_exclude_0.08_to_0.12",
)

sensitivity_020.to_csv(
    EVALUATION_DIR / "sensitivity_ctnt_gt_0.20_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)
sensitivity_gray.to_csv(
    EVALUATION_DIR / "sensitivity_exclude_0.08_to_0.12_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

baseline_metrics = {
    "analysis": "primary_ctnt_gt_0.10",
    "model": "EchoJEPA baseline",
    "n": int(len(paired_y)),
    "n_SICM": int((paired_y == 1).sum()),
    "n_sepsis_only": int((paired_y == 0).sum()),
    "auc": paired_baseline_auc,
    "auc_95ci_lower": paired_baseline_ci[0],
    "auc_95ci_upper": paired_baseline_ci[1],
    "accuracy": np.nan,
    "sensitivity": np.nan,
    "specificity": np.nan,
    "f1": np.nan,
    "brier": np.nan,
    "threshold": np.nan,
    "tp": np.nan,
    "tn": np.nan,
    "fp": np.nan,
    "fn": np.nan,
}

pd.DataFrame(
    [
        full_metrics,
        baseline_metrics,
        metrics_020,
        metrics_gray,
    ]
).to_csv(
    EVALUATION_DIR / "external_model_performance.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 80)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 80)

for metrics in [full_metrics, metrics_020, metrics_gray]:
    print(f"\n{metrics['analysis']} | {metrics['model']}")
    print(f"N           : {metrics['n']}")
    print(f"SICM/control: {metrics['n_SICM']}/{metrics['n_sepsis_only']}")
    print(
        f"AUC         : {metrics['auc']:.4f} "
        f"(95% CI {metrics['auc_95ci_lower']:.4f}-"
        f"{metrics['auc_95ci_upper']:.4f})"
    )
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"Specificity : {metrics['specificity']:.4f}")
    print(f"F1          : {metrics['f1']:.4f}")
    print(f"Brier       : {metrics['brier']:.4f}")
    print(
        f"TP/TN/FP/FN : "
        f"{metrics['tp']}/{metrics['tn']}/"
        f"{metrics['fp']}/{metrics['fn']}"
    )

print("\nPAIRED EXTERNAL MODEL COMPARISON")
print(f"N                : {len(paired_y)}")
print(
    f"Full model AUC   : {paired_full_auc:.4f} "
    f"(95% CI {paired_full_ci[0]:.4f}-{paired_full_ci[1]:.4f})"
)
print(
    f"EchoJEPA AUC     : {paired_baseline_auc:.4f} "
    f"(95% CI {paired_baseline_ci[0]:.4f}-{paired_baseline_ci[1]:.4f})"
)
print(
    f"Delta AUC        : {delong_result['delta_auc']:.4f} "
    f"(95% CI {delong_result['delta_ci_lower']:.4f}-"
    f"{delong_result['delta_ci_upper']:.4f})"
)
print(f"DeLong P         : {delong_result['p_value']:.6g}")

print("\nCALIBRATION")
print(f"Brier            : {full_metrics['brier']:.4f}")
print(
    f"Intercept        : {calibration_stats['intercept']:.4f} "
    f"(95% CI {calibration_stats['intercept_ci_lower']:.4f}-"
    f"{calibration_stats['intercept_ci_upper']:.4f})"
)
print(
    f"Slope            : {calibration_stats['slope']:.4f} "
    f"(95% CI {calibration_stats['slope_ci_lower']:.4f}-"
    f"{calibration_stats['slope_ci_upper']:.4f})"
)

print("\nDCA")
print(
    f"Threshold range  : {DCA_THRESHOLD_MIN:.2f}-"
    f"{DCA_THRESHOLD_MAX:.2f} ({DCA_POINTS} equally spaced points)"
)
print(f"Paired prevalence: {prevalence:.4f}")

print("\nOutputs:", EVALUATION_DIR)
