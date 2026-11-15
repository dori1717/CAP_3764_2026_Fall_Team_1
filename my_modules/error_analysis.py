"""Error analysis for the final-grade (G3) prediction model.

Input: a table of test-set predictions with the columns ``y_true`` and ``y_pred``
(optionally also feature columns such as G1, G2, absences, failures).

Residual convention: ``residual = y_true - y_pred``
(positive = the model under-predicts the grade, negative = it over-predicts).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("y_true", "y_pred")
FEATURES_TO_CHECK = ("G1", "G2", "absences", "failures")


def load_predictions(path):
    """Read the predictions CSV and check that it has the required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the modeling notebook first so it saves the "
            "test predictions (columns y_true, y_pred) to this file."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")
    return df


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted Final Grade (G3)",
                             ax=None, save_path=None, jitter=0.15, seed=42, annotate_top=0):
    """Scatter plot of actual vs. predicted grades with a y = x reference line.

    G3 is an integer (0-20), so points stack on top of each other; a small
    jitter on the x-axis and transparency keep every student visible.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(y_true + rng.uniform(-jitter, jitter, len(y_true)), y_pred,
               alpha=0.5, s=28, edgecolor="none", label="Students (test set)")
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction (y = x)")
    # Optionally label the students with the largest absolute errors
    for i in np.argsort(-np.abs(y_true - y_pred))[:annotate_top]:
        ax.annotate(f"actual {y_true[i]:.0f}, predicted {y_pred[i]:.1f}",
                    xy=(y_true[i], y_pred[i]), xytext=(y_true[i] + 2, y_pred[i] + 1.5),
                    arrowprops={"arrowstyle": "->", "color": "gray"}, fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual G3")
    ax.set_ylabel("Predicted G3")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


def add_errors(df, y_true_col="y_true", y_pred_col="y_pred"):
    """Return a copy of df with ``residual`` and ``abs_error`` columns."""
    out = df.copy()
    out["residual"] = out[y_true_col] - out[y_pred_col]
    out["abs_error"] = out["residual"].abs()
    return out


def overall_metrics(df, y_true_col="y_true", y_pred_col="y_pred"):
    """MAE, RMSE and R^2 on the predictions table."""
    d = add_errors(df, y_true_col, y_pred_col)
    ss_res = (d["residual"] ** 2).sum()
    ss_tot = ((d[y_true_col] - d[y_true_col].mean()) ** 2).sum()
    return {
        "MAE": d["abs_error"].mean(),
        "RMSE": float(np.sqrt((d["residual"] ** 2).mean())),
        "R2": 1 - ss_res / ss_tot,
    }


def error_by_group(df, y_true_col="y_true", y_pred_col="y_pred"):
    """MAE for (a) students with G3 = 0 vs. the rest and (b) by actual-grade bin."""
    d = add_errors(df, y_true_col, y_pred_col)
    d["group"] = np.where(d[y_true_col] == 0, "G3 = 0", "G3 > 0")
    by_zero = d.groupby("group")["abs_error"].agg(n="count", MAE="mean")

    bins = [-0.1, 0, 7, 10, 13, 16, 20]
    labels = ["0", "1-7", "8-10", "11-13", "14-16", "17-20"]
    d["grade_bin"] = pd.cut(d[y_true_col], bins=bins, labels=labels)
    by_bin = d.groupby("grade_bin", observed=True)["abs_error"].agg(n="count", MAE="mean")
    return by_zero, by_bin


def error_by_feature(df, feature, y_true_col="y_true", y_pred_col="y_pred", q=4):
    """MAE by feature level (quartile bins for numeric features with many values)."""
    d = add_errors(df, y_true_col, y_pred_col)
    col = d[feature]
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > 6:
        col = pd.qcut(col, q=q, duplicates="drop")
    return d.groupby(col, observed=True)["abs_error"].agg(n="count", MAE="mean")


def top_errors(df, n=10, y_true_col="y_true", y_pred_col="y_pred"):
    """The n rows with the largest absolute error."""
    d = add_errors(df, y_true_col, y_pred_col)
    return d.sort_values("abs_error", ascending=False).head(n)


def error_report(df, y_true_col="y_true", y_pred_col="y_pred", features=FEATURES_TO_CHECK):
    """Plain-text summary with the numbers needed to explain the biggest errors."""
    lines = []
    m = overall_metrics(df, y_true_col, y_pred_col)
    lines.append(f"Overall: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")

    by_zero, by_bin = error_by_group(df, y_true_col, y_pred_col)
    lines += ["", "MAE: G3 = 0 vs. the rest", by_zero.round(2).to_string()]
    lines += ["", "MAE by actual-grade bin", by_bin.round(2).to_string()]

    for f in features:
        if f in df.columns:
            lines += ["", f"MAE by {f}",
                      error_by_feature(df, f, y_true_col, y_pred_col).round(2).to_string()]

    d = add_errors(df, y_true_col, y_pred_col)
    zero = d[y_true_col] == 0
    if zero.any():
        share_students = zero.mean()
        share_error = d.loc[zero, "abs_error"].sum() / d["abs_error"].sum()
        lines += ["", f"G3 = 0 rows are {share_students:.1%} of the test students "
                      f"but {share_error:.1%} of the total absolute error."]
    return "\n".join(lines)
