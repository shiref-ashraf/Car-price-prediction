"""
evaluate.py — Evaluation metrics, plots, and feature-importance analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

from config import SEED, OUTPUT_DIR, METRICS_FILE


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics in original USD space.

    Returns
    -------
    dict with keys: MSE, RMSE, NRMSE, MAE, MAPE, R2
    """
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    nrmse = rmse / (y_true.max() - y_true.min())

    metrics = {
        "MSE":   round(float(mse),   2),
        "RMSE":  round(float(rmse),  2),
        "NRMSE": round(float(nrmse), 6),
        "MAE":   round(float(mae),   2),
        "MAPE":  round(float(mape),  4),
        "R2":    round(float(r2),    6),
    }
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Metrics") -> None:
    """Pretty-print a metrics dictionary."""
    print("=" * 52)
    print(f"  {title}")
    print("=" * 52)
    print(f"  MSE   : ${metrics['MSE']:>18,.2f}")
    print(f"  RMSE  : ${metrics['RMSE']:>18,.2f}")
    print(f"  NRMSE :  {metrics['NRMSE']:>18.4f}  (RMSE / price range)")
    print(f"  MAE   : ${metrics['MAE']:>18,.2f}")
    print(f"  MAPE  :  {metrics['MAPE']:>17.2f}%")
    print(f"  R²    :  {metrics['R2']:>18.4f}")
    print("=" * 52)


def save_metrics(metrics: dict, path: str = METRICS_FILE) -> None:
    """Persist metrics dict to JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_training_history(history, save_path: str = None) -> None:
    """Plot loss and MAE curves for train / validation splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train MSE (log-space)")
    axes[0].plot(history.history["val_loss"], label="Val MSE (log-space)")
    axes[0].set_title("Loss (MSE on log-price)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].legend()

    axes[1].plot(history.history["mae"],     label="Train MAE (log-space)")
    axes[1].plot(history.history["val_mae"], label="Val MAE (log-space)")
    axes[1].set_title("MAE (log-price space)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE"); axes[1].legend()

    plt.tight_layout()
    _save_or_show(fig, save_path or f"{OUTPUT_DIR}/training_history.png")


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                        save_path: str = None) -> None:
    """Scatter plot of predicted vs. actual prices, plus residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=15,
                    color="steelblue", edgecolors="none")
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Price (USD)")
    axes[0].set_ylabel("Predicted Price (USD)")
    axes[0].set_title("Predicted vs. Actual")
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (Actual − Predicted, USD)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    _save_or_show(fig, save_path or f"{OUTPUT_DIR}/pred_vs_actual.png")


def plot_feature_importance(model, X_test: pd.DataFrame,
                            y_test_log: np.ndarray,
                            top_n: int = 15,
                            save_path: str = None) -> pd.DataFrame:
    """
    Permutation importance proxy for a Keras model.
    Returns the importance DataFrame.
    """
    class _KerasWrapper(BaseEstimator):
        def __init__(self, km): self.km = km
        def fit(self, X, y): return self
        def predict(self, X): return self.km.predict(X).flatten()

    perm = permutation_importance(
        _KerasWrapper(model), X_test.values, y_test_log,
        n_repeats=10, random_state=SEED, scoring="r2",
    )
    imp_df = (pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm.importances_mean,
    })
    .sort_values("Importance", ascending=False)
    .head(top_n))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="steelblue")
    ax.set_xlabel("Permutation Importance (drop in R²)")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    _save_or_show(fig, save_path or f"{OUTPUT_DIR}/feature_importance.png")

    return imp_df


# ── Bias–Variance diagnostic ──────────────────────────────────────────────────
def bias_variance_report(train_metrics: dict, test_metrics: dict) -> None:
    """Print a simple bias–variance diagnostic table."""
    print("\n[evaluate] Bias–Variance Diagnostic")
    print(f"  Train R²  : {train_metrics['R2']:.4f}")
    print(f"  Test  R²  : {test_metrics['R2']:.4f}")
    gap = train_metrics["R2"] - test_metrics["R2"]
    print(f"  R² Gap    : {gap:.4f}")
    if test_metrics["R2"] < 0.80:
        print("  → High Bias  (underfitting): consider deeper / wider model.")
    elif gap > 0.10:
        print("  → High Variance (overfitting): increase dropout / L2, reduce model size.")
    else:
        print("  → Good Bias–Variance Balance.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_or_show(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Plot saved → {path}")
