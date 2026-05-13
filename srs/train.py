"""
train.py — End-to-end training script.

Usage
─────
    python train.py
    python train.py --data path/to/car_prices_dataset.csv
"""

import argparse
import numpy as np

from data_preprocessing import build_datasets
from model import build_model, train_model
from evaluate import (
    compute_metrics, print_metrics, save_metrics,
    plot_training_history, plot_pred_vs_actual,
    plot_feature_importance, bias_variance_report,
)
from config import DATA_PATH


def main(data_path: str = DATA_PATH) -> None:
    print("\n" + "=" * 60)
    print("  Car Price Prediction — Deep Learning Training Pipeline")
    print("=" * 60)

    # ── 1. Preprocessing ──────────────────────────────────────────
    print("\n[STEP 1] Preprocessing …")
    (X_train, X_test,
     y_train_log, y_test_log,
     y_train_orig, y_test_orig,
     scaler, model_mean_map, global_mean) = build_datasets(data_path)

    X_train_np = X_train.values.astype(np.float32)
    X_test_np  = X_test.values.astype(np.float32)
    y_train_np = y_train_log.values.astype(np.float32)
    y_test_np  = y_test_log.values.astype(np.float32)

    # ── 2. Build Model ────────────────────────────────────────────
    print("\n[STEP 2] Building model …")
    model = build_model(input_dim=X_train_np.shape[1])
    model.summary()

    # ── 3. Train ──────────────────────────────────────────────────
    print("\n[STEP 3] Training …")
    history = train_model(model, X_train_np, y_train_np)
    plot_training_history(history)

    # ── 4. Evaluate ───────────────────────────────────────────────
    print("\n[STEP 4] Evaluating …")

    # Test set
    y_pred_log  = model.predict(X_test_np).flatten()
    y_pred_orig = np.expm1(y_pred_log)
    y_test_arr  = y_test_orig.values

    test_metrics = compute_metrics(y_test_arr, y_pred_orig)
    print_metrics(test_metrics, "Test Set Evaluation")
    save_metrics(test_metrics)

    # Train set (for bias–variance)
    y_train_pred_log  = model.predict(X_train_np).flatten()
    y_train_pred_orig = np.expm1(y_train_pred_log)
    train_metrics = compute_metrics(y_train_orig.values, y_train_pred_orig)
    print_metrics(train_metrics, "Train Set Evaluation (Bias–Variance Check)")

    bias_variance_report(train_metrics, test_metrics)

    # ── 5. Plots ──────────────────────────────────────────────────
    print("\n[STEP 5] Generating plots …")
    plot_pred_vs_actual(y_test_arr, y_pred_orig)
    plot_feature_importance(model, X_test, y_test_np)

    print("\n" + "=" * 60)
    print("  Training complete. All outputs saved to /outputs/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car price DL model.")
    parser.add_argument("--data", default=DATA_PATH,
                        help="Path to car_prices_dataset.csv")
    args = parser.parse_args()
    main(args.data)
