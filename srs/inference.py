"""
inference.py — Load trained model and run predictions on new / unseen data.

Usage
─────
    # Single car (dict)
    python inference.py

    # Batch CSV
    python inference.py --csv path/to/new_cars.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import (
    MODEL_WEIGHTS_FILE, SCALER_FILE, ENCODER_FILE,
    HIDDEN_UNITS, DROPOUT_RATES, ACTIVATION, OUTPUT_ACT,
    SCALE_COLS, ORDINAL_COL, TARGET_ENC_COL,
)


# ── Load artefacts ────────────────────────────────────────────────────────────
def load_model(input_dim: int) -> keras.Model:
    """Re-create model architecture and load saved weights."""
    from model import build_model
    model = build_model(input_dim)
    model.load_weights(MODEL_WEIGHTS_FILE)
    print(f"[inference] Weights loaded from {MODEL_WEIGHTS_FILE}")
    return model


def load_artefacts() -> dict:
    """Load scaler and encoder metadata."""
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        meta = pickle.load(f)
    return {"scaler": scaler, **meta}


# ── Feature engineering (mirrors preprocessing) ───────────────────────────────
def preprocess_single(car: dict, artefacts: dict) -> np.ndarray:
    """
    Transform a single car dict into a model-ready numpy row.

    Required keys
    ─────────────
      Brand, Model, Year, Engine_CC, Power_BHP,
      Fuel_Type, Transmission, Owner_Type, KM_Driven
    """
    df = pd.DataFrame([car])

    # Feature engineering
    current_year = 2024
    df["Vehicle_Age"] = current_year - df["Year"]
    df["KM_per_Year"] = df["KM_Driven"] / (df["Vehicle_Age"] + 1)
    df.drop(columns=["Year"], inplace=True)

    # Ordinal encoding
    oe = artefacts["ordinal_encoder"]
    df[ORDINAL_COL] = oe.transform(df[[ORDINAL_COL]]).astype(int)

    # One-Hot encoding (align columns to training schema)
    ohe_cols = ["Brand", "Fuel_Type", "Transmission"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Target encoding for Model
    model_val = car[TARGET_ENC_COL]
    df["Model_enc"] = artefacts["model_mean_map"].get(model_val,
                                                       artefacts["global_mean"])
    df.drop(columns=[TARGET_ENC_COL], errors="ignore", inplace=True)

    # Align to training columns (fill missing OHE cols with 0)
    train_cols = artefacts["ohe_columns"]
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_cols]

    # Scale
    df_f = df.copy().astype(float)
    df_f[SCALE_COLS] = artefacts["scaler"].transform(df_f[SCALE_COLS])

    return df_f.values.astype(np.float32)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_price(car: dict, model: keras.Model, artefacts: dict) -> float:
    """Return predicted price in USD for a single car dict."""
    X = preprocess_single(car, artefacts)
    log_pred = model.predict(X, verbose=0).flatten()[0]
    return float(np.expm1(log_pred))


def predict_batch(df_raw: pd.DataFrame,
                  model: keras.Model, artefacts: dict) -> np.ndarray:
    """Return predicted prices (USD) for a DataFrame of raw cars."""
    rows = [preprocess_single(row, artefacts) for row in df_raw.to_dict("records")]
    X = np.vstack(rows)
    log_preds = model.predict(X, verbose=0).flatten()
    return np.expm1(log_preds)


# ── Demo ──────────────────────────────────────────────────────────────────────
DEMO_CARS = [
    {
        "Brand": "Toyota", "Model": "Camry", "Year": 2020,
        "Engine_CC": 2500, "Power_BHP": 180.0, "Fuel_Type": "Petrol",
        "Transmission": "Automatic", "Owner_Type": "First", "KM_Driven": 35000,
    },
    {
        "Brand": "BMW", "Model": "M3", "Year": 2018,
        "Engine_CC": 3000, "Power_BHP": 430.0, "Fuel_Type": "Petrol",
        "Transmission": "Automatic", "Owner_Type": "Second", "KM_Driven": 62000,
    },
    {
        "Brand": "Ford", "Model": "Focus", "Year": 2014,
        "Engine_CC": 1600, "Power_BHP": 120.0, "Fuel_Type": "Diesel",
        "Transmission": "Manual", "Owner_Type": "Third", "KM_Driven": 140000,
    },
    {
        "Brand": "Tesla", "Model": "Model S", "Year": 2022,
        "Engine_CC": 0, "Power_BHP": 670.0, "Fuel_Type": "Electric",
        "Transmission": "Automatic", "Owner_Type": "First", "KM_Driven": 12000,
    },
]


def run_demo() -> None:
    artefacts = load_artefacts()
    input_dim = len(artefacts["ohe_columns"])
    model = load_model(input_dim)

    print("\n" + "=" * 60)
    print("  Car Price Prediction — Live Demo (Unseen Vehicles)")
    print("=" * 60)
    print(f"{'#':<4} {'Car':<30} {'Predicted Price (USD)':>22}")
    print("-" * 60)
    for i, car in enumerate(DEMO_CARS, 1):
        label = f"{car['Brand']} {car['Model']} ({car['Year']})"
        pred  = predict_price(car, model, artefacts)
        print(f"{i:<4} {label:<30} ${pred:>20,.2f}")
    print("=" * 60)


def run_csv(csv_path: str) -> None:
    df_raw = pd.read_csv(csv_path)
    artefacts = load_artefacts()
    input_dim = len(artefacts["ohe_columns"])
    model = load_model(input_dim)

    preds = predict_batch(df_raw, model, artefacts)
    df_raw["Predicted_Price_USD"] = np.round(preds, 2)
    out_path = csv_path.replace(".csv", "_predictions.csv")
    df_raw.to_csv(out_path, index=False)
    print(f"[inference] Predictions saved → {out_path}")
    print(df_raw[["Brand", "Model", "Year", "Predicted_Price_USD"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car price inference demo.")
    parser.add_argument("--csv", default=None,
                        help="Optional: path to a CSV of unseen cars.")
    args = parser.parse_args()

    if args.csv:
        run_csv(args.csv)
    else:
        run_demo()
