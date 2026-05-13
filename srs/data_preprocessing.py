"""
data_preprocessing.py — All data loading, feature engineering, encoding,
splitting, and scaling. Returns ready-to-train numpy arrays.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from config import (
    SEED, DATA_PATH, TEST_SIZE, CLIP_PERCENTILE,
    TARGET_COL, YEAR_COL, OHE_COLS, ORDINAL_COL, ORDINAL_CATS,
    TARGET_ENC_COL, SCALE_COLS,
    SCALER_FILE, ENCODER_FILE
)


# ── 1. Load ────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols from {path}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Vehicle_Age and KM_per_Year; drop raw Year column."""
    df = df.copy()
    current_year = 2024
    df["Vehicle_Age"] = current_year - df[YEAR_COL]
    df["KM_per_Year"] = df["KM_Driven"] / (df["Vehicle_Age"] + 1)
    df.drop(columns=[YEAR_COL], inplace=True)
    print("[engineer_features] Added: Vehicle_Age, KM_per_Year")
    return df


# ── 3. Outlier Clipping ───────────────────────────────────────────────────────
def clip_outliers(df: pd.DataFrame,
                cols: list[str],
                lo_pct: float = CLIP_PERCENTILE[0],
                hi_pct: float = CLIP_PERCENTILE[1]) -> pd.DataFrame:
    """Clip each column to the given percentile range (in-place copy)."""
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(lo_pct)
        hi = df[col].quantile(hi_pct)
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        print(f"  [{col}] clipped {n_clipped} rows → [{lo:.1f}, {hi:.1f}]")
    return df


# ── 4. Encode ─────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    One-Hot encode Brand / Fuel_Type / Transmission (drop_first=True).
    Ordinal-encode Owner_Type (Third=0, Second=1, First=2).
    Returns transformed df and the fitted OrdinalEncoder.
    """
    df = df.copy()

    # One-Hot
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)
    print(f"[encode] OHE columns: {OHE_COLS}")

    # Ordinal
    oe = OrdinalEncoder(categories=[ORDINAL_CATS])
    df[ORDINAL_COL] = oe.fit_transform(df[[ORDINAL_COL]]).astype(int)
    print(f"[encode] Ordinal: {ORDINAL_COL} → {dict(zip(ORDINAL_CATS, range(len(ORDINAL_CATS))))}")
    return df, oe


# ── 5. Train / Val / Test split with leakage-free target encoding ─────────────
def split_and_target_encode(
    df: pd.DataFrame,
    y_log: pd.Series,
    model_col: pd.Series,
    y_orig: pd.Series,
) -> tuple:
    """
    1. Split into train / test (80/20).
    2. Target-encode Model on *train mean only* to avoid data leakage.
    Returns (X_train, X_test, y_train_log, y_test_log,
            y_train_orig, y_test_orig, model_mean_map, global_mean)
    """
    X_raw = df.drop(columns=[TARGET_ENC_COL], errors="ignore")

    (X_train_r, X_test_r,
    y_train_log, y_test_log,
    model_train, model_test,
    y_train_orig, y_test_orig) = train_test_split(
        X_raw, y_log, model_col, y_orig,
        test_size=TEST_SIZE, random_state=SEED
    )

    # Target encoding — compute on train split only
    train_df_for_enc = model_train.to_frame().join(y_train_orig.rename("Price_USD"))
    model_mean_map = train_df_for_enc.groupby(TARGET_ENC_COL)["Price_USD"].mean()
    global_mean = y_train_orig.mean()

    X_train_r = X_train_r.copy()
    X_test_r  = X_test_r.copy()
    X_train_r["Model_enc"] = model_train.map(model_mean_map).fillna(global_mean)
    X_test_r["Model_enc"]  = model_test.map(model_mean_map).fillna(global_mean)

    print(f"[split] Train: {X_train_r.shape}  Test: {X_test_r.shape}")
    return (X_train_r, X_test_r,
            y_train_log, y_test_log,
            y_train_orig, y_test_orig,
            model_mean_map, global_mean)


# ── 6. Scale ──────────────────────────────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train; transform both splits. Optionally pickle."""
    X_train = X_train.copy().astype(float)
    X_test  = X_test.copy().astype(float)

    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

    if save:
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[scale] Scaler saved → {SCALER_FILE}")

    print(f"[scale] Scaled columns: {SCALE_COLS}")
    return X_train, X_test, scaler


# ── 7. Master pipeline ────────────────────────────────────────────────────────
def build_datasets(path: str = DATA_PATH):
    """
    End-to-end preprocessing pipeline.
    Returns:
        X_train, X_test            : pd.DataFrames (ready to convert to numpy)
        y_train_log, y_test_log    : pd.Series  (log-space targets)
        y_train_orig, y_test_orig  : pd.Series  (original USD targets)
        scaler                     : fitted StandardScaler
        model_mean_map             : pd.Series  (for inference target encoding)
        global_mean                : float
    """
    df = load_data(path)

    # Log-transform target
    df["log_Price"] = np.log1p(df[TARGET_COL])
    y_log  = df["log_Price"]
    y_orig = df[TARGET_COL]
    model_col = df[TARGET_ENC_COL]

    # Feature engineering
    df_feat = engineer_features(df.drop(columns=[TARGET_COL, "log_Price"]))

    # Clip outliers
    clip_cols = ["Engine_CC", "Power_BHP", "KM_Driven", "KM_per_Year"]
    # KM_per_Year is added by engineer_features
    clip_cols_present = [c for c in clip_cols if c in df_feat.columns]
    df_feat = clip_outliers(df_feat, clip_cols_present)

    # Encode
    df_enc, oe = encode_categoricals(df_feat)

    # Split + target encode
    (X_train_r, X_test_r,
    y_train_log, y_test_log,
    y_train_orig, y_test_orig,
    model_mean_map, global_mean) = split_and_target_encode(
        df_enc, y_log, model_col, y_orig
    )

    # Scale
    X_train, X_test, scaler = scale_features(X_train_r, X_test_r)

    # Persist encoder meta for inference
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump({
            "ordinal_encoder": oe,
            "model_mean_map":  model_mean_map,
            "global_mean":     global_mean,
            "ohe_columns":     list(X_train.columns),
        }, f)
    print(f"[pipeline] Encoder meta saved → {ENCODER_FILE}")

    return (X_train, X_test,
            y_train_log, y_test_log,
            y_train_orig, y_test_orig,
            scaler, model_mean_map, global_mean)


if __name__ == "__main__":
    (X_tr, X_te, ytr_l, yte_l, ytr_o, yte_o, sc, mmm, gm) = build_datasets()
    print("\nPreprocessing complete.")
    print(f"X_train: {X_tr.shape}  X_test: {X_te.shape}")
    print(f"Features: {list(X_tr.columns)}")