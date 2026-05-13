"""
config.py — Central configuration for all hyperparameters and paths.
"""

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "car_prices_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR= os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
TEST_SIZE       = 0.20   # 20 % hold-out test set
VAL_SPLIT       = 0.15   # 15 % of train → validation (inside Keras)
CLIP_PERCENTILE = (0.01, 0.99)   # outlier clipping bounds

# Columns
TARGET_COL   = "Price_USD"
YEAR_COL     = "Year"
OHE_COLS     = ["Brand", "Fuel_Type", "Transmission"]
ORDINAL_COL  = "Owner_Type"
ORDINAL_CATS = ["Third", "Second", "First"]   # lowest → highest
TARGET_ENC_COL = "Model"
SCALE_COLS   = ["Engine_CC", "Power_BHP", "KM_Driven",
                "Vehicle_Age", "KM_per_Year", "Model_enc"]

# ── Model ─────────────────────────────────────────────────────────────────────
HIDDEN_UNITS  = [128, 64, 32]
DROPOUT_RATES = [0.15, 0.10, 0.0]   # per layer (0 = no dropout on last hidden)
ACTIVATION    = "relu"
OUTPUT_ACT    = "linear"

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE    = 16
EPOCHS        = 300

# EarlyStopping
ES_PATIENCE   = 30
ES_MONITOR    = "val_loss"
ES_RESTORE    = True

# ReduceLROnPlateau
REDUCE_FACTOR  = 0.5
REDUCE_PATIENCE= 12
REDUCE_MIN_LR  = 1e-6

# ── Output files ──────────────────────────────────────────────────────────────
MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, "best_model.weights.h5")
SCALER_FILE        = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_FILE       = os.path.join(MODEL_DIR, "encoder_meta.pkl")
METRICS_FILE       = os.path.join(OUTPUT_DIR, "metrics.json")
