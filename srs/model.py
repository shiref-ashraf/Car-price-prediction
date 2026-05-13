"""
model.py — Model architecture definition and training loop.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from config import (
    SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    HIDDEN_UNITS, DROPOUT_RATES, ACTIVATION, OUTPUT_ACT,
    ES_PATIENCE, ES_MONITOR, ES_RESTORE,
    REDUCE_FACTOR, REDUCE_PATIENCE, REDUCE_MIN_LR,
    VAL_SPLIT, MODEL_WEIGHTS_FILE
)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Architecture ──────────────────────────────────────────────────────────────
def build_model(input_dim: int) -> keras.Model:
    """
    Deep regression network with BatchNorm + Dropout regularisation.

    Architecture
    ─────────────────────────────────────
      Input  (input_dim,)
        │
      Dense(128, relu) → BatchNorm → Dropout(0.15)
        │
      Dense(64,  relu) → BatchNorm → Dropout(0.10)
        │
      Dense(32,  relu)
        │
      Dense(1,  linear)          ← predicts log(Price + 1)
    ─────────────────────────────────────
    Loss : MSE in log-space
    Optimizer : Adam (lr=1e-3) with ReduceLROnPlateau
    """
    assert len(HIDDEN_UNITS) == len(DROPOUT_RATES), \
        "HIDDEN_UNITS and DROPOUT_RATES must have the same length."

    model_layers = [layers.Input(shape=(input_dim,))]

    for units, drop in zip(HIDDEN_UNITS, DROPOUT_RATES):
        model_layers.append(layers.Dense(units, activation=ACTIVATION,
                                         kernel_initializer="he_normal"))
        model_layers.append(layers.BatchNormalization())
        if drop > 0:
            model_layers.append(layers.Dropout(drop))

    model_layers.append(layers.Dense(1, activation=OUTPUT_ACT))

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks() -> list:
    """Return EarlyStopping + ReduceLROnPlateau callbacks."""
    cb_stop = callbacks.EarlyStopping(
        monitor=ES_MONITOR,
        patience=ES_PATIENCE,
        restore_best_weights=ES_RESTORE,
        verbose=1,
    )
    cb_reduce = callbacks.ReduceLROnPlateau(
        monitor=ES_MONITOR,
        factor=REDUCE_FACTOR,
        patience=REDUCE_PATIENCE,
        min_lr=REDUCE_MIN_LR,
        verbose=1,
    )
    return [cb_stop, cb_reduce]


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(
    model: keras.Model,
    X_train,
    y_train_log,
    save_weights: bool = True,
) -> keras.callbacks.History:
    """
    Train the model; optionally save best weights.

    Parameters
    ----------
    model        : compiled Keras model from build_model()
    X_train      : numpy array or DataFrame (float32)
    y_train_log  : log-transformed target array / Series
    save_weights : persist weights to MODEL_WEIGHTS_FILE

    Returns
    -------
    history : Keras History object
    """
    cb_list = get_callbacks()

    if save_weights:
        cb_list.append(
            callbacks.ModelCheckpoint(
                filepath=MODEL_WEIGHTS_FILE,
                monitor=ES_MONITOR,
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )
        )

    history = model.fit(
        X_train, y_train_log,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    print(f"\n[train] Stopped at epoch {len(history.history['loss'])}")
    best_val = min(history.history[ES_MONITOR])
    print(f"[train] Best val_loss: {best_val:.6f}")
    return history


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model(input_dim=25)
    m.summary()
    print("\nModel built successfully.")
