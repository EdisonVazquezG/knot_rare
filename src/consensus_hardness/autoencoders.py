# src/consensus_hardness/autoencoders.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, regularizers, callbacks, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .norms import safe_name


def set_all_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_hidden_dims(input_dim: int, latent_dim: int | None = None) -> list[int]:
    if input_dim <= 32:
        return [64, 32]
    if input_dim <= 80:
        return [128, 64]
    if input_dim <= 200:
        return [256, 128]
    if input_dim <= 500:
        return [512, 256, 128]
    return [512, 256, 128]


def build_mlp_autoencoder(
    input_dim: int,
    latent_dim: int,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.05,
    l2: float = 1e-6,
    learning_rate: float = 1e-3,
) -> tuple[Model, Model]:
    if hidden_dims is None:
        hidden_dims = default_hidden_dims(input_dim, latent_dim)

    inp = layers.Input(shape=(input_dim,), name="input")

    x = inp
    for h in hidden_dims:
        x = layers.Dense(
            h,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2),
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    z = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    x = z
    for h in reversed(hidden_dims):
        x = layers.Dense(
            h,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2),
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    out = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = Model(inp, out, name=f"AE_d{input_dim}_k{latent_dim}")
    encoder = Model(inp, z, name=f"Encoder_d{input_dim}_k{latent_dim}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model, encoder


def compute_reconstruction_errors_batched(
    model: Model,
    Xs: np.ndarray,
    batch_size: int = 8192,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = Xs.shape[0]
    d = Xs.shape[1]

    sse = np.zeros(n, dtype=np.float32)
    mse = np.zeros(n, dtype=np.float32)
    nre = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        xb = Xs[start:end]
        xhat = model.predict(xb, batch_size=batch_size, verbose=0)

        diff = xb - xhat

        sse_b = np.sum(diff**2, axis=1)
        norm_b = np.sum(xb**2, axis=1)

        sse[start:end] = sse_b
        mse[start:end] = sse_b / d
        nre[start:end] = sse_b / (norm_b + eps)

    return sse, mse, nre


def train_autoencoder_for_invariant(
    X: np.ndarray,
    latent_dim: int,
    name: str,
    seed: int = 42,
    validation_size: float = 0.15,
    stratify: np.ndarray | pd.Series | None = None,
    epochs: int = 200,
    batch_size: int = 4096,
    patience: int = 20,
    learning_rate: float = 1e-3,
    dropout: float = 0.05,
    l2: float = 1e-6,
    save_dir: str | Path | None = None,
) -> dict:
    """
    Full-dataset AE baseline.

    The scaler is fitted on all X, and the model is trained using an internal
    train/validation split. Reconstruction errors are then computed for all rows.
    """

    set_all_seeds(seed)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    n = Xs.shape[0]
    input_dim = Xs.shape[1]

    train_idx, val_idx = train_test_split(
        np.arange(n),
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    model, encoder = build_mlp_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=None,
        dropout=dropout,
        l2=l2,
        learning_rate=learning_rate,
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(5, patience // 3),
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        Xs[train_idx],
        Xs[train_idx],
        validation_data=(Xs[val_idx], Xs[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    sse, mse, nre = compute_reconstruction_errors_batched(
        model,
        Xs,
        batch_size=8192,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save(save_dir / f"autoencoder_{safe_name(name)}_k{latent_dim}.keras")
        encoder.save(save_dir / f"encoder_{safe_name(name)}_k{latent_dim}.keras")

    return {
        "name": name,
        "latent_dim": int(latent_dim),
        "input_dim": int(input_dim),
        "compression_ratio": float(input_dim / latent_dim),
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "history": history.history,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_final_loss": float(history.history["loss"][-1]),
        "val_final_loss": float(history.history["val_loss"][-1]),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "sse": sse,
        "mse": mse,
        "nre": nre,
    }


def train_autoencoders_for_representations(
    X_dict: dict[str, np.ndarray],
    latent_dims: dict[str, int],
    seed: int = 42,
    validation_size: float = 0.15,
    stratify: np.ndarray | pd.Series | None = None,
    epochs: int = 200,
    batch_size: int = 4096,
    patience: int = 20,
    learning_rate: float = 1e-3,
    dropout: float = 0.05,
    l2: float = 1e-6,
    save_dir: str | Path | None = None,
) -> dict[str, dict]:
    results = {}

    for name, X in X_dict.items():
        if name not in latent_dims:
            raise KeyError(f"Missing latent dimension for {name}.")

        print(f"\nTraining AE: {name}")

        results[name] = train_autoencoder_for_invariant(
            X=X,
            latent_dim=latent_dims[name],
            name=name,
            seed=seed,
            validation_size=validation_size,
            stratify=stratify,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            learning_rate=learning_rate,
            dropout=dropout,
            l2=l2,
            save_dir=save_dir,
        )

        print(
            name,
            "latent:",
            results[name]["latent_dim"],
            "best val loss:",
            results[name]["best_val_loss"],
            "compression:",
            results[name]["compression_ratio"],
        )

    return results


def summarize_autoencoder_reconstruction(
    ae_results: dict[str, dict],
    pca_results: dict,
    X_dict: dict[str, np.ndarray],
    alpha: float = 0.99,
) -> pd.DataFrame:
    rows = []

    for name, out in ae_results.items():
        pca_sse = pca_results[name][alpha]["sse"]
        pca_mse = pca_sse / X_dict[name].shape[1]

        ae_mse = out["mse"]

        rows.append(
            {
                "invariant": name,
                "input_dim": out["input_dim"],
                "latent_dim": out["latent_dim"],
                "compression_ratio": out["compression_ratio"],
                "ae_best_val_loss": out["best_val_loss"],
                "pca_median_mse": np.median(pca_mse),
                "ae_median_mse": np.median(ae_mse),
                "median_mse_improvement_pca_over_ae": np.median(pca_mse) / np.median(ae_mse),
                "pca_mean_mse": np.mean(pca_mse),
                "ae_mean_mse": np.mean(ae_mse),
                "mean_mse_improvement_pca_over_ae": np.mean(pca_mse) / np.mean(ae_mse),
            }
        )

    return pd.DataFrame(rows)


def add_autoencoder_errors_to_meta(
    meta: pd.DataFrame,
    ae_results: dict[str, dict],
    suffix: str = "ae_99",
    consensus: set[int] | None = None,
    consensus_col: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = meta.copy()

    for name, res in ae_results.items():
        key = safe_name(name)

        out[f"{key}_{suffix}_sse"] = res["sse"]
        out[f"{key}_{suffix}_mse"] = res["mse"]
        out[f"{key}_{suffix}_nre"] = res["nre"]

    sse_cols = [f"{safe_name(name)}_{suffix}_sse" for name in ae_results.keys()]
    nre_cols = [f"{safe_name(name)}_{suffix}_nre" for name in ae_results.keys()]

    out[f"mean_{suffix}_sse"] = out[sse_cols].mean(axis=1)
    out[f"max_{suffix}_sse"] = out[sse_cols].max(axis=1)
    out[f"mean_{suffix}_nre"] = out[nre_cols].mean(axis=1)
    out[f"max_{suffix}_nre"] = out[nre_cols].max(axis=1)

    if consensus is not None and consensus_col is not None:
        out[consensus_col] = False
        out.loc[list(consensus), consensus_col] = True

    return out, sse_cols, nre_cols