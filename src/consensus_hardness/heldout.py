# src/consensus_hardness/heldout.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .autoencoders import (
    set_all_seeds,
    build_mlp_autoencoder,
    compute_reconstruction_errors_batched,
)
from .hardsets import top_tail_indices, consensus_from_hard_sets, summarize_consensus_set
from .enrichment import standard_knot_enrichment_table
from .norms import safe_name

import tensorflow as tf
from tensorflow.keras import callbacks


def make_train_val_test_indices(
    meta: pd.DataFrame,
    split_seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a frozen split.

    The default is target-free.  Pass ``signature_bin`` explicitly only for
    the manuscript sensitivity analysis that reproduces the earlier split.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8

    idx_all = np.arange(len(meta))
    strat = None if stratify_col is None else meta[stratify_col].astype(str).values

    train_idx, temp_idx = train_test_split(
        idx_all,
        test_size=val_size + test_size,
        random_state=split_seed,
        shuffle=True,
        stratify=strat,
    )

    strat_temp = None if strat is None else strat[temp_idx]
    relative_test_size = test_size / (val_size + test_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=split_seed,
        shuffle=True,
        stratify=strat_temp,
    )

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def train_autoencoder_on_split(
    X: np.ndarray,
    name: str,
    latent_dim: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_seed: int = 0,
    epochs: int = 200,
    batch_size: int = 4096,
    patience: int = 20,
    learning_rate: float = 1e-3,
    dropout: float = 0.05,
    l2: float = 1e-6,
    save_dir: str | Path | None = None,
) -> dict:
    set_all_seeds(model_seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)

    input_dim = X_train.shape[1]

    model, encoder = build_mlp_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
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
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    test_sse, test_mse, test_nre = compute_reconstruction_errors_batched(
        model,
        X_test,
        batch_size=8192,
    )

    val_sse, val_mse, val_nre = compute_reconstruction_errors_batched(
        model,
        X_val,
        batch_size=8192,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save(save_dir / f"heldout_ae_{safe_name(name)}_k{latent_dim}_seed{model_seed}.keras")
        encoder.save(save_dir / f"heldout_encoder_{safe_name(name)}_k{latent_dim}_seed{model_seed}.keras")

    return {
        "name": name,
        "latent_dim": int(latent_dim),
        "input_dim": int(input_dim),
        "compression_ratio": float(input_dim / latent_dim),
        "model_seed": int(model_seed),
        "scaler": scaler,
        "model": model,
        "encoder": encoder,
        "history": history.history,
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "val_sse": val_sse,
        "val_mse": val_mse,
        "val_nre": val_nre,
        "test_sse": test_sse,
        "test_mse": test_mse,
        "test_nre": test_nre,
    }


def train_heldout_autoencoders(
    X_dict: dict[str, np.ndarray],
    latent_dims: dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 200,
    batch_size: int = 4096,
    patience: int = 20,
    learning_rate: float = 1e-3,
    dropout: float = 0.05,
    l2: float = 1e-6,
    save_dir: str | Path | None = None,
) -> dict[int, dict[str, dict]]:
    all_runs = {}

    for seed in seeds:
        print("\n" + "=" * 80)
        print("Running AE held-out seed:", seed)
        print("=" * 80)

        all_runs[seed] = {}

        for name, X in X_dict.items():
            print("\nTraining:", name, "seed:", seed)

            all_runs[seed][name] = train_autoencoder_on_split(
                X=X,
                name=name,
                latent_dim=latent_dims[name],
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                model_seed=seed,
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
                "best val loss:",
                all_runs[seed][name]["best_val_loss"],
            )

    return all_runs


def build_heldout_hard_sets(
    result_dict: dict[str, dict],
    score_name: str = "test_sse",
    tail_mass: float = 0.01,
) -> dict[str, set[int]]:
    hard_sets = {}

    for name, out in result_dict.items():
        scores = out[score_name]
        global_test_idx = out["test_idx"]

        local_tail_idx = top_tail_indices(scores, tail_mass=tail_mass)
        global_tail_idx = global_test_idx[local_tail_idx]

        hard_sets[name] = set(map(int, global_tail_idx.tolist()))

    return hard_sets


def summarize_heldout_run(
    result_dict: dict[str, dict],
    meta: pd.DataFrame,
    label: str,
    score_name: str = "test_sse",
    tail_mass: float = 0.01,
) -> tuple[dict[str, set[int]], set[int], dict, pd.DataFrame]:
    hard_sets = build_heldout_hard_sets(
        result_dict,
        score_name=score_name,
        tail_mass=tail_mass,
    )

    consensus = consensus_from_hard_sets(hard_sets)

    test_idx_run = list(result_dict.values())[0]["test_idx"]

    background_df = meta.loc[test_idx_run].copy()
    selected_df = meta.loc[sorted(consensus)].copy()

    summary = summarize_consensus_set(
        consensus,
        meta,
        score_name=score_name,
        evr_threshold=None,
        tail_mass=tail_mass,
    )

    # Important correction for held-out analysis
    summary["tail_size_each_invariant"] = int(np.ceil(tail_mass * len(test_idx_run)))
    summary["background_size"] = int(len(test_idx_run))
    summary["label"] = label

    enrich = standard_knot_enrichment_table(
        selected_df,
        background_df,
    )

    return hard_sets, consensus, summary, enrich

def summarize_heldout_ae_runs(
    ae_runs: dict[int, dict[str, dict]],
    meta: pd.DataFrame,
    tail_mass: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    summaries = []
    enrichments = {}
    consensus_sets = {}
    hard_sets_by_seed = {}

    for seed, run in ae_runs.items():
        hard_sets, consensus, summary, enrich = summarize_heldout_run(
            run,
            meta,
            label=f"AE held-out k99 seed {seed}",
            score_name="test_sse",
            tail_mass=tail_mass,
        )

        hard_sets_by_seed[seed] = hard_sets
        consensus_sets[seed] = consensus
        summaries.append(summary)

        enrich = enrich.copy()
        enrich["seed"] = seed
        enrichments[seed] = enrich

    summary_df = pd.DataFrame(summaries)
    enrichment_all = pd.concat(enrichments.values(), ignore_index=True)

    return summary_df, enrichment_all, consensus_sets, hard_sets_by_seed


def pairwise_jaccard_matrix(
    consensus_sets: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = list(consensus_sets.keys())
    mat = np.zeros((len(seeds), len(seeds)))
    rows = []

    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            a = consensus_sets[s1]
            b = consensus_sets[s2]

            inter = len(a & b)
            union = len(a | b)

            jacc = inter / union if union else np.nan
            mat[i, j] = jacc

            if i < j:
                rows.append(
                    {
                        "seed_a": s1,
                        "seed_b": s2,
                        "size_a": len(a),
                        "size_b": len(b),
                        "overlap": inter,
                        "jaccard": jacc,
                    }
                )

    return pd.DataFrame(mat, index=seeds, columns=seeds), pd.DataFrame(rows)


def consensus_frequency_across_sets(
    meta: pd.DataFrame,
    test_idx: np.ndarray,
    consensus_sets: dict[int, set[int]],
    output_col: str = "ae_consensus_frequency",
) -> pd.DataFrame:
    freq = pd.Series(0, index=test_idx)

    for _, sset in consensus_sets.items():
        for idx in sset:
            freq.loc[idx] += 1

    out = meta.loc[test_idx].copy()
    out[output_col] = freq.values

    return out


def fit_pca_on_split(
    X: np.ndarray,
    name: str,
    k: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)

    pca = PCA(n_components=k, svd_solver="full")
    pca.fit(X_train)

    Z_test = pca.transform(X_test)
    Xhat_test = pca.inverse_transform(Z_test)

    diff = X_test - Xhat_test

    test_sse = np.sum(diff**2, axis=1)
    test_mse = test_sse / X_test.shape[1]
    test_nre = test_sse / (np.sum(X_test**2, axis=1) + eps)

    return {
        "name": name,
        "k": int(k),
        "input_dim": int(X_test.shape[1]),
        "compression_ratio": float(X_test.shape[1] / k),
        "evr_train": float(np.sum(pca.explained_variance_ratio_)),
        "scaler": scaler,
        "pca": pca,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "test_sse": test_sse,
        "test_mse": test_mse,
        "test_nre": test_nre,
    }


def run_holdout_pca_for_representations(
    X_dict: dict[str, np.ndarray],
    k_dict: dict[str, int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, dict]:
    results = {}

    for name, X in X_dict.items():
        print("PCA held-out:", name)

        results[name] = fit_pca_on_split(
            X=X,
            name=name,
            k=k_dict[name],
            train_idx=train_idx,
            test_idx=test_idx,
        )

    return results


def summarize_holdout_reconstruction(
    result_dict: dict[str, dict],
) -> pd.DataFrame:
    rows = []

    for name, out in result_dict.items():
        rows.append(
            {
                "invariant": name,
                "input_dim": out["input_dim"],
                "k": out["k"],
                "compression_ratio": out["compression_ratio"],
                "evr_train": out["evr_train"],
                "median_test_mse": np.median(out["test_mse"]),
                "mean_test_mse": np.mean(out["test_mse"]),
                "median_test_sse": np.median(out["test_sse"]),
                "mean_test_sse": np.mean(out["test_sse"]),
            }
        )

    return pd.DataFrame(rows)


def add_holdout_errors_to_meta(
    meta: pd.DataFrame,
    result_dict: dict[str, dict],
    score_prefix: str,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    out = meta.copy()

    for name, res in result_dict.items():
        key = safe_name(name)

        out[f"{key}_{score_prefix}_sse"] = np.nan
        out[f"{key}_{score_prefix}_mse"] = np.nan
        out[f"{key}_{score_prefix}_nre"] = np.nan

        out.loc[test_idx, f"{key}_{score_prefix}_sse"] = res["test_sse"]
        out.loc[test_idx, f"{key}_{score_prefix}_mse"] = res["test_mse"]
        out.loc[test_idx, f"{key}_{score_prefix}_nre"] = res["test_nre"]

    sse_cols = [f"{safe_name(name)}_{score_prefix}_sse" for name in result_dict.keys()]
    nre_cols = [f"{safe_name(name)}_{score_prefix}_nre" for name in result_dict.keys()]

    out[f"mean_{score_prefix}_sse"] = out[sse_cols].mean(axis=1)
    out[f"max_{score_prefix}_sse"] = out[sse_cols].max(axis=1)
    out[f"mean_{score_prefix}_nre"] = out[nre_cols].mean(axis=1)
    out[f"max_{score_prefix}_nre"] = out[nre_cols].max(axis=1)

    return out
