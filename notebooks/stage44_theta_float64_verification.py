# %%
"""Stage 44: rerun ONLY centered Theta in float64, preserving Stage 41 design.

Keep stage43_split_tuple_audit.py beside this file (it need not be executed first).
Run: %run "/path/to/notebooks/stage44_theta_float64_verification.py"

Reads original Theta CSV, never notebook X_dict. Reuses the three other views
and the saved Stage-41 centered and historical standardized results. Uses the
same fixed k, seed, validation calibration, test ranks and ID tie policy.
No AE training, prevalence refits, grid randomizations or partition changes.
Writes a new manifest-bound checkpoint in 44_theta_float64_verification.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_helper_path = Path(__file__).resolve().with_name("stage43_split_tuple_audit.py")
if not _helper_path.is_file():
    raise FileNotFoundError("Keep stage43_split_tuple_audit.py in the same folder as Stage 44; no need to run it first.")
_spec = importlib.util.spec_from_file_location("_knot_stage43_helpers", _helper_path)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def conditional_score(cal_sse, cal_norm, test_sse, test_norm):
    """Exactly the Stage-41 calibration, including empty-bin fallback."""
    edges = np.unique(np.quantile(cal_norm, np.linspace(0, 1, 101)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    cb = np.searchsorted(edges[1:-1], cal_norm, side="right")
    tb = np.searchsorted(edges[1:-1], test_norm, side="right")
    populated = np.unique(cb)
    score = np.empty(len(test_sse), dtype=np.float64)
    for b in np.unique(tb):
        ref = np.sort(cal_sse[cb == b])
        if not len(ref):
            nearest = populated[np.argmin(np.abs(populated - b))]
            ref = np.sort(cal_sse[cb == nearest])
        score[tb == b] = np.searchsorted(ref, test_sse[tb == b], side="right") / (len(ref) + 1)
    return score


def aggregate(scores):
    ranked = np.column_stack([rankdata(s, method="average") / len(s) for s in scores])
    if not np.isfinite(ranked).all():
        raise ValueError("Nonfinite scores")
    return np.sort(ranked, axis=1)[:, -3]


def top_n(scores, ids, n):
    if not 0 < n <= len(ids) or len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Invalid selection size or scores")
    mask = np.zeros(len(ids), dtype=bool)
    mask[np.lexsort((ids, scores))[-n:]] = True
    return mask


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def fit_centered64(X, split, k, seed, batch):
    tr, ca, te = (split[key] for key in ("train_idx", "val_idx", "test_idx"))
    if k >= min(len(tr), X.shape[1]) or k < 1:
        raise ValueError("Invalid fixed PCA dimension")
    # Fill a single float64 training array without a full advanced-index int copy.
    Z = np.empty((len(tr), X.shape[1]), dtype=np.float64)
    for start in range(0, len(tr), batch):
        block = np.asarray(X[tr[start:start+batch]])
        if (np.abs(block.astype(np.float64)) > 2**53).any():
            raise ValueError("Source integers exceed exact float64 range; a higher-precision design is required.")
        Z[start:start+len(block)] = block
    scaler = StandardScaler(with_std=False, copy=False)
    scaler.fit_transform(Z)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed, copy=False).fit(Z)
    del Z
    if scaler.mean_.dtype != np.float64 or pca.components_.dtype != np.float64 or pca.explained_variance_ratio_.dtype != np.float64:
        raise RuntimeError("Fit was not performed entirely in float64")

    def evaluate(indices):
        errors = np.empty(len(indices), dtype=np.float64)
        norms = np.empty(len(indices), dtype=np.float64)
        for start in range(0, len(indices), batch):
            raw = np.asarray(X[indices[start:start+batch]])
            if (np.abs(raw.astype(np.float64)) > 2**53).any():
                raise ValueError("Source integers exceed exact float64 range")
            z = scaler.transform(raw.astype(np.float64))
            residual = z - pca.inverse_transform(pca.transform(z))
            if z.dtype != np.float64 or residual.dtype != np.float64:
                raise RuntimeError("Reconstruction was not performed in float64")
            errors[start:start+len(z)] = np.sum(residual * residual, axis=1, dtype=np.float64)
            norms[start:start+len(z)] = np.log1p(np.sum(z * z, axis=1, dtype=np.float64))
        return errors, norms

    cal_sse, cal_norm = evaluate(ca)
    test_sse, test_norm = evaluate(te)
    for a in (cal_sse, cal_norm, test_sse, test_norm):
        if not np.isfinite(a).all() or (a < 0).any():
            raise RuntimeError("Invalid numerical reconstruction result")
    return dict(test_score=conditional_score(cal_sse, cal_norm, test_sse, test_norm),
                test_raw_sse=test_sse, test_log_norm=test_norm,
                val_raw_sse=cal_sse, val_log_norm=cal_norm,
                k=np.array([k]), evr=np.array([pca.explained_variance_ratio_.sum()], dtype=np.float64),
                scaler_mean=scaler.mean_, pca_mean=pca.mean_, pca_components=pca.components_,
                pca_explained_variance=pca.explained_variance_,
                fit_dtype=np.array([pca.components_.dtype.name]))


def load_result(path, ids, key="test_score"):
    with np.load(path, allow_pickle=False) as q:
        result = {k: q[k].copy() for k in q.files}
    if "test_ids" in result and not np.array_equal(result["test_ids"].astype(str), ids):
        raise ValueError(f"Cached test-ID order mismatch: {path}")
    if key not in result or result[key].shape != (len(ids),) or not np.isfinite(result[key]).all():
        raise ValueError(f"Invalid cached scores: {path}")
    return result


def validate_stage41_hashes(old, paths):
    hashes = old.get("input_hashes", {})
    actual = {}
    for path in paths:
        values = [value for name, value in hashes.items() if Path(name).name == path.name]
        if len(set(values)) != 1:
            raise ValueError(f"Stage-41 manifest must bind exactly one content hash for {path.name}")
        actual[str(path.resolve())] = H.digest(path)
        if actual[str(path.resolve())] != values[0]:
            raise ValueError(f"Input differs from Stage 41: {path}. Select the original run; no fitting performed.")
    return actual


def main():
    p = H.parser(__doc__)
    p.add_argument("--stage41-dir", type=Path)
    p.add_argument("--phenotype", type=Path)
    args = p.parse_args()
    if args.chunk_rows < 1:
        p.error("--chunk-rows must be positive")
    root = args.root.expanduser().resolve()
    out = args.out or root / "44_theta_float64_verification"
    old_dir = args.stage41_dir or root / "41_theta_preprocessing_direct_fixed"
    old_manifest_path = H.required(old_dir / "manifest.json")
    old = json.loads(old_manifest_path.read_text())
    if old.get("rank_rule") != "fixed" or old.get("norm_bins") != 100:
        raise ValueError("This stage verifies the fixed-k, 100-bin Stage-41 design only")
    k, n, seed = (int(old[key]) for key in ("k_fixed", "n_selected", "seed"))
    batch = int(old["batch_size"])
    if batch < 1:
        raise ValueError("Invalid saved batch size")
    atlas, ids_all, split, inputs = H.load_frozen(args)
    te = split["test_idx"]
    ids = ids_all[te]
    source = H.required(args.data_dir / "theta_upto15.csv")
    phenotype_path = H.required(args.phenotype or root / "20_mathematical_phenotype/complete_mathematical_phenotype_atlas.parquet")
    score_paths = [H.required(root / "23_anomaly_score_baselines/checkpoints" / (H.safe(v) + "_test_scores.npz")) for v in H.VIEWS[:4]]
    old_path = H.required(old_dir / "centered_only.npz")
    previous = load_result(old_path, ids)
    if "test_ids" not in previous:
        raise ValueError("Stage-41 centered checkpoint lacks test_ids")
    if int(previous["k"][0]) != k:
        raise ValueError("Stage-41 centered k differs from its manifest")
    cache = {v: load_result(path, ids, "test_conditional_percentile_100") for v, path in zip(H.VIEWS[:4], score_paths)}
    if int(cache["Theta"]["pca_k"][0]) != k:
        raise ValueError("Historical Theta k differs from Stage 41")
    current_hashes = validate_stage41_hashes(old, inputs + [source, phenotype_path] + score_paths)
    phenotype = pd.read_csv(phenotype_path) if phenotype_path.suffix == ".csv" else pd.read_parquet(phenotype_path)
    if phenotype[H.ID].isna().any() or phenotype[H.ID].duplicated().any():
        raise ValueError("Invalid phenotype identifiers")
    phenotype[H.ID] = phenotype[H.ID].astype(str)
    phenotype = phenotype.set_index(H.ID).reindex(ids)
    width_col = next(c for c in ["khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count"] if c in phenotype)
    support_col = next(c for c in ["khovanov_support_size", "kh_support_size", "khovanov_f_support_size"] if c in phenotype)
    width, support = phenotype[width_col].to_numpy(float), phenotype[support_col].to_numpy(float)
    alternating = pd.to_numeric(atlas.iloc[te].is_alternating).to_numpy(float)
    if not np.isfinite(np.column_stack([width, support, alternating])).all():
        raise ValueError("Missing phenotype values")
    other_scores = [cache[v]["test_conditional_percentile_100"] for v in H.VIEWS[:3]]
    baseline = dict(test_score=cache["Theta"]["test_conditional_percentile_100"],
                    test_raw_sse=cache["Theta"]["test_raw_sse"],
                    test_log_norm=cache["Theta"]["test_log_norm"], k=np.array([k]))
    print(f"Only centered Theta will be fitted: float64, k={k}, seed={seed}, n={n}, batch={batch}.")
    environment_match = old.get("numpy") == np.__version__ and old.get("sklearn") == sklearn.__version__
    if not environment_match:
        print(f"Environment differs: Stage41 numpy={old.get('numpy')}, sklearn={old.get('sklearn')}; current={np.__version__}, {sklearn.__version__}.")
        print("Comparison will be marked precision_plus_environment; do not attribute every change solely to precision.")
    if args.check_only:
        print("Preflight and Stage-41 input hashes verified; run without --check-only to fit.")
        return
    extra_files = [old_path, old_manifest_path, Path(__file__), _helper_path]
    manifest = dict(stage=44, version=1, dtype="float64", k=k, n=n, seed=seed, batch_size=batch,
                    source_read_chunk_rows=args.chunk_rows, numpy=np.__version__, sklearn=sklearn.__version__,
                    scipy=scipy.__version__, python=platform.python_version(),
                    stage41_environment_matches=environment_match,
                    input_sha256={**current_hashes, **{str(f.resolve()): H.digest(f) for f in extra_files}})
    H.freeze(out, manifest)
    new_path = out / "centered_only_float64.npz"
    if new_path.exists():
        result = load_result(new_path, ids)
        print("Reused manifest-matched float64 checkpoint.")
    else:
        print("Reading original Theta coefficients; ignoring notebook X_dict.", flush=True)
        with H.source_matrix(source, "Theta", ids_all, args.work_dir, args.chunk_rows) as (X, columns, source_info):
            if "feature_names" not in previous or list(previous["feature_names"].astype(str)) != columns:
                raise ValueError("Source feature order differs from Stage 41")
            # If Stage 41 used live X_dict, require that it was exactly the
            # float32 cast of this canonical source matrix, as that stage hashed it.
            live_hash = old.get("input_hashes", {}).get("live_coefficients:Theta")
            if live_hash:
                h = H.hashlib.sha256()
                h.update(str((X.shape, np.dtype(np.float32).str, columns)).encode())
                for start in range(0, len(X), 8192):
                    h.update(np.ascontiguousarray(X[start:start+8192], dtype=np.float32).tobytes())
                if h.hexdigest() != live_hash:
                    raise ValueError("Stage-41 live coefficients do not match original-source float32 conversion")
            print("Fitting centered Theta in float64...", flush=True)
            result = fit_centered64(X, split, k, seed, batch)
            result["test_ids"] = ids
            result["feature_names"] = np.asarray(columns, str)
            H.atomic_npz(new_path, **result)
            H.atomic_json(out / "source_read_audit.json", source_info)
    if str(result["fit_dtype"][0]) != "float64" or result["pca_components"].dtype != np.float64:
        raise ValueError("Checkpoint is not float64")
    results = {"historical_standardized": baseline, "centered_only_stage41": previous, "centered_only_float64": result}
    masks, aggregates, rows, selected = {}, {}, [], []
    for name, value in results.items():
        aggregates[name] = aggregate(other_scores + [value["test_score"]])
        masks[name] = top_n(aggregates[name], ids, n)
        mask = masks[name]
        cutoff = aggregates[name][mask].min()
        rows.append(dict(variant=name, n=int(mask.sum()), k=k, evr=float(value.get("evr", [np.nan])[0]),
            kh_diagonal_mean=float(width[mask].mean()), kh_diagonal_ge_3_prop=float((width[mask] >= 3).mean()),
            kh_diagonal_ge_4_prop=float((width[mask] >= 4).mean()), kh_support_mean=float(support[mask].mean()),
            alternating_prop=float(alternating[mask].mean()),
            raw_sse_min=float(value["test_raw_sse"].min()), raw_sse_median=float(np.median(value["test_raw_sse"])),
            raw_sse_max=float(value["test_raw_sse"].max()), raw_sse_zero_n=int((value["test_raw_sse"] == 0).sum()),
            boundary_tied_total=int((aggregates[name] == cutoff).sum()),
            boundary_tied_selected=int(((aggregates[name] == cutoff) & mask).sum())))
        selected.extend({"variant": name, H.ID: knot} for knot in ids[mask])
    comparisons = []
    for old_name in ("centered_only_stage41", "historical_standardized"):
        a, b = masks[old_name], masks["centered_only_float64"]
        comparisons.append(dict(reference=old_name, new="centered_only_float64", intersection=int((a & b).sum()),
            union=int((a | b).sum()), jaccard=float((a & b).sum() / (a | b).sum()),
            theta_score_spearman=correlation(results[old_name]["test_score"], result["test_score"]),
            aggregate_spearman=correlation(aggregates[old_name], aggregates["centered_only_float64"]),
            raw_sse_spearman=correlation(results[old_name]["test_raw_sse"], result["test_raw_sse"]),
            delta_mean_width=float(width[b].mean()-width[a].mean()),
            delta_ge4_prop=float((width[b] >= 4).mean()-(width[a] >= 4).mean()),
            comparison_scope=("precision_only_recorded_packages" if environment_match else "precision_plus_environment")
                if old_name == "centered_only_stage41" else "preprocessing_and_precision"))
    evr = float(result["evr"][0])
    numerical_ok = bool(np.isfinite(evr) and 0 <= evr <= 1 + 1e-12)
    H.atomic_json(out / "float64_numeric_checks.json", dict(
        fit_dtype=str(result["fit_dtype"][0]), evr=evr, evr_excess_above_one=max(0., evr-1),
        evr_within_unit_interval_tolerance=numerical_ok, tolerance=1e-12,
        stage41_environment_matches=environment_match,
        all_test_residuals_finite=bool(np.isfinite(result["test_raw_sse"]).all()),
        note="Numerical checks do not by themselves establish selection stability; inspect the comparison table."))
    pd.DataFrame(rows).to_csv(out / "theta_float64_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "theta_float64_comparison.csv", index=False)
    pd.DataFrame(selected).to_csv(out / "theta_float64_selected_ids.csv", index=False)
    pd.DataFrame({H.ID: ids, "stage41_centered_selected": masks["centered_only_stage41"],
                  "float64_selected": masks["centered_only_float64"],
                  "stage41_centered_score": previous["test_score"], "float64_score": result["test_score"],
                  "stage41_centered_sse": previous["test_raw_sse"], "float64_sse": result["test_raw_sse"]}).to_csv(out / "theta_float64_test_comparison.csv", index=False)
    (out / "interpretation.txt").write_text(
        "Only centered Theta was refitted; all other view scores remain frozen.\n"
        "Inspect Jaccard and phenotype differences; no automatic robustness threshold is imposed.\n"
        "The historical standardized comparison changes preprocessing as well as precision.\n"
        "This does not test the canonical 220-knot population or its 48-knot subset.\n"
        "If the saved package versions differ, the centered comparison also includes an environment change.\n"
        "The summary ZIP omits the NPZ checkpoint; retain it in Drive.\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    H.bundle(out, "stage44_review.zip")
    if not numerical_ok:
        raise RuntimeError("Outputs saved, but the float64 explained-variance check failed; inspect before using in the paper.")
    print("STAGE 44 COMPLETE:", out)


if __name__ == "__main__":
    main()
