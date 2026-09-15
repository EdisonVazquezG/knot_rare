# %% [markdown]
# Stage 41 — Fit only missing Theta preprocessing variants; reuse other views.
#
# Standalone version for paper_run.ipynb, matching the Stage-39 workflow.
# Copy this file into the same notebooks/ folder as Stage 39 and run:
#   %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage41_theta_preprocessing.py"
#
# Reads OUTPUT_DIR / DATA_DIR from the notebook when available; alternatively
# set KNOT_OUTPUT_DIR / KNOT_DATA_DIR. No patch or fourth helper file required.
# Stages 41/42 EXECUTE by default; --check-only checks inputs without new fits.
# Existing analysis outputs stay read-only. New results use *_direct folders.
# Checkpoints resume completed variants/resolutions and reject changed inputs.
# The three files can run independently; running 40 first is recommended.
#
# Dependencies: numpy, pandas, scipy, scikit-learn, and pyarrow.
# Scientific scope: Stage 41 is the fixed-size held-out benchmark; Stage 42
# uses the distinct crossing-number split and frozen Stage-31 selection.

# %%
from __future__ import annotations
from pathlib import Path

import os

# Paths: environment override, then paper_run state, then historical defaults.
def _direct_state():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            return shell.user_ns
    except ImportError:
        pass
    return globals()


def _direct_default_data():
    return Path(os.environ.get("KNOT_DATA_DIR", str(_direct_state().get(
        "DATA_DIR", "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants"))))


def _direct_default_root():
    historical = _direct_default_data() / "processed_consensus_hardness" / "corrected_run_20260819"
    return Path(os.environ.get("KNOT_OUTPUT_DIR", str(_direct_state().get("OUTPUT_DIR", historical))))


def _direct_memory_view(name, ids=None):
    state = _direct_state()
    matrices, frame = state.get("X_dict"), state.get("meta")
    if matrices is None or frame is None or name not in matrices:
        return None
    if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Invalid IDs in notebook meta; reload alignment cells only.")
    X = np.asarray(matrices[name], dtype=np.float32)
    actual = frame[ID].astype(str).to_numpy()
    if len(actual) != len(X):
        raise ValueError(f"meta / X_dict row count mismatch: {name}")
    columns = state.get("feature_cols_dict", {}).get(name)
    if columns is None:
        columns = [f"coordinate_{j}" for j in range(X.shape[1])]
    columns = list(map(str, columns))
    if len(columns) != X.shape[1]:
        raise ValueError(f"Feature-column mismatch: {name}")
    if ids is not None and not np.array_equal(actual, np.asarray(ids, str)):
        raise ValueError(
            "Notebook meta and frozen atlas ID order differ. "
            "Reload the original alignment cells before this stage."
        )
    return X, columns


# Bundled helpers: no dependency on a new consensus_hardness module.
ID_COLS = ["knot_id", "knot_id_clean", "knot_id_base"]

META_CANDIDATES = [
    "number_of_crossings",
    "table_number",
    "is_alternating",
    "signature",
    "minimum_exponent",
    "maximum_exponent",
    "s_invariant",
]

def clean_knot_id(x) -> str | None:
    if pd.isna(x):
        return None
    return str(x).strip()

def base_knot_id(x, mirror_symbol: str = "!") -> str | None:
    if pd.isna(x):
        return None
    return str(x).strip().replace(mirror_symbol, "")

def add_knot_ids(
    df: pd.DataFrame,
    id_col: str = "knot_id",
    mirror_symbol: str = "!",
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"Expected id column '{id_col}' not found.")

    out = df.copy()
    out["knot_id_clean"] = out[id_col].map(clean_knot_id)
    out["knot_id_base"] = out[id_col].map(
        lambda x: base_knot_id(x, mirror_symbol=mirror_symbol)
    )
    return out

def canonicalize_mirrors_by_signature(
    df: pd.DataFrame,
    id_col: str = "knot_id",
    base_col: str = "knot_id_base",
    signature_col: str = "signature",
    mirror_symbol: str = "!",
) -> pd.DataFrame:
    """
    Keep one representative per base knot ID.

    Priority:
    1. Prefer representative with non-negative signature.
    2. If tied, prefer non-mirror ID, i.e. without mirror_symbol.
    """

    required = {id_col, base_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    out["_is_nonmirror"] = ~out[id_col].astype(str).str.contains(
        mirror_symbol,
        regex=False,
    )

    if signature_col in out.columns:
        out["_signature_numeric"] = pd.to_numeric(
            out[signature_col],
            errors="coerce",
        )
        out["_has_nonnegative_signature"] = out["_signature_numeric"] >= 0

        sort_cols = [
            base_col,
            "_has_nonnegative_signature",
            "_is_nonmirror",
        ]
        ascending = [True, False, False]

        drop_cols = [
            "_signature_numeric",
            "_has_nonnegative_signature",
            "_is_nonmirror",
        ]
    else:
        sort_cols = [base_col, "_is_nonmirror"]
        ascending = [True, False]
        drop_cols = ["_is_nonmirror"]

    out = (
        out.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(base_col, keep="first")
        .drop(columns=drop_cols)
        .copy()
    )

    return out



import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ID = "knot_id_base"
VIEWS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
METHODS = ("raw_sse", "relative_nre", "residual_mahalanobis",
           "residual_isolation_forest", "conditional_percentile_100")
SOURCES = {
    "Alexander": ("Alexander_upto17.csv", "A"),
    "Jones": ("Jones_upto17_MIRRORS.csv", "J"),
    "HOMFLY-PT": ("HomflyPt_upto15_MIRRORS.csv", "a"),
    "Theta": ("theta_upto15.csv", "T"),
    "Khovanov": ("even_KH_upto17.pkl", "F_"),
}


def safe(name):
    return name.replace("-", "_").replace(" ", "_")


def unique_file(root, name):
    hits = sorted(Path(root).rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} below {root}; found {hits}")
    return hits[0]


def fingerprint(paths):
    """Content hashes, so caches cannot silently survive input changes."""
    out = {}
    for path in paths:
        path = Path(path)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        out[str(path.resolve())] = h.hexdigest()
    # Bind checkpoints to live coefficients as well as archived source files.
    for name, (filename, _) in SOURCES.items():
        if any(Path(p).name == filename for p in paths):
            live = _direct_memory_view(name)
            if live is not None:
                X, columns = live
                h = hashlib.sha256()
                h.update(str((X.shape, X.dtype.str, columns)).encode())
                for start in range(0, len(X), 8192):
                    h.update(memoryview(np.ascontiguousarray(X[start:start+8192])).cast("B"))
                out["live_coefficients:" + name] = h.hexdigest()
    return out


def freeze_manifest(out, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError(f"Inputs/design changed: choose a new output directory; preserving {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_atlas_split(root):
    atlas_path = unique_file(root, "final_hard_regime_atlas.parquet")
    split_path = unique_file(root, "heldout_ae_seed_0.npz")
    atlas = pd.read_parquet(atlas_path).reset_index(drop=True)
    if atlas[ID].isna().any() or atlas[ID].duplicated().any():
        raise ValueError("Atlas IDs must be unique and nonmissing")
    atlas[ID] = atlas[ID].astype(str)
    with np.load(split_path, allow_pickle=False) as p:
        split = {k: np.asarray(p[k], dtype=np.int64) for k in ("train_idx", "val_idx", "test_idx")}
    flat = np.concatenate(list(split.values()))
    if not np.array_equal(np.sort(flat), np.arange(len(atlas))):
        raise ValueError("Frozen split must be an exact disjoint partition of atlas rows")
    return atlas, split, [atlas_path, split_path]


def load_outcomes(root, ids):
    path = unique_file(root, "complete_mathematical_phenotype_atlas.parquet")
    df = pd.read_parquet(path)
    if df[ID].duplicated().any():
        raise ValueError("Duplicate phenotype IDs")
    df[ID] = df[ID].astype(str)
    df = df.set_index(ID).reindex(ids)
    width = next(c for c in ("khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count",
                             "khovanov_diagonal_count") if c in df)
    support = next(c for c in ("khovanov_support_size", "kh_support_size",
                               "khovanov_f_support_size") if c in df)
    result = df[[width, support]].to_numpy(float)
    if not np.isfinite(result).all():
        raise ValueError("Missing/nonfinite phenotype values")
    return result[:, 0], result[:, 1], path


def top_n(scores, ids, n):
    scores, ids = np.asarray(scores, float), np.asarray(ids, str)
    if not 0 < n <= len(scores) or len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Invalid scores, IDs, or target cardinality")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate selection IDs")
    mask = np.zeros(len(scores), bool)
    # Exact historical Stage 23B/33 tie policy: larger lexicographic IDs first.
    mask[np.lexsort((ids, scores))[-n:]] = True
    return mask


def c3(scores):
    matrix = np.column_stack([rankdata(s, method="average") / len(s) for s in scores])
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite scores")
    return np.sort(matrix, axis=1)[:, -3]


def overlap(a, b):
    a, b = set(a), set(b)
    return dict(n_a=len(a), n_b=len(b), intersection=len(a & b), union=len(a | b),
                jaccard=len(a & b) / len(a | b) if a | b else float("nan"))


def norm_bins(cal_norm, test_norm, bins=100):
    edges = np.unique(np.quantile(cal_norm, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return (np.searchsorted(edges[1:-1], cal_norm, side="right"),
            np.searchsorted(edges[1:-1], test_norm, side="right"))


def conditional_score(cal_sse, cal_norm, test_sse, test_norm):
    cb, tb = norm_bins(cal_norm, test_norm)
    score = np.empty(len(test_sse))
    populated = np.unique(cb)
    for b in np.unique(tb):
        ref = np.sort(cal_sse[cb == b])
        if not len(ref):
            nearest = populated[np.argmin(abs(populated - b))]
            ref = np.sort(cal_sse[cb == nearest])
        score[tb == b] = np.searchsorted(ref, test_sse[tb == b], side="right") / (len(ref) + 1)
    return score


def coarsen(codes, target):
    """Exactly Stage 31: coarsen occupied 100-bin codes, not fresh quantiles."""
    u = np.unique(codes)
    n = min(target, len(u))
    if n < 1:
        raise ValueError("Empty codes or invalid bin count")
    return np.minimum(np.floor(np.searchsorted(u, codes) * n / len(u)).astype(int), n - 1)


def feature_mask(train, prevalence):
    if not 0 <= prevalence <= 1:
        raise ValueError("Prevalence must be in [0,1]")
    return np.mean(train != 0, axis=0) >= prevalence


def read_view(path, name, ids, chunk_rows=5000):
    """Read one archived view; keep canonical representatives exactly as Stage 23.

    CSV inputs are streamed. Legacy PKL input must fit in RAM, as in Stage 35.
    """

    live = _direct_memory_view(name, ids)
    if live is not None:
        print(f"Reusing aligned {name} coefficients from paper_run memory.", flush=True)
        return live
    path = Path(path)
    wanted = set(map(str, ids))
    chunks = pd.read_csv(path, chunksize=chunk_rows) if path.suffix == ".csv" else [pd.read_pickle(path)]
    parts = []
    for chunk in chunks:
        base = chunk["knot_id"].astype(str).str.strip().str.replace("!", "", regex=False)
        kept = chunk.loc[base.isin(wanted)]
        if len(kept):
            parts.append(kept.copy())
    if not parts:
        raise ValueError(f"No requested IDs in {path}")
    frame = canonicalize_mirrors_by_signature(add_knot_ids(pd.concat(parts, ignore_index=True)))
    frame = frame.set_index(ID).loc[np.asarray(ids, str)]

    prefix = SOURCES[name][1]
    cols = [c for c in frame if c not in ID_COLS + META_CANDIDATES and str(c).startswith(prefix)]
    # Match representations.split_metadata_and_features exactly.
    X = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    if not cols or not np.isfinite(X).all():
        raise ValueError(f"Invalid {name} coefficient matrix")
    return X, cols


def stratified_draws(codes, selected, endpoints, reps, seed):
    """Fixed within-cell counts; return Monte Carlo means and mobility.

    Sample only cells containing selected knots; endpoint sums avoid allocating
    a full atlas-sized boolean mask on every replicate.
    """
    codes, selected, y = np.asarray(codes), np.asarray(selected, bool), np.asarray(endpoints, float)
    if len(codes) != len(selected) or len(y) != len(codes) or not selected.any():
        raise ValueError("Invalid fixed-selection null inputs")
    if reps < 2 or not np.isfinite(y).all():
        raise ValueError("Need >=2 draws and finite outcomes")
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(codes[order]))]
    cells = np.split(order, starts[1:])
    sizes = np.array([len(g) for g in cells])
    random, fixed = [], []
    for g in cells:
        k = int(selected[g].sum())
        if k == len(g):
            fixed.extend(g)
        elif k:
            random.append((g, k))
    fixed = np.asarray(fixed, int)
    total = y[fixed].sum(axis=0)
    draws = np.empty((reps, y.shape[1]))
    rng = np.random.default_rng(seed)
    for r in range(reps):
        value = total.copy()
        for g, k in random:
            value += y[rng.choice(g, k, replace=False)].sum(axis=0)
        draws[r] = value / selected.sum()
    return draws, dict(n_strata=len(cells), median_cell=float(np.median(sizes)),
                       q25_cell=float(np.quantile(sizes, .25)),
                       singleton_prop=float(np.mean(sizes == 1)),
                       selected_fixed=len(fixed), selected_movable=int(selected.sum() - len(fixed)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


# %%
# Stage implementation

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler






def fit_variant(X, split, crossings, scale, prevalence, rank_rule, k, seed, batch):
    tr, ca, te = (split[x] for x in ("train_idx", "val_idx", "test_idx"))
    keep = feature_mask(X[tr], prevalence)
    if keep.sum() < k and rank_rule == "fixed":
        raise ValueError("Filtered dimension below fixed k; do not silently change rank")
    scaler = StandardScaler(with_std=scale)
    Z = scaler.fit_transform(X[tr][:, keep])
    pca = PCA(n_components=k if rank_rule == "fixed" else .99,
              svd_solver="randomized" if rank_rule == "fixed" else "full",
              random_state=seed).fit(Z)
    train_pc = pca.transform(Z)
    pc_crossing = [float(spearmanr(train_pc[:, j], crossings[tr]).statistic)
                   for j in range(min(3, pca.n_components_))]
    del Z, train_pc

    def evaluate(idx):
        sse, norms = np.empty(len(idx)), np.empty(len(idx))
        for start in range(0, len(idx), batch):
            stop = min(start + batch, len(idx))
            z = scaler.transform(X[idx[start:stop]][:, keep])
            residual = z - pca.inverse_transform(pca.transform(z))
            # Match historical residual_batch accumulation in float64.
            sse[start:stop] = np.sum(residual.astype(float) ** 2, axis=1)
            norms[start:stop] = np.log1p(np.sum(z.astype(float) ** 2, axis=1))
        return sse, norms

    ca_sse, ca_norm = evaluate(ca)
    te_sse, te_norm = evaluate(te)
    return dict(test_score=conditional_score(ca_sse, ca_norm, te_sse, te_norm),
                test_log_norm=te_norm, test_raw_sse=te_sse,
                retained_features=keep, k=np.array([pca.n_components_]),
                evr=np.array([pca.explained_variance_ratio_.sum()]),
                train_pc_crossing_rho=np.asarray(pc_crossing))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=_direct_default_root())
    p.add_argument("--data-dir", type=Path, default=_direct_default_data())
    p.add_argument("--out", type=Path)
    p.add_argument("--rank-rule", choices=("fixed", "evr99"), default="fixed")
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--run", action="store_true", default=True)
    p.add_argument("--check-only", action="store_false", dest="run",
                   help="Check inputs without fitting or simulating")
    a = p.parse_args()
    root = a.output_dir
    out = a.out or root / f"41_theta_preprocessing_direct_{a.rank_rule}"
    atlas, split, inputs = load_atlas_split(root)
    test_meta = atlas.iloc[split["test_idx"]]
    ids = test_meta[ID].to_numpy(str)
    width, support, phenotype_path = load_outcomes(root, ids)
    paths = [root / "23_anomaly_score_baselines" / "checkpoints" / f"{safe(v)}_test_scores.npz"
             for v in VIEWS[:-1]]
    scores = {}
    for v, path in zip(VIEWS[:-1], paths):
        with np.load(path, allow_pickle=False) as q:
            scores[v] = np.asarray(q["test_conditional_percentile_100"], float)
            if v == "Theta":
                k = int(q["pca_k"][0])
                baseline = dict(test_score=scores[v], test_log_norm=q["test_log_norm"].copy(),
                                test_raw_sse=q["test_raw_sse"].copy(), k=np.array([k]))
        if len(scores[v]) != len(ids) or not np.isfinite(scores[v]).all():
            raise ValueError(f"Invalid cached {v} scores")
    source = a.data_dir / "theta_upto15.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    old_summary = root / "23_anomaly_score_baselines" / "score_family_external_phenotypes.csv"
    frame = pd.read_csv(old_summary)
    ref = frame.loc[(frame.method == "conditional_percentile_100") &
                    (frame.family == "No Khovanov >=3/4"), "n"]
    if len(ref) != 1:
        raise ValueError("Cannot identify no-Khovanov reference cardinality")
    n = int(ref.iloc[0])
    old_mask = top_n(c3([scores[v] for v in VIEWS[:-1]]), ids, n)
    print(f"Ready: train={len(split['train_idx'])}, test={len(ids)}, frozen n={n}, cached Theta k={k}.")
    print("Reuses Alexander/Jones/HOMFLY scores and the historical standardized Theta reference.")
    print("New fits: centered only; prevalence >=1%; prevalence >=5%.")
    if a.rank_rule == "evr99":
        print("EVR99 also fits a new standardized reference; distinct from the historical fixed-k benchmark.")
    if not a.run:
        print("Preflight only. Add --run to execute missing variants.")
        return
    manifest = dict(stage=41, rank_rule=a.rank_rule, k_fixed=k, n_selected=n,
                    seed=a.seed, batch_size=a.batch_size, norm_bins=100,
                    sklearn=sklearn.__version__, numpy=np.__version__,
                    input_hashes=fingerprint(inputs + paths + [source, phenotype_path, old_summary,
                        Path(__file__)]))
    freeze_manifest(out, manifest)
    variants = {"centered_only": (False, 0.), "prevalence_01": (True, .01),
                "prevalence_05": (True, .05)}
    if a.rank_rule == "evr99":
        variants = {"standardized_evr99": (True, 0.), **variants}
    results = {"historical_standardized_fixed_k": baseline}
    X = None
    for name, (scale, prevalence) in variants.items():
        path = out / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as q:
                if not np.array_equal(q["test_ids"], ids):
                    raise ValueError("Cached variant test-ID mismatch")
                result = {key: q[key].copy() for key in q.files}
            print("Reused", name)
        else:
            if X is None:
                print("Reading Theta only; no other raw representations loaded.", flush=True)
                X, cols = read_view(source, "Theta", atlas[ID].to_numpy(str))
            print("Fitting", name, flush=True)
            result = fit_variant(X, split, atlas.number_of_crossings.to_numpy(), scale,
                                 prevalence, a.rank_rule, k, a.seed, a.batch_size)
            result["test_ids"] = ids
            result["feature_names"] = np.asarray(cols, str)
            temp = out / f"{name}.tmp.npz"
            np.savez_compressed(temp, **result)
            temp.replace(path)
        results[name] = result
    rows, selected = [], []
    reference_name = "historical_standardized_fixed_k" if a.rank_rule == "fixed" else "standardized_evr99"
    masks = {}
    for name, result in results.items():
        masks[name] = top_n(c3([scores[v] for v in VIEWS[:3]] + [result["test_score"]]), ids, n)
    for name, result in results.items():
        mask = masks[name]
        boundary = c3([scores[v] for v in VIEWS[:3]] + [result["test_score"]])
        cutoff = boundary[mask].min()
        row = dict(variant=name, comparison_reference=reference_name, n=int(mask.sum()),
                   kh_diagonal_mean=float(width[mask].mean()), kh_diagonal_ge_3_prop=float((width[mask]>=3).mean()),
                   kh_diagonal_ge_4_prop=float((width[mask]>=4).mean()), kh_support_mean=float(support[mask].mean()),
                   alternating_prop=float(test_meta.is_alternating.to_numpy()[mask].mean()),
                   k=int(result["k"][0]), evr=float(result.get("evr", [np.nan])[0]),
                   retained_features=int(result["retained_features"].sum()) if "retained_features" in result else np.nan,
                   jaccard_historical=overlap(ids[mask], ids[old_mask])["jaccard"],
                   jaccard_reference=overlap(ids[mask], ids[masks[reference_name]])["jaccard"],
                   boundary_tied_total=int((boundary == cutoff).sum()),
                   boundary_tied_selected=int(((boundary == cutoff) & mask).sum()),
                   rho_raw_sse_log_norm=float(spearmanr(result["test_raw_sse"], result["test_log_norm"]).statistic),
                   rho_conditional_log_norm=float(spearmanr(result["test_score"], result["test_log_norm"]).statistic))
        for j, rho in enumerate(result.get("train_pc_crossing_rho", [])):
            row[f"train_pc{j+1}_crossing_rho"] = float(rho)
        rows.append(row)
        selected.extend(dict(variant=name, **{ID: x}) for x in ids[mask])
    pd.DataFrame(rows).to_csv(out / "theta_sensitivity_summary.csv", index=False)
    pd.DataFrame(selected).to_csv(out / "theta_sensitivity_selected_ids.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
