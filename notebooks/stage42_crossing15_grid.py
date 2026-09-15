# %% [markdown]
# Stage 42 — Freeze Stage-31 selections; reuse 2-bin null and simulate finer grids.
#
# Standalone version for paper_run.ipynb, matching the Stage-39 workflow.
# Copy this file into the same notebooks/ folder as Stage 39 and run:
#   %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage42_crossing15_grid.py"
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
from sklearn.preprocessing import StandardScaler



METRICS = ("mean_kh_diag", "P_kh_diag_ge3", "P_kh_diag_ge4", "mean_kh_support")


def build_bins(root, data, out, members):
    atlas_path = unique_file(root, "final_hard_regime_atlas.parquet")
    atlas = pd.read_parquet(atlas_path).reset_index(drop=True)
    if atlas[ID].duplicated().any():
        raise ValueError("Duplicate atlas IDs")
    ids = atlas[ID].astype(str).to_numpy()
    cross = atlas.number_of_crossings.to_numpy(int)
    tr, ca, te = (np.flatnonzero(cross <= 13), np.flatnonzero(cross == 14), np.flatnonzero(cross == 15))
    if not np.array_equal(ids[te], members[ID].to_numpy(str)):
        raise ValueError("Stage-31 members are not in canonical atlas test order")
    result = {}
    for v in VIEWS:
        checkpoint = out / f"{safe(v)}_norm_bins.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as q:
                if not np.array_equal(q["test_ids"], ids[te]):
                    raise ValueError("Norm checkpoint IDs changed")
                result[v] = q["test_norm_bin"].copy()
            print("Reused norm codes:", v, flush=True)
            continue
        print("Reconstructing norms only:", v, flush=True)
        X, _ = read_view(data / SOURCES[v][0], v, ids)
        scaler = StandardScaler().fit(X[tr])

        def norms(idx):
            values = np.empty(len(idx))
            for start in range(0, len(idx), 8192):
                stop = min(start + 8192, len(idx))
                z = scaler.transform(X[idx[start:stop]]).astype(np.float32)
                values[start:stop] = np.log1p(np.sum(z.astype(float) ** 2, axis=1))
            return values

        _, result[v] = norm_bins(norms(ca), norms(te), 100)
        temp = out / f"{safe(v)}_norm_bins.tmp.npz"
        np.savez_compressed(temp, test_ids=ids[te], test_norm_bin=result[v])
        temp.replace(checkpoint)
        del X
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=_direct_default_root())
    p.add_argument("--data-dir", type=Path, default=_direct_default_data())
    p.add_argument("--norm-cache", type=Path, help="NPZ: test_ids and {safe_view}_test_norm_bin_100")
    p.add_argument("--out", type=Path)
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026091442)
    p.add_argument("--run", action="store_true", default=True)
    p.add_argument("--check-only", action="store_false", dest="run",
                   help="Check inputs without fitting or simulating")
    a = p.parse_args()
    if a.reps < 2:
        raise ValueError("--reps must be >=2")
    out = a.out or a.output_dir / "42_crossing15_grid_direct"
    old = a.output_dir / "31_crossing_number_extrapolation"
    member_path = old / "crossing15_selected_members.csv"
    members = pd.read_csv(member_path)
    ids = members[ID].astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids) or not members.number_of_crossings.eq(15).all():
        raise ValueError("Invalid frozen Stage-31 test universe")
    flag = members["no_khovanov_ge3of4"].astype(str).str.lower()
    if not flag.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid frozen selection flag")
    selected = flag.isin(["true", "1"]).to_numpy()
    d = members.khovanov_F_q_minus_2t_diagonal_count.to_numpy(float)
    support = members.khovanov_F_support_size.to_numpy(float)
    endpoints = np.column_stack([d, d >= 3, d >= 4, support])
    if not selected.any() or not np.isfinite(endpoints).all():
        raise ValueError("Empty selection or nonfinite endpoints")
    observed = endpoints[selected].mean(axis=0)
    inputs = [member_path, Path(__file__)]
    old_null = old / "crossing15_noKh_fixed_selection_null.parquet"
    old_summary = old / "crossing15_fixed_selection_null_summary.csv"
    for path in (old_null, old_summary):
        if path.exists():
            inputs.append(path)
    if a.norm_cache:
        inputs.append(a.norm_cache)
    else:
        if a.data_dir is None:
            raise ValueError("Supply --norm-cache from the Stage-31 runtime, or --data-dir to rebuild norms only")
        inputs += [unique_file(a.output_dir, "final_hard_regime_atlas.parquet")]
        inputs += [a.data_dir / SOURCES[v][0] for v in VIEWS]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    print(f"Frozen test n={len(members):,}, selected n={selected.sum()}, observed mean width={observed[0]:.6f}")
    print("Grid: 2, 5, 10, 20; stratify on alternation, |signature| and all five norm codes.")
    print("2-bin null:", "reuse stored draws" if old_null.exists() else "new simulation")
    if not a.run:
        print("Preflight only. Add --run to calculate the trajectory; no PCA will be fitted.")
        return
    freeze_manifest(out, dict(stage=42, reps_new=a.reps, seed=a.seed, bins=[2, 5, 10, 20],
                              calibration_norm_bins=100, selected_n=int(selected.sum()),
                              input_hashes=fingerprint(inputs), sklearn=sklearn.__version__, numpy=np.__version__))
    if a.norm_cache:
        with np.load(a.norm_cache, allow_pickle=False) as q:
            if not np.array_equal(q["test_ids"].astype(str), ids):
                raise ValueError("Norm-cache IDs must exactly match frozen Stage-31 member order")
            bins = {v: q[f"{safe(v)}_test_norm_bin_100"].copy() for v in VIEWS}
    else:
        bins = build_bins(a.output_dir, a.data_dir, out, members)
    for v, b in bins.items():
        if len(b) != len(ids) or not np.isfinite(b).all() or (b < 0).any() or (b >= 100).any() or not np.equal(b, np.floor(b)).all():
            raise ValueError(f"Invalid 100-bin codes: {v}")
    rows = []
    for resolution in (2, 5, 10, 20):
        strata = pd.DataFrame(dict(alternating=members.is_alternating.to_numpy(),
                                   abs_sigma=members.signature.abs().to_numpy()))
        for v in VIEWS:
            strata[v] = coarsen(bins[v], resolution)
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(strata), sort=False)
        # Only two inexpensive draws to get mobility diagnostics; never a refit.
        _, diag = stratified_draws(codes, selected, endpoints, 2, a.seed)
        if resolution == 2 and old_summary.exists():
            prior = pd.read_csv(old_summary)
            ref = prior.loc[prior.analysis == "noKh_external_mean_diagonal"]
            if len(ref) != 1 or not np.isclose(ref.observed.iloc[0], observed[0]):
                raise ValueError("Stored Stage-31 summary disagrees with frozen members")
            for key in ("n_strata", "median_cell", "selected_fixed", "selected_movable"):
                if not np.isclose(ref[f"strata_{key}"].iloc[0], diag[key]):
                    raise ValueError(f"Reconstructed 2-bin strata disagree with Stage 31: {key}. Stop and audit norms.")
        checkpoint = out / f"null_bins_{resolution}.csv"
        if resolution == 2 and old_null.exists():
            archived = pd.read_parquet(old_null)
            if "n" in archived and not archived["n"].eq(selected.sum()).all():
                raise ValueError("Stored Stage-31 draws have a different selection cardinality")
            draws = archived[list(METRICS)].to_numpy(float)
            source = "reused_stage31_draws"
        elif checkpoint.exists():
            draws = pd.read_csv(checkpoint)[list(METRICS)].to_numpy(float)
            source = "reused_stage42_draws"
        else:
            print(f"Simulating bins={resolution}: fixed={diag['selected_fixed']}, movable={diag['selected_movable']}", flush=True)
            draws, _ = stratified_draws(codes, selected, endpoints, a.reps, a.seed + resolution)
            temp = out / f"null_bins_{resolution}.tmp.csv"
            pd.DataFrame(draws, columns=METRICS).to_csv(temp, index=False)
            temp.replace(checkpoint)
            source = "new_stage42_draws"
        if len(draws) < 2 or not np.isfinite(draws).all():
            raise ValueError("Invalid null checkpoint")
        for j, metric in enumerate(METRICS):
            vals = draws[:, j]
            rows.append(dict(bins_per_view=resolution, metric=metric, n=int(selected.sum()),
                             observed=observed[j], null_mean=float(vals.mean()),
                             observed_minus_null=float(observed[j] - vals.mean()),
                             null_mean_mcse=float(vals.std(ddof=1) / np.sqrt(len(vals))),
                             null_q025=float(np.quantile(vals, .025)), null_q975=float(np.quantile(vals, .975)),
                             empirical_p_upper=float((1 + np.sum(vals >= observed[j])) / (len(vals) + 1)),
                             draws=len(vals), source=source, **diag))
    summary = pd.DataFrame(rows)
    summary["p_holm_within_resolution_4"] = summary.groupby("bins_per_view")["empirical_p_upper"].transform(lambda s: holm(s.to_numpy()))
    summary["p_holm_entire_grid_16"] = holm(summary.empirical_p_upper.to_numpy())
    summary.to_csv(out / "crossing15_grid_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
