# %% [markdown]
# Stage 40 — Audit cached selections and notebook evidence; no model fitting.
#
# Standalone version for paper_run.ipynb, matching the Stage-39 workflow.
# Copy this file into the same notebooks/ folder as Stage 39 and run:
#   %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage40_revision_audit_v2.py"
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


# Artifact identity includes its producing stage. Identical filenames in
# Stage-38 sensitivity runs are separate experiments, not ambiguous copies.
ARTIFACT_PATHS = {
    "score_selected_test_ids.csv":
        "23_anomaly_score_baselines/score_selected_test_ids.csv",
    "size_matched_selected_ids.csv":
        "23B_size_matched_score_sensitivity/size_matched_selected_ids.csv",
    "size_matched_score_jaccard.csv":
        "23B_size_matched_score_sensitivity/size_matched_score_jaccard.csv",
    "conditional_no_khovanov_selected_n31.csv":
        "23D_conditional_no_khovanov_withheld_n31/conditional_no_khovanov_selected_n31.csv",
    "final_hard_regime_atlas.parquet":
        "17_final_paper_outputs/final_hard_regime_atlas.parquet",
    "complete_mathematical_phenotype_atlas.parquet":
        "20_mathematical_phenotype/complete_mathematical_phenotype_atlas.parquet",
    "heldout_ae_seed_0.npz":
        "07_heldout_ae_target_free/scores/heldout_ae_seed_0.npz",
}
RESOLVED_ARTIFACTS = {}


def primary_artifact(root, name, required=True):
    root = Path(root)
    if name not in ARTIFACT_PATHS:
        raise ValueError(f"No producing stage declared for {name}")
    canonical = root / ARTIFACT_PATHS[name]
    # Support explicitly flattened exports, never a recursive substitution
    # from a different stage. The canonical stage always takes precedence.
    exported = root / name
    chosen = canonical if canonical.is_file() else exported if exported.is_file() else None
    RESOLVED_ARTIFACTS[name] = {
        "artifact": name,
        "expected_path": str(canonical),
        "selected_path": str(chosen) if chosen is not None else "",
        "resolution": "canonical_stage" if chosen == canonical else
                      "flat_export" if chosen is not None else "missing",
    }
    if chosen is None and required:
        raise FileNotFoundError(
            f"Missing primary artifact: {canonical}. A flat export at {exported} "
            "is also supported; files from other experiments are not substituted."
        )
    return chosen


def unique_file(root, name):
    return primary_artifact(root, name, required=True)


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
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    print("STAGE 40 v2 — explicit artifact paths")
    print("Executing:", Path(__file__).resolve())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=_direct_default_root())
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    a.out = a.out or a.output_dir / "40_revision_audit_direct"
    a.out.mkdir(parents=True, exist_ok=True)
    notebook_path = REPO / "notebooks" / "paper_run.ipynb"
    if notebook_path.exists():
        nb = json.loads(notebook_path.read_text())
    else:
        print(f"Notebook not found at {notebook_path}; auditing numeric caches only.")
        nb = {"cells": []}
    inventory = []
    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        refs = re.findall(r"%run\s+[\"']?([^\n\"']+\.py)", source)
        outputs = []
        errors = []
        for o in cell.get("outputs", []):
            value = o.get("text", o.get("data", {}).get("text/plain", []))
            outputs.append(value if isinstance(value, str) else "".join(value))
            if o.get("output_type") == "error":
                errors.append(o.get("ename", "error") + ": " + o.get("evalue", ""))
        if i in (167, 168):
            # Stages 23C/23D live inline rather than in referenced scripts.
            (a.out / f"cell_{i:03d}_output.txt").write_text("\n\n".join(outputs))
            (a.out / f"cell_{i:03d}_source.py").write_text(source)
        if refs:
            for ref in refs:
                name = Path(ref.strip()).name
                inventory.append(dict(cell_index=i, script=name,
                                      script_present=(REPO / "notebooks" / name).exists(),
                                      execution_count=cell.get("execution_count"),
                                      saved_text_chars=sum(map(len, outputs)), errors="; ".join(errors)))
            # Text evidence only; retain original notebook untouched, without image blobs.
            text = "\n\n".join(outputs)
            if text:
                (a.out / f"cell_{i:03d}_output.txt").write_text(text)
    with (a.out / "notebook_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_index", "script", "script_present", "execution_count", "saved_text_chars", "errors"])
        w.writeheader()
        w.writerows(inventory)
    missing = [r for r in inventory if not r["script_present"]]
    print(f"Notebook stage calls: {len(inventory)}; referenced scripts absent: {len(missing)}")
    for row in missing:
        print(f"  cell {row['cell_index']}: {row['script']} (saved output: {row['saved_text_chars']} chars)")
    status = dict(notebook=str(notebook_path), missing_script_calls=missing,
                  drive_cache_status="not inspected; no --output-dir provided")
    if a.output_dir:
        status["drive_cache_status"] = audit_caches(a.output_dir, a.out)
    pd.DataFrame(list(RESOLVED_ARTIFACTS.values())).to_csv(
        a.out / "resolved_artifact_paths.csv", index=False
    )
    (a.out / "audit_status.json").write_text(json.dumps(status, indent=2) + "\n")
    for filename in ("conditional_cohort_phenotypes.csv", "conditional_cohort_comparison.csv"):
        table_path = a.out / filename
        if table_path.exists():
            print("\n" + filename)
            print(pd.read_csv(table_path).to_string(index=False))
    print("Saved evidence:", a.out)


def audit_caches(root, out):
    import numpy as np
    import pandas as pd


    expected = [
        "size_matched_selected_ids.csv", "size_matched_score_jaccard.csv",
        "score_selected_test_ids.csv", "crossing15_selected_members.csv",
        "crossing15_noKh_fixed_selection_null.parquet",
        "heldout_view_ablation_n31_summary.csv", "heldout_view_ablation_n31_selected_ids.csv",
        "multiplier_gcm_calibration_summary.csv", "score_geometry_pairs.csv",
        "conditional_no_khovanov_selected_n31.csv",
    ]
    availability = []
    for name in expected:
        hits = sorted(root.rglob(name))
        chosen = primary_artifact(root, name, required=False) if name in ARTIFACT_PATHS else None
        availability.append(dict(artifact=name, count=len(hits), paths="; ".join(map(str, hits)),
                                 selected_path=str(chosen) if chosen is not None else ""))
    pd.DataFrame(availability).to_csv(out / "cache_inventory.csv", index=False)
    rows = []
    for design, filename in (("original_variable_size", "score_selected_test_ids.csv"),
                             ("equal_size", "size_matched_selected_ids.csv")):
        selected_path = primary_artifact(root, filename, required=False)
        if selected_path is None:
            continue
        df = pd.read_csv(selected_path)
        if df.duplicated(["family", "method", ID]).any():
            raise ValueError(f"Duplicate selected IDs in {selected_path}")
        for family, f in df.groupby("family"):
            sets = {m: set(g[ID].astype(str)) for m, g in f.groupby("method")}
            if design == "equal_size" and len({len(s) for s in sets.values()}) != 1:
                raise ValueError(f"Unequal cardinalities in {family}")
            for ma, mb in combinations(sorted(sets), 2):
                rows.append(dict(design=design, family=family, method_a=ma, method_b=mb,
                                 **overlap(sets[ma], sets[mb])))
    if rows:
        pd.DataFrame(rows).to_csv(out / "jaccard_with_cardinalities.csv", index=False)

    # Independently reconstruct fixed-n lists from cached scores to catch stale
    # selected-ID CSVs; this ranks arrays only and never refits any model.
    checkpoints = root / "23_anomaly_score_baselines" / "checkpoints"
    paths = [checkpoints / f"{safe(v)}_test_scores.npz" for v in VIEWS]
    verification = []
    if all(path.exists() for path in paths):
        atlas, split, _ = load_atlas_split(root)
        ids = atlas.iloc[split["test_idx"]][ID].to_numpy(str)
        scores = {}
        for v, path in zip(VIEWS, paths):
            with np.load(path, allow_pickle=False) as q:
                scores[v] = {m: np.asarray(q[f"test_{m}"], float) for m in METHODS}
            if any(len(s) != len(ids) for s in scores[v].values()):
                raise ValueError(f"Wrong cached score length for {v}")
        summary = pd.read_csv(root / "23_anomaly_score_baselines" / "score_family_external_phenotypes.csv")
        selected_path = primary_artifact(root, "size_matched_selected_ids.csv", required=False)
        saved = pd.read_csv(selected_path) if selected_path is not None else None
        reconstructed = []
        for family, old_family, views in (("All 5 C3", "All 5 >=3/5", VIEWS),
                                          ("No Khovanov C3", "No Khovanov >=3/4", VIEWS[:-1])):
            ref = summary.loc[(summary.method == METHODS[-1]) & (summary.family == old_family), "n"]
            if len(ref) != 1:
                raise ValueError(f"Missing reference cardinality: {old_family}")
            n = int(ref.iloc[0])
            for m in METHODS:
                mask = top_n(c3([scores[v][m] for v in views]), ids, n)
                reconstructed.extend(dict(family=family, method=m, **{ID: x}) for x in ids[mask])
                if saved is not None:
                    old = saved.loc[(saved.family == family) & (saved.method == m), ID].astype(str)
                    verification.append(dict(family=family, method=m,
                                             **overlap(ids[mask], old), identical=set(ids[mask]) == set(old)))
        pd.DataFrame(reconstructed).to_csv(out / "reconstructed_equal_size_ids.csv", index=False)
        reconstructed_frame = pd.DataFrame(reconstructed)
        for family, group in reconstructed_frame.groupby("family"):
            sets = {m: set(g[ID]) for m, g in group.groupby("method")}
            for ma, mb in combinations(sorted(sets), 2):
                rows.append(dict(design="equal_size_reconstructed", family=family,
                                 method_a=ma, method_b=mb, **overlap(sets[ma], sets[mb])))
        pd.DataFrame(rows).to_csv(out / "jaccard_with_cardinalities.csv", index=False)
        if verification:
            pd.DataFrame(verification).to_csv(out / "selected_id_cache_verification.csv", index=False)
            if not all(r["identical"] for r in verification):
                raise RuntimeError("Stale selected-ID cache detected; see selected_id_cache_verification.csv")
        # Historical inline Stage 23D uses DIRECT conditional percentiles and
        # ascending test-row index at ties, unlike rank-transformed Stage 23B/33.
        direct = np.sort(np.column_stack([scores[v][METHODS[-1]] for v in VIEWS[:-1]]), axis=1)[:, 1]
        direct_mask = np.zeros(len(ids), bool)
        direct_mask[np.lexsort((np.arange(len(ids)), -direct))[:n]] = True
        reranked_mask = top_n(c3([scores[v][METHODS[-1]] for v in VIEWS[:-1]]), ids, n)

        width, support, _ = load_outcomes(root, ids)
        comparisons = [dict(comparison="23D_direct_vs_23B_reranked", **overlap(ids[direct_mask], ids[reranked_mask]))]
        cohorts = []
        for name, mask in (("23D_direct_index_ties", direct_mask), ("23B_33_reranked_ID_ties", reranked_mask)):
            cohorts.append(dict(cohort=name, n=int(mask.sum()), mean_width=float(width[mask].mean()),
                                mean_support=float(support[mask].mean())))
        selected_path = primary_artifact(root, "conditional_no_khovanov_selected_n31.csv", required=False)
        if selected_path is not None:
            archived = pd.read_csv(selected_path)[ID].astype(str)
            comparisons.append(dict(comparison="23D_reconstructed_vs_archived", **overlap(ids[direct_mask], archived)))
        pd.DataFrame(comparisons).to_csv(out / "conditional_cohort_comparison.csv", index=False)
        pd.DataFrame(cohorts).to_csv(out / "conditional_cohort_phenotypes.csv", index=False)
    return dict(artifact_inventory="cache_inventory.csv", jaccard_pairs=len(rows),
                score_reconstruction_available=all(path.exists() for path in paths))


if __name__ == "__main__":
    main()
