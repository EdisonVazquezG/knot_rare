# %% [markdown]
# Stage 37 — Mirror-symmetrized score S_sym
#
# Revision request (Ernesto, "Build mirror invariance into a score"):
#
#     S_sym(K) = ( S(K) + S(Kbar) ) / 2      =>      S_sym(K) = S_sym(Kbar),
#
# "because mirroring twice returns the original knot.  This proves the desired
#  symmetry of the score.  It does not depend on a percentage of membership
#  recovery."
#
# Stage 31B established, descriptively, that membership is representative
# dependent (Jaccard 0.435-0.460 across random assignments).  This stage
# removes the dependence by construction instead of measuring it.
#
# TWO DESIGN POINTS
#
# 1. We average SCORES, never feature vectors.  The reviewer's warning applies
#    directly here: the gated Theta mirror action is a coefficient sign flip, so
#    averaging Theta feature vectors would annihilate the entire view.
#
# 2. The conditional percentile is a rank within a reference population, not a
#    pointwise function, so "evaluate S at Kbar" needs care.  We score the UNION
#    population of 2N points (canonical representative and opposite
#    representative of every knot).  Relabelling K <-> Kbar leaves that union
#    set-wise unchanged, so the two percentiles merely swap and their mean is
#    invariant.  The symmetry is therefore exact, and section 6 asserts it
#    numerically as an implementation check on the proof.
#
# Khovanov is never mirrored and never used to build a score; the stored F_-part
# diagonal count is evaluated afterwards as the withheld external endpoint.
#
# Prerequisite: Stage 31A must have gated the Theta mirror action.  This stage
# HARD-STOPS otherwise, exactly as Stage 31B does.
#
# Fresh-session usage:
#   %run .../stage37_mirror_symmetrized_score.py
#
# Environment overrides:
#   KNOT_DATA_DIR, KNOT_OUTPUT_DIR, STAGE37_PCA_SEED, STAGE37_BATCH_SIZE

from __future__ import annotations

from pathlib import Path
import gc
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import consensus_hardness as ch
from consensus_hardness.io import load_table
from consensus_hardness.preprocessing import (
    add_knot_ids,
    filter_crossings,
    canonicalize_mirrors_by_signature,
)
from consensus_hardness.representations import split_metadata_and_features

# ---------------------------------------------------------------------------
# 0. Bootstrap
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants"
)
DEFAULT_ROOT = (
    DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"
)

DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(
    os.environ.get("KNOT_OUTPUT_DIR", str(globals().get("OUTPUT_DIR", DEFAULT_ROOT)))
)
if not ROOT.exists() or not DATA_DIR.exists():
    raise FileNotFoundError("Mount Drive or set KNOT_DATA_DIR / KNOT_OUTPUT_DIR.")

OUT = ROOT / "37_mirror_symmetrized_score"
CHECKPOINTS = OUT / "checkpoints"
for d in (OUT, CHECKPOINTS):
    d.mkdir(parents=True, exist_ok=True)

CONFIG = globals().get("CONFIG", ch.canonical_run_config())
ID_COL = "knot_id_base"
MIRROR_SYMBOL = "!"
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
VIEW_SPECS = {
    "Alexander": ("Alexander_upto17.csv", ["A"]),
    "Jones": ("Jones_upto17_MIRRORS.csv", ["J"]),
    "HOMFLY-PT": ("HomflyPt_upto15_MIRRORS.csv", ["a"]),
    "Theta": ("theta_upto15.csv", ["T"]),
}
FIXED_K = {"Alexander": 4, "Jones": 10, "HOMFLY-PT": 32, "Theta": 10}
N_NORM_BINS = 100
PCA_SEED = int(os.environ.get("STAGE37_PCA_SEED", "20261123"))
BATCH_SIZE = int(os.environ.get("STAGE37_BATCH_SIZE", "8192"))

print("=" * 78)
print("STAGE 37 — mirror-symmetrized score")
print("=" * 78)


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {ROOT}")
    return hits[0]


def safe_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# 1. Stage 31A gate: the Theta mirror action must be empirically established
# ---------------------------------------------------------------------------
GATE_PATH = ROOT / "31A_theta_mirror_action_audit" / "theta_mirror_action_gate.json"
if not GATE_PATH.exists():
    raise FileNotFoundError(
        "Stage 31A gate is missing. Run stage31A_theta_mirror_action_audit.py "
        f"first; expected {GATE_PATH}"
    )
gate31a = json.loads(GATE_PATH.read_text())
if gate31a.get("theta_mirror_action_for_stage31B") != "coefficient_sign_flip":
    raise RuntimeError(
        "Stage 37 only implements a Theta coefficient sign flip, matching the "
        "Stage 31A gate. Gate currently says: "
        f"{gate31a.get('theta_mirror_action_for_stage31B')}"
    )
print("Theta mirror action (gated by Stage 31A): coefficient sign flip")

# ---------------------------------------------------------------------------
# 2. Frozen atlas, canonical selection and withheld Khovanov endpoints
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(
    drop=True
)
phenotype = pd.read_parquet(
    find_one("complete_mathematical_phenotype_atlas.parquet")
).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(
        phenotype, on=ID_COL, how="left", validate="one_to_one"
    )

FROZEN_IDS = atlas[ID_COL].to_numpy().astype(str)
N = len(FROZEN_IDS)
TAIL_N = int(round(0.01 * N))
print(f"Frozen atlas: {N:,} knots; per-view tail {TAIL_N:,}")

def _pick(frame, names, what):
    for c in names:
        if c in frame.columns:
            return c
    raise RuntimeError(
        f"No {what} column found. Tried {names}. "
        f"Available: {[c for c in frame.columns if 'kh' in str(c).lower()]}"
    )


# Candidate names kept identical to Stage 34 so the endpoints match the audit.
kh_diag_col = _pick(
    phenotype,
    (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    ),
    "Khovanov diagonal-count",
)
kh_support_col = _pick(
    phenotype,
    ("khovanov_support_size", "kh_support_size", "khovanov_f_support_size"),
    "Khovanov support-size",
)
print(f"Khovanov endpoints: {kh_diag_col}, {kh_support_col}")
W_F = phenotype[kh_diag_col].to_numpy(float)
KH_SUPPORT = phenotype[kh_support_col].to_numpy(float)

hard_saved = pd.read_csv(
    find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str}
)
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=FROZEN_IDS).to_dict()
canonical_votes = np.zeros(N, dtype=np.uint8)
for nm in NO_KHOVANOV:
    view_ids = hard_saved.loc[hard_saved["invariant"].eq(nm), ID_COL].astype(str)
    m = np.zeros(N, dtype=bool)
    m[[id_to_pos[x] for x in view_ids]] = True
    canonical_votes += m.astype(np.uint8)
canonical_selected = canonical_votes >= 3
if int(canonical_selected.sum()) != 220:
    raise RuntimeError(
        f"Canonical no-Kh selection n={int(canonical_selected.sum())}, expected 220."
    )
print(f"Canonical no-Khovanov 3-of-4 selection: {int(canonical_selected.sum())}")


# ---------------------------------------------------------------------------
# 3. Scoring helpers
# ---------------------------------------------------------------------------
def sse_and_norm(Z_iter, scaler, pca, n_rows):
    """Squared reconstruction error and squared norm in standardized space."""
    sse = np.empty(n_rows, dtype=np.float64)
    nrm = np.empty(n_rows, dtype=np.float64)
    pos = 0
    for Xb in Z_iter:
        z = scaler.transform(np.asarray(Xb, dtype=np.float32)).astype(np.float32)
        retained = pca.transform(z)
        residual = z - pca.inverse_transform(retained)
        k = len(z)
        sse[pos : pos + k] = np.sum(residual.astype(np.float64) ** 2, axis=1)
        nrm[pos : pos + k] = np.sum(z.astype(np.float64) ** 2, axis=1)
        pos += k
    if pos != n_rows:
        raise RuntimeError("row count mismatch while scoring")
    return sse, nrm


def conditional_percentile(sse, norm_sq, n_bins=N_NORM_BINS):
    """Average-rank percentile of SSE within equal-frequency log-norm bins.

    This replicates Stage 31B's full_atlas_conditional() exactly, including the
    order of np.unique relative to the infinite endpoints and the division by
    len(idx) rather than len(idx)+1.  Both details matter: the tail is taken
    globally, so any per-bin rescaling that differs with bin size shifts which
    knots enter the tail.

    Section 4 calls this twice per view, on two different populations:
      * the canonical rows alone, reproducing the frozen score; and
      * the union of canonical and opposite representatives, which is what
        makes the symmetrized score exactly invariant.
    """
    sse = np.asarray(sse, dtype=np.float64)
    lognorm = np.log1p(np.asarray(norm_sq, dtype=np.float64))

    edges = np.quantile(lognorm, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    bins = np.searchsorted(edges[1:-1], lognorm, side="right").astype(np.int32)

    cond = np.empty(len(sse), dtype=np.float64)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        cond[idx] = rankdata(sse[idx], method="average") / len(idx)
    return cond, bins


def stable_top_tail(scores):
    scores = np.asarray(scores, dtype=float)
    if len(scores) != N or not np.isfinite(scores).all():
        raise ValueError("Invalid full-atlas score vector")
    order = np.lexsort((FROZEN_IDS, scores))
    mask = np.zeros(N, dtype=bool)
    mask[order[-TAIL_N:]] = True
    return mask


def jaccard(a, b):
    u = int(np.sum(a | b))
    return float(np.sum(a & b) / u) if u else float("nan")


# ---------------------------------------------------------------------------
# 4. Per view: canonical score, opposite-representative score, symmetrized score
# ---------------------------------------------------------------------------
S_canonical = {}       # canonical rows, ranked within the union population
S_canonical_only = {}  # canonical rows, ranked within the canonical atlas
S_opposite = {}
S_symmetric = {}
view_rows = []

for view_index, name in enumerate(NO_KHOVANOV):
    ckpt = CHECKPOINTS / f"{safe_name(name)}_sym_scores_v2.npz"
    if ckpt.exists():
        with np.load(ckpt, allow_pickle=False) as z:
            S_canonical[name] = z["s_can"]
            S_opposite[name] = z["s_opp"]
            S_symmetric[name] = z["s_sym"]
            S_canonical_only[name] = z["s_can_only"]
        print(f"[{name}] loading checkpoint")
        view_rows.append(json.loads((CHECKPOINTS / f"{safe_name(name)}_meta.json").read_text()))
        continue

    filename, prefixes = VIEW_SPECS[name]
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    print("\n" + "=" * 78)
    print(f"VIEW: {name} | {path.name}")
    print("=" * 78)

    raw = load_table(path)
    cleaned = add_knot_ids(raw, id_col="knot_id", mirror_symbol=MIRROR_SYMBOL)
    cleaned = filter_crossings(cleaned, min_crossings=3, max_crossings=15)
    canonical = canonicalize_mirrors_by_signature(
        cleaned,
        id_col="knot_id",
        base_col=ID_COL,
        signature_col="signature",
        mirror_symbol=MIRROR_SYMBOL,
    )
    canon_index = canonical.set_index(ID_COL, drop=False)
    missing = [x for x in FROZEN_IDS if x not in canon_index.index]
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} frozen IDs missing, e.g. {missing[:5]}")
    canon_rows = canon_index.loc[FROZEN_IDS].reset_index(drop=True)

    _, ids_check, Xcanon, feat_cols = split_metadata_and_features(
        canon_rows, feature_prefixes=prefixes, id_col=ID_COL
    )
    if not np.array_equal(ids_check.astype(str), FROZEN_IDS):
        raise RuntimeError(f"{name}: feature alignment failed")
    Xcanon = np.asarray(Xcanon, dtype=np.float32)

    # --- opposite representative, per the Stage 31A/31B mirror actions --------
    opposite_missing = 0
    Xopp = None
    theta_sign_flip = False

    if name in ("Jones", "HOMFLY-PT"):
        canonical_id_by_base = pd.Series(
            canon_rows["knot_id_clean"].astype(str).to_numpy(),
            index=canon_rows[ID_COL].astype(str).to_numpy(),
        )
        current_id = cleaned[ID_COL].astype(str).map(canonical_id_by_base)
        alt_rows = cleaned.loc[
            current_id.notna()
            & cleaned["knot_id_clean"].astype(str).ne(current_id.astype(str))
        ].copy()
        alt_rows = alt_rows.drop_duplicates(ID_COL, keep="first").set_index(
            ID_COL, drop=False
        )
        has_alt = pd.Index(FROZEN_IDS).isin(alt_rows.index)
        opposite_missing = int((~has_alt).sum())
        opp_rows = canon_rows.copy()
        if has_alt.any():
            opp_rows.loc[has_alt, feat_cols] = alt_rows.loc[
                FROZEN_IDS[has_alt], feat_cols
            ].to_numpy()
        Xopp = (
            opp_rows[feat_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        del canonical_id_by_base, current_id, alt_rows, opp_rows
        action = "archived_opposite_row"
    elif name == "Theta":
        # Sign flip applied batchwise below; never materialise a second matrix.
        theta_sign_flip = True
        action = "coefficient_sign_flip_stage31A_gate"
    elif name == "Alexander":
        # Delta is mirror invariant: the opposite representative has the same
        # stored coefficients, so S(K) = S(Kbar) and S_sym = S for this view.
        action = "unchanged"
    else:
        raise ValueError(name)

    del raw, cleaned, canonical, canon_rows
    gc.collect()

    # --- fit the score function once, on the canonical population ------------
    scaler = StandardScaler(copy=True)
    scaler.fit(Xcanon)
    Zfit = scaler.transform(Xcanon).astype(np.float32)
    pca = PCA(
        n_components=int(FIXED_K[name]),
        svd_solver="randomized",
        random_state=PCA_SEED + view_index,
    )
    pca.fit(Zfit)
    del Zfit
    gc.collect()

    def batches(X, flip=False):
        for start in range(0, len(X), BATCH_SIZE):
            b = X[start : start + BATCH_SIZE]
            yield (-b) if flip else b

    sse_can, nrm_can = sse_and_norm(batches(Xcanon), scaler, pca, N)
    if theta_sign_flip:
        sse_opp, nrm_opp = sse_and_norm(batches(Xcanon, flip=True), scaler, pca, N)
    elif Xopp is not None:
        sse_opp, nrm_opp = sse_and_norm(batches(Xopp), scaler, pca, N)
        del Xopp
    else:  # Alexander
        sse_opp, nrm_opp = sse_can.copy(), nrm_can.copy()

    del Xcanon
    gc.collect()

    # --- union-population percentile -----------------------------------------
    # (a) Canonical-only ranking: reproduces the frozen full-atlas score, and is
    #     the correct baseline against which to audit this implementation.
    s_can_only, _ = conditional_percentile(sse_can, nrm_can)

    # (b) Union ranking: the reference population is the 2N set of canonical and
    #     opposite representatives, which is invariant under relabelling.
    sse_union = np.concatenate([sse_can, sse_opp])
    nrm_union = np.concatenate([nrm_can, nrm_opp])
    pct_union, _ = conditional_percentile(sse_union, nrm_union)

    s_can = pct_union[:N]
    s_opp = pct_union[N:]
    s_sym = 0.5 * (s_can + s_opp)

    S_canonical[name] = s_can
    S_canonical_only[name] = s_can_only
    S_opposite[name] = s_opp
    S_symmetric[name] = s_sym
    np.savez_compressed(
        ckpt, s_can=s_can, s_opp=s_opp, s_sym=s_sym, s_can_only=s_can_only
    )

    row = {
        "view": name,
        "mirror_action": action,
        "k99": int(FIXED_K[name]),
        "archived_opposite_missing": opposite_missing,
        "identical_representative_prop": float(np.mean(sse_can == sse_opp)),
        "spearman_can_vs_opp": float(
            pd.Series(s_can).corr(pd.Series(s_opp), method="spearman")
        ),
    }
    view_rows.append(row)
    (CHECKPOINTS / f"{safe_name(name)}_meta.json").write_text(json.dumps(row, indent=2))
    print(
        f"[{name}] action={action}  identical rows="
        f"{row['identical_representative_prop']:.4%}  "
        f"rho(can,opp)={row['spearman_can_vs_opp']:.4f}"
    )
    del sse_can, sse_opp, nrm_can, nrm_opp, sse_union, nrm_union, pct_union
    gc.collect()

view_audit = pd.DataFrame(view_rows)
view_audit.to_csv(OUT / "symmetrized_view_audit.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Consensus selections
# ---------------------------------------------------------------------------
def consensus(score_dict):
    votes = np.zeros(N, dtype=np.uint8)
    for nm in NO_KHOVANOV:
        votes += stable_top_tail(score_dict[nm]).astype(np.uint8)
    return votes >= 3


sel_sym = consensus(S_symmetric)
sel_recomputed_canonical = consensus(S_canonical_only)

print(f"\nSymmetrized 3-of-4 selection:            {int(sel_sym.sum())}")
print(f"Recomputed canonical (implementation):   {int(sel_recomputed_canonical.sum())}")
baseline_jaccard = jaccard(sel_recomputed_canonical, canonical_selected)
print(
    f"  agreement with frozen canonical 220:   Jaccard {baseline_jaccard:.4f}"
)
if baseline_jaccard < 0.90:
    raise RuntimeError(
        f"This implementation reproduces the frozen canonical selection only at "
        f"Jaccard {baseline_jaccard:.4f} (n={int(sel_recomputed_canonical.sum())} "
        f"vs 220). The symmetrized comparison is not interpretable until the "
        "baseline matches: any difference would confound S_sym with a scoring "
        "discrepancy."
    )

# ---------------------------------------------------------------------------
# 6. Exact-symmetry assertion
#
# Relabelling every knot by its opposite representative swaps s_can and s_opp.
# S_sym must be bitwise unchanged.  This asserts the proof against the code.
# ---------------------------------------------------------------------------
S_swapped = {
    nm: 0.5 * (S_opposite[nm] + S_canonical[nm]) for nm in NO_KHOVANOV
}
max_dev = max(
    float(np.max(np.abs(S_swapped[nm] - S_symmetric[nm]))) for nm in NO_KHOVANOV
)
sel_swapped = consensus(S_swapped)
if max_dev > 0.0 or not np.array_equal(sel_swapped, sel_sym):
    raise RuntimeError(
        f"S_sym is not invariant under relabelling (max deviation {max_dev:g}). "
        "The implementation does not realise the construction."
    )
print(f"\nExact-symmetry assertion passed (max deviation {max_dev:g}).")
print("S_sym(K) = S_sym(Kbar) holds by construction, not by recovery rate.")

# ---------------------------------------------------------------------------
# 7. Withheld Khovanov endpoints, descriptive
# ---------------------------------------------------------------------------
def endpoints(mask, label):
    return {
        "selection": label,
        "n": int(mask.sum()),
        "jaccard_with_frozen_canonical": jaccard(mask, canonical_selected),
        "frozen_canonical_recovered_prop": (
            float(np.sum(mask & canonical_selected) / canonical_selected.sum())
        ),
        "mean_W_F": float(W_F[mask].mean()),
        "P_W_F_ge3": float(np.mean(W_F[mask] >= 3)),
        "P_W_F_ge4": float(np.mean(W_F[mask] >= 4)),
        "mean_kh_support": float(KH_SUPPORT[mask].mean()),
        "alternating_prop": (
            float(np.mean(phenotype.loc[mask, "is_alternating"].to_numpy().astype(bool)))
            if "is_alternating" in phenotype.columns
            else np.nan
        ),
    }


summary = pd.DataFrame(
    [
        endpoints(canonical_selected, "frozen_canonical_3of4"),
        endpoints(sel_recomputed_canonical, "recomputed_canonical_3of4"),
        endpoints(sel_sym, "mirror_symmetrized_3of4"),
    ]
)
summary.to_csv(OUT / "symmetrized_selection_summary.csv", index=False)

pd.DataFrame(
    {
        ID_COL: FROZEN_IDS[sel_sym],
        "W_F": W_F[sel_sym],
        "in_frozen_canonical": canonical_selected[sel_sym],
    }
).to_csv(OUT / "symmetrized_selected_knot_ids.csv", index=False)

np.savez_compressed(
    OUT / "symmetrized_scores.npz",
    **{f"S_sym__{safe_name(nm)}": S_symmetric[nm] for nm in NO_KHOVANOV},
    **{f"S_can__{safe_name(nm)}": S_canonical[nm] for nm in NO_KHOVANOV},
    selection_symmetrized=sel_sym,
    selection_frozen_canonical=canonical_selected,
)

gate = {
    "stage": 37,
    "construction": "S_sym(K) = (S(K) + S(Kbar)) / 2 on the union reference population",
    "symmetry": "exact by construction; asserted numerically",
    "max_symmetry_deviation": max_dev,
    "khovanov_mirrored": False,
    "khovanov_used_in_scoring": False,
    "n_symmetrized_selection": int(sel_sym.sum()),
    "jaccard_with_frozen_canonical": jaccard(sel_sym, canonical_selected),
    "theta_mirror_action": gate31a.get("theta_mirror_action_for_stage31B"),
}
(OUT / "stage37_gate.json").write_text(json.dumps(gate, indent=2))

print("\nPer-view audit:")
print(view_audit.to_string(index=False))
print("\nSelection comparison:")
print(summary.to_string(index=False))
print("\nSaved Stage 37 to:", OUT)