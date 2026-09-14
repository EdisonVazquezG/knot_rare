# %% [markdown]
# Stage 36 — Euler-characteristic cancellation E(K) as an exploratory endpoint
#
# Revision request (Ernesto, "Consider one further homological quantity"):
#
#   h_ij = dim_Q Kh^{i,j}(K; Q),      c_j = sum_i (-1)^i h_ij
#
#   E(K) = (1/2) ( sum_{i,j} h_ij  -  sum_j |c_j| )  =  sum_j min(a_j, b_j) >= 0
#
# where a_j and b_j are the total ranks in even and odd homological degree at
# quantum degree j.  The identity follows from a + b - |a - b| = 2 min(a, b).
# E counts cancellation lost when passing from homology to its graded Euler
# characteristic.  It uses actual homology ranks, not standardized coordinates.
#
# IMPORTANT ENCODING POINT
#   E is defined over Q, so only the archived F_ (free-part) coordinates enter.
#   The T2_ / T4_ torsion families are correctly EXCLUDED here: torsion does not
#   contribute to the Euler characteristic.  This is the one endpoint in the
#   paper where restricting to F_ is mathematically right rather than a caveat.
#
# The reviewer also required: "Its normalization must be matched to the stored
# Jones data."  Section 4 below does not assume a convention.  It enumerates
# candidate conventions and keeps only one that exactly reproduces the stored
# Jones coefficients for essentially every knot.  If none does, the stage FAILS
# LOUDLY rather than returning a number computed under an unverified convention.
#
# Fresh-session usage:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage36_euler_cancellation_endpoint.py
#
# Optional environment overrides:
#   KNOT_DATA_DIR, KNOT_OUTPUT_DIR
#   STAGE36_SELECTION_COL   name of the frozen no-Khovanov selection indicator
#   STAGE36_NULL_REPS       default 2000

from __future__ import annotations

from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Bootstrap (same pattern as Stage 35 so alignment is identical)
# ---------------------------------------------------------------------------
import consensus_hardness as ch
from consensus_hardness.io import load_table
from consensus_hardness.preprocessing import (
    add_knot_ids,
    filter_crossings,
    canonicalize_mirrors_by_signature,
)

DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"

DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))

if not ROOT.exists() or not DATA_DIR.exists():
    raise FileNotFoundError("Mount Drive or set KNOT_DATA_DIR / KNOT_OUTPUT_DIR.")

OUT = ROOT / "36_euler_cancellation_endpoint"
OUT.mkdir(parents=True, exist_ok=True)

ID_COL = "knot_id_base"
BATCH = 20000
NULL_REPS = int(os.environ.get("STAGE36_NULL_REPS", "2000"))
SEED = 20260913

print("=" * 78)
print("STAGE 36 — Euler-characteristic cancellation endpoint")
print("=" * 78)
print("DATA_DIR   =", DATA_DIR)
print("OUTPUT_DIR =", ROOT)

# ---------------------------------------------------------------------------
# 1. Frozen phenotype table: knot identifiers and external covariates
# ---------------------------------------------------------------------------
def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {ROOT}")
    return hits[0]


# Same sources and ordering as Stage 34, so selection labels align row by row.
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(
    find_one("complete_mathematical_phenotype_atlas.parquet")
).reset_index(drop=True)

atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(
        phenotype, on=ID_COL, how="left", validate="one_to_one"
    )

ids = atlas[ID_COL].to_numpy()
n_knots = len(ids)
print(f"Frozen universe: {n_knots:,} knots")

# ---------------------------------------------------------------------------
# 2. Archived Khovanov table, aligned exactly as in Stage 35
# ---------------------------------------------------------------------------
kh_path = DATA_DIR / "even_KH_upto17.pkl"
if not kh_path.exists():
    raise FileNotFoundError(kh_path)

print("Loading archived Khovanov table:", kh_path)
raw = load_table(kh_path)
clean = add_knot_ids(raw, id_col="knot_id", mirror_symbol="!")
clean = filter_crossings(clean, min_crossings=3, max_crossings=15)
canonical = canonicalize_mirrors_by_signature(
    clean,
    id_col="knot_id",
    base_col=ID_COL,
    signature_col="signature",
    mirror_symbol="!",
)
idx = canonical.set_index(ID_COL, drop=False)
missing = [x for x in ids if x not in idx.index]
if missing:
    raise RuntimeError(f"Archived Khovanov table misses frozen IDs, e.g. {missing[:5]}")

kh = idx.loc[ids].reset_index(drop=True)
if not np.array_equal(kh[ID_COL].astype(str).to_numpy(), ids):
    raise RuntimeError("Khovanov alignment failure")

F_COLS = [c for c in kh.columns if str(c).startswith("F_")]
print(f"Free-part coordinates: {len(F_COLS)}")
if not F_COLS:
    raise RuntimeError("No F_ coordinates found.")

# ---------------------------------------------------------------------------
# 3. Parse (q, t) gradings and build even/odd aggregation matrices
#
# Column convention observed in the archive: 'F_q-35_t-13', 'F_q-15_t0', ...
# We parse explicitly and refuse to guess.
# ---------------------------------------------------------------------------
GRADING_RE = re.compile(r"^F_q(-?\d+)_t(-?\d+)$")

q_of, t_of = [], []
unparsed = []
for col in F_COLS:
    m = GRADING_RE.match(str(col))
    if m is None:
        unparsed.append(str(col))
        continue
    q_of.append(int(m.group(1)))
    t_of.append(int(m.group(2)))

if unparsed:
    raise RuntimeError(
        f"{len(unparsed)} F_ coordinates do not match 'F_q<int>_t<int>', "
        f"e.g. {unparsed[:5]}. Stage 36 will not guess a grading convention."
    )

q_of = np.asarray(q_of, dtype=int)
t_of = np.asarray(t_of, dtype=int)

q_values = np.unique(q_of)
q_index = {q: k for k, q in enumerate(q_values)}
n_q = len(q_values)

# Quantum degrees of unreduced Khovanov homology of a KNOT are odd.
odd_q_prop = float(np.mean(q_of % 2 != 0))
print(f"Distinct quantum degrees: {n_q}  (range {q_values.min()}..{q_values.max()})")
print(f"Proportion of coordinates with odd q: {odd_q_prop:.6f}")

# M_even[c, k] = 1 iff coordinate c has quantum degree q_values[k] and even t.
M_even = np.zeros((len(F_COLS), n_q), dtype=np.float32)
M_odd = np.zeros((len(F_COLS), n_q), dtype=np.float32)
for c, (q, t) in enumerate(zip(q_of, t_of)):
    if t % 2 == 0:
        M_even[c, q_index[q]] = 1.0
    else:
        M_odd[c, q_index[q]] = 1.0

# ---------------------------------------------------------------------------
# 4. Compute a_j, b_j, c_j, E(K); audit integrality and the defining identity
# ---------------------------------------------------------------------------
A = np.zeros((n_knots, n_q), dtype=np.float64)  # even homological degree
B = np.zeros((n_knots, n_q), dtype=np.float64)  # odd homological degree

noninteger = 0
negative = 0
for start in range(0, n_knots, BATCH):
    stop = min(start + BATCH, n_knots)
    arr = kh.iloc[start:stop][F_COLS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("Non-finite entry in archived F_ coordinates.")
    noninteger += int(np.sum(np.abs(arr - np.round(arr)) > 1e-9))
    negative += int(np.sum(arr < 0))
    A[start:stop] = arr @ M_even
    B[start:stop] = arr @ M_odd

if noninteger:
    raise RuntimeError(
        f"{noninteger} archived F_ entries are not integers. E(K) requires "
        "homology ranks; this table appears transformed or standardized."
    )
if negative:
    raise RuntimeError(f"{negative} archived F_ entries are negative; not ranks.")

C = A - B                      # graded Euler characteristic coefficients c_j
TOTAL = (A + B).sum(axis=1)    # sum_{i,j} h_ij
E = np.minimum(A, B).sum(axis=1)

identity_lhs = 0.5 * (TOTAL - np.abs(C).sum(axis=1))
identity_max_dev = float(np.max(np.abs(identity_lhs - E)))
if identity_max_dev > 1e-9:
    raise RuntimeError(f"E identity violated; max deviation {identity_max_dev:g}")
if float(E.min()) < 0:
    raise RuntimeError("E(K) is negative somewhere; impossible by construction.")

print(f"Identity check passed (max deviation {identity_max_dev:.3g})")
print(
    f"E(K): mean {E.mean():.4f}, median {np.median(E):.1f}, "
    f"max {E.max():.0f}, zero for {float(np.mean(E == 0)):.4%} of knots"
)

# ---------------------------------------------------------------------------
# 5. Normalization audit against the stored Jones data
#
# For unreduced Khovanov homology of a knot,
#     sum_{i,j} (-1)^i q^j dim Kh^{i,j}(K) = (q + q^{-1}) V_K(q^2),
# where V_K is the Jones polynomial normalized so that V_unknot = 1.  If the
# stored Jones columns 'J<n>' are the coefficients a_n of t^n in V_K(t), then
#     c_m = a_{(m-1)/2} + a_{(m+1)/2}      for odd m.
#
# We do not assume this.  We test it, its mirror variant and sign variants, and
# require one candidate to reproduce the stored coefficients exactly for
# essentially every knot.
# ---------------------------------------------------------------------------
jones_path = DATA_DIR / "Jones_upto17_MIRRORS.csv"
if not jones_path.exists():
    raise FileNotFoundError(
        f"{jones_path} not found; the normalization audit cannot be skipped."
    )

print("Loading archived Jones table:", jones_path)
j_raw = load_table(jones_path)
j_clean = add_knot_ids(j_raw, id_col="knot_id", mirror_symbol="!")
j_clean = filter_crossings(j_clean, min_crossings=3, max_crossings=15)
j_canon = canonicalize_mirrors_by_signature(
    j_clean,
    id_col="knot_id",
    base_col=ID_COL,
    signature_col="signature",
    mirror_symbol="!",
)
j_idx = j_canon.set_index(ID_COL, drop=False)
j_missing = [x for x in ids if x not in j_idx.index]
if j_missing:
    raise RuntimeError(f"Jones table misses frozen IDs, e.g. {j_missing[:5]}")
jones = j_idx.loc[ids].reset_index(drop=True)

JONES_RE = re.compile(r"^J(-?\d+)$")
j_cols, j_exps = [], []
for col in jones.columns:
    m = JONES_RE.match(str(col))
    if m is not None:
        j_cols.append(col)
        j_exps.append(int(m.group(1)))
j_exps = np.asarray(j_exps, dtype=int)
print(f"Jones coordinates: {len(j_cols)} (range {j_exps.min()}..{j_exps.max()})")

# Audit on a random subsample first; a full pass only for the winning candidate.
rng_audit = np.random.default_rng(SEED)
sample_n = min(20000, n_knots)
sample_idx = np.sort(rng_audit.choice(n_knots, size=sample_n, replace=False))

J_sample = jones.iloc[sample_idx][j_cols].to_numpy(dtype=np.float64)
C_sample = C[sample_idx]


def predict_c(J, exps, q_vals, *, mirror: bool, sign: int) -> np.ndarray:
    """Predicted c_j from Jones coefficients under a candidate convention."""
    lookup = {int(e): k for k, e in enumerate(exps)}
    out = np.zeros((J.shape[0], len(q_vals)), dtype=np.float64)
    for k, m in enumerate(q_vals):
        if m % 2 == 0:
            continue  # unreduced Khovanov of a knot lives in odd q
        total = np.zeros(J.shape[0], dtype=np.float64)
        for n in (((m - 1) // 2), ((m + 1) // 2)):
            nn = -n if mirror else n
            col = lookup.get(int(nn))
            if col is not None:
                total += J[:, col]
        out[:, k] = sign * total
    return out


candidates = {
    "t_standard": dict(mirror=False, sign=+1),
    "t_standard_negated": dict(mirror=False, sign=-1),
    "t_inverse": dict(mirror=True, sign=+1),
    "t_inverse_negated": dict(mirror=True, sign=-1),
}

audit_rows = []
for label, kwargs in candidates.items():
    pred = predict_c(J_sample, j_exps, q_values, **kwargs)
    exact = np.all(np.abs(pred - C_sample) < 1e-9, axis=1)
    audit_rows.append(
        {
            "candidate": label,
            "mirror": kwargs["mirror"],
            "sign": kwargs["sign"],
            "n_audited": int(sample_n),
            "exact_match_n": int(exact.sum()),
            "exact_match_prop": float(exact.mean()),
            "max_abs_deviation": float(np.max(np.abs(pred - C_sample))),
        }
    )

audit = pd.DataFrame(audit_rows).sort_values("exact_match_prop", ascending=False)
audit.to_csv(OUT / "jones_euler_characteristic_convention_audit.csv", index=False)
print("\nNormalization audit (subsample):")
print(audit.to_string(index=False))

# ---------------------------------------------------------------------------
# 5b. Per-knot convention resolution
#
# The sign-flipped candidates are expected to fail outright: they test whether
# the overall sign convention is right.  The two remaining candidates differ
# only by t -> 1/t, i.e. by which member of a mirror pair is stored.  Since
# V_Kbar(t) = V_K(1/t), a knot whose Khovanov row and Jones row describe
# OPPOSITE representatives will reproduce under `t_inverse` rather than
# `t_standard`.  Amphichiral-in-Jones knots (V(t) = V(1/t)) reproduce under both
# and act as an internal consistency check.
#
# So the correct verification is per knot, not global: every knot must be
# reproduced by at least one of the two orientation-consistent candidates.
# Failing that, the normalization itself is wrong and the stage aborts.
# ---------------------------------------------------------------------------
sign_ok = all(
    float(r["exact_match_prop"]) < 1e-6
    for r in audit_rows
    if r["sign"] == -1
)
if not sign_ok:
    raise RuntimeError(
        "A sign-negated candidate reproduced the stored Jones coefficients. "
        "The assumed Euler-characteristic sign convention is wrong; inspect "
        "jones_euler_characteristic_convention_audit.csv before proceeding."
    )

match_std = np.zeros(n_knots, dtype=bool)
match_inv = np.zeros(n_knots, dtype=bool)
full_maxdev = 0.0
for start in range(0, n_knots, BATCH):
    stop = min(start + BATCH, n_knots)
    Jb = jones.iloc[start:stop][j_cols].to_numpy(dtype=np.float64)
    Cb = C[start:stop]
    for label, store in (("t_standard", match_std), ("t_inverse", match_inv)):
        dev = np.abs(predict_c(Jb, j_exps, q_values, **candidates[label]) - Cb)
        store[start:stop] = np.all(dev < 1e-9, axis=1)
        full_maxdev = max(full_maxdev, float(dev.min(initial=np.inf)))

resolved = match_std | match_inv
full_prop = float(resolved.mean())
both = match_std & match_inv
opposite_rep = match_inv & ~match_std

print(f"\nReproduced by at least one orientation convention: "
      f"{int(resolved.sum()):,}/{n_knots:,} ({full_prop:.6%})")
print(f"  same representative as Jones      : {int((match_std & ~both).sum()):,}")
print(f"  OPPOSITE representative to Jones  : {int(opposite_rep.sum()):,} "
      f"({opposite_rep.mean():.4%})")
print(f"  reproduced by both (V(t)=V(1/t))  : {int(both.sum()):,} "
      f"({both.mean():.4%})")

if full_prop < 0.999:
    unresolved = np.flatnonzero(~resolved)[:10]
    raise RuntimeError(
        f"Only {full_prop:.4%} of knots are reproduced by either orientation "
        "convention. The Euler-characteristic normalization itself is not "
        f"verified. Example unresolved IDs: {list(ids[unresolved])}"
    )

winner = "per_knot_orientation_resolved"

# Representative-consistency audit between the two archived tables.
rep_audit = pd.DataFrame(
    {
        ID_COL: ids,
        "reproduced_t_standard": match_std,
        "reproduced_t_inverse": match_inv,
        "jones_khovanov_same_representative": match_std,
        "orientation_ambiguous": both,
    }
)
rep_audit.to_csv(OUT / "khovanov_jones_representative_audit.csv", index=False)

print(
    "\n[!] ENCODING FINDING: the archived Khovanov and Jones tables do not "
    "always store the same member of a mirror pair.\n"
    "    This does NOT affect W_F or E(K): both are mirror-invariant, since "
    "mirroring sends Kh^{i,j} to Kh^{-i,-j},\n"
    "    which negates the q-2t diagonal set (preserving its cardinality) and "
    "sends a_j, b_j to a_-j, b_-j (preserving sum_j min).\n"
    "    It CAN affect per-coordinate standardized Khovanov norms. See "
    "khovanov_jones_representative_audit.csv."
)

# ---------------------------------------------------------------------------
# 6. Per-knot export
# ---------------------------------------------------------------------------
per_knot = pd.DataFrame(
    {
        ID_COL: ids,
        "E_cancellation": E.astype(np.int64),
        "total_free_rank": TOTAL.astype(np.int64),
        "n_occupied_q_degrees": (A + B > 0).sum(axis=1).astype(np.int64),
        "sum_abs_euler_coeff": np.abs(C).sum(axis=1).astype(np.int64),
    }
)
per_knot["E_fraction_of_rank"] = np.where(
    per_knot["total_free_rank"] > 0,
    2.0 * per_knot["E_cancellation"] / per_knot["total_free_rank"],
    0.0,
)
per_knot.to_csv(OUT / "euler_cancellation_per_knot.csv", index=False)
print("\nWrote per-knot E(K) for", f"{n_knots:,}", "knots")

# ---------------------------------------------------------------------------
# 7. Exploratory association with the withheld-Khovanov selection
#
# Same stratified, fixed-cardinality construction used elsewhere in the paper:
# alternation, exact |sigma|, crossing number, two amplitude bins per
# representation norm, plus the withheld Khovanov norm.  Mobility is reported
# with the effect, as required by the revision.
# ---------------------------------------------------------------------------
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")

hard_saved = pd.read_csv(
    find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str}
)
id_to_pos = pd.Series(np.arange(n_knots, dtype=np.int64), index=ids).to_dict()

votes = np.zeros(n_knots, dtype=np.uint8)
for name in NO_KHOVANOV:
    view_ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    m = np.zeros(n_knots, dtype=bool)
    m[[id_to_pos[x] for x in view_ids]] = True
    votes += m.astype(np.uint8)

selected = votes >= 3
n_selected = int(selected.sum())
print(f"\nCanonical no-Khovanov 3-of-4 selection: {n_selected}")
if n_selected != 220:
    raise RuntimeError(
        f"Canonical no-Kh selection n={n_selected}, expected 220. "
        "The frozen artifacts do not match the manuscript."
    )

if True:

    def _col(frame, names):
        for nm in names:
            if nm in frame.columns:
                return frame[nm].to_numpy()
        raise RuntimeError(f"None of {names} present in phenotype table.")

    alternating = _col(phenotype, ("is_alternating", "alternating"))
    sigma = _col(phenotype, ("signature", "sigma"))
    crossings = _col(
        phenotype, ("number_of_crossings", "crossing_number", "crossings")
    )

    norm_frames = [phenotype] + ([atlas] if atlas is not phenotype else [])
    norm_cols = []
    norm_source = {}
    for frame in norm_frames:
        for c in frame.columns:
            if c in norm_source:
                continue
            name = str(c)
            if not re.search(r"norm", name, flags=re.I):
                continue
            if frame[c].dtype.kind not in "fi":
                continue
            # mean_/max_ summaries are deterministic functions of the five
            # per-representation norms; including them only over-stratifies.
            if re.match(r"^(mean|max|min)_", name, flags=re.I):
                continue
            norm_cols.append(c)
            norm_source[c] = frame
    print(f"Amplitude columns used ({len(norm_cols)}): {norm_cols}")

    def coarsen(values, target=2):
        v = np.asarray(values, dtype=float)
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(len(v))
        return np.minimum((ranks * target / len(v)).astype(int), target - 1)

    # E is bounded by total rank, so "less cancellation" could mean either
    # smaller homology or proportionally less cancellation at comparable size.
    # Only the cancelled fraction 2E/total isolates cancellation itself.
    total_rank = per_knot["total_free_rank"].to_numpy(dtype=float)
    frac = per_knot["E_fraction_of_rank"].to_numpy(dtype=float)

    ENDPOINTS = {
        "mean_E": E,
        "mean_total_free_rank": total_rank,
        "mean_cancelled_fraction_of_rank": frac,
    }

    def build_sampler(n_bins: int):
        """Exact common cells at a given amplitude resolution."""
        strata = pd.DataFrame(
            {
                "alternating": alternating.astype(str),
                "abs_sigma": np.abs(sigma).astype(int).astype(str),
                "crossings": np.asarray(crossings).astype(int).astype(str),
            }
        )
        if n_bins > 1:
            for c in norm_cols:
                strata[f"{c}_bin{n_bins}"] = coarsen(
                    norm_source[c][c].to_numpy(), n_bins
                ).astype(str)

        codes, _ = pd.factorize(pd.MultiIndex.from_frame(strata), sort=False)
        codes = codes.astype(np.int32)

        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        starts = np.r_[0, 1 + np.flatnonzero(np.diff(cs))]
        stops = np.r_[starts[1:], len(order)]

        random_groups, fixed_blocks, sizes = [], [], []
        for a, b in zip(starts, stops):
            cell = order[a:b]
            k = int(selected[cell].sum())
            sizes.append(len(cell))
            if k == 0:
                continue
            if k == len(cell):
                fixed_blocks.append(cell)
            else:
                random_groups.append((cell, k))
        fixed_idx = (
            np.concatenate(fixed_blocks) if fixed_blocks else np.empty(0, dtype=int)
        )
        return random_groups, fixed_idx, np.asarray(sizes)

    def two_sided_p(vals, obs):
        """Empirical two-sided p; this association sits in the lower tail."""
        vals = np.asarray(vals, dtype=float)
        n = len(vals)
        p_up = (1 + np.sum(vals >= obs)) / (1 + n)
        p_lo = (1 + np.sum(vals <= obs)) / (1 + n)
        return float(min(1.0, 2 * min(p_up, p_lo))), float(p_lo), float(p_up)

    # -----------------------------------------------------------------------
    # Amplitude-resolution trajectory.
    #
    # Table 3 of the manuscript shows that the W_F contrast collapses from
    # 0.557 at two bins to 0.023 at twenty.  It would be incoherent to report
    # that for the primary endpoint and then quote a coarse-grid total-rank
    # difference without the same check.  We therefore run every endpoint over
    # the same grid and report mobility at each resolution.
    # -----------------------------------------------------------------------
    BIN_GRID = (20, 10, 5, 2, 1)

    rows = []
    for n_bins in BIN_GRID:
        random_groups, fixed_idx, sizes = build_sampler(n_bins)
        n_fixed = int(len(fixed_idx))
        n_movable = int(sum(k for _, k in random_groups))
        print(
            f"  bins/view {n_bins:>2}: strata {len(sizes):>6}  "
            f"median cell {np.median(sizes):>7.1f}  "
            f"fixed {n_fixed:>3}  movable {n_movable:>3}"
        )

        rng = np.random.default_rng(SEED)
        draws = {k: np.empty(NULL_REPS, dtype=float) for k in ENDPOINTS}
        for rep in range(NULL_REPS):
            mask = np.zeros(n_knots, dtype=bool)
            mask[fixed_idx] = True
            for cell, k in random_groups:
                mask[rng.choice(cell, size=k, replace=False)] = True
            for name, vec in ENDPOINTS.items():
                draws[name][rep] = vec[mask].mean()

        for name, vec in ENDPOINTS.items():
            obs = float(vec[selected].mean())
            d = draws[name]
            sd = float(d.std(ddof=1))
            p_two, p_lo, p_up = two_sided_p(d, obs)
            rows.append(
                {
                    "bins_per_view": n_bins,
                    "metric": name,
                    "observed": obs,
                    "null_mean": float(d.mean()),
                    "observed_minus_null": obs - float(d.mean()),
                    "null_sd": sd,
                    "z_null": (obs - float(d.mean())) / sd if sd > 0 else np.nan,
                    "p_lower": p_lo,
                    "p_upper": p_up,
                    "p_two_sided": p_two,
                    "null_reps": NULL_REPS,
                    "strata_n_strata": int(len(sizes)),
                    "strata_median_cell": float(np.median(sizes)),
                    "strata_selected_fixed": n_fixed,
                    "strata_selected_movable": n_movable,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "euler_cancellation_resolution_trajectory.csv", index=False
    )

    print("\nResolution trajectory (observed - null):")
    print(
        result.pivot(
            index="bins_per_view", columns="metric", values="observed_minus_null"
        ).to_string()
    )
    print("\nMovable selected knots by resolution:")
    print(
        result.drop_duplicates("bins_per_view")
        .set_index("bins_per_view")[
            ["strata_median_cell", "strata_selected_fixed", "strata_selected_movable"]
        ]
        .to_string()
    )


    per_knot.loc[selected].to_csv(
        OUT / "euler_cancellation_selected_knots.csv", index=False
    )
    sel_col = "canonical_no_khovanov_3of4"

# ---------------------------------------------------------------------------
# 8. Gate
# ---------------------------------------------------------------------------
gate = {
    "stage": 36,
    "status": "EULER_CANCELLATION_VERIFIED",
    "n_knots": int(n_knots),
    "n_free_coordinates": int(len(F_COLS)),
    "torsion_families_excluded": True,
    "torsion_exclusion_reason": "torsion does not contribute to the Euler characteristic",
    "identity_max_deviation": identity_max_dev,
    "verified_jones_convention": winner,
    "full_atlas_reproduction_prop": full_prop,
    "jones_khovanov_opposite_representative_prop": float(opposite_rep.mean()),
    "jones_khovanov_orientation_ambiguous_prop": float(both.mean()),
    "mean_E": float(E.mean()),
    "prop_E_zero": float(np.mean(E == 0)),
    "selection_column": sel_col,
}
(OUT / "stage36_gate.json").write_text(json.dumps(gate, indent=2))
print("\nSaved Stage 36 to:", OUT)