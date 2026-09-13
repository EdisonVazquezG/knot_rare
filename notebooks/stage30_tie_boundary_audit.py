# %% [markdown]
# Stage 30 — Deterministic 1% boundary/tie audit
#
# Audits the top-tail selection rule used by the frozen raw and conditional
# analyses. It reports the exact cutoff tie block, verifies the stable-ID
# deterministic tie break, and randomizes ONLY boundary ties to quantify how
# much the final multiview consensus/phenotype could change.
#
# Standalone: reads frozen artifacts below OUTPUT_DIR / corrected run.

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
if not ROOT.exists():
    raise FileNotFoundError(ROOT)
OUT = ROOT / "30_tie_boundary_audit"
OUT.mkdir(parents=True, exist_ok=True)

N_RANDOM = int(os.environ.get("STAGE30_TIE_RANDOM_REPS", "1000"))
SEED = int(os.environ.get("STAGE30_TIE_SEED", "20260830"))
ID_COL = "knot_id_base"
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")

def safe_name(name):
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")

def find_one(filename):
    found = sorted(ROOT.rglob(filename))
    if not found:
        raise FileNotFoundError(filename)
    return found[0]

atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")
N = len(atlas)
ids = atlas[ID_COL].astype(str).to_numpy()

S_COL = next(c for c in ("s_invariant_qc", "s_invariant", "s") if c in atlas)
G = np.abs(atlas[S_COL].to_numpy(float) - atlas["signature"].to_numpy(float))
NONALT = atlas["is_alternating"].to_numpy(int) == 0
KH_DIAG_COL = next(
    c for c in ("khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count")
    if c in phenotype
)
KH_DIAG = phenotype[KH_DIAG_COL].to_numpy(float)

id_to_pos = pd.Series(np.arange(N, dtype=int), index=ids).to_dict()

def load_hard_masks(path):
    frame = pd.read_csv(path, dtype={ID_COL: str})
    out = {}
    for name in INVARIANTS:
        kid = frame.loc[frame["invariant"].eq(name), ID_COL].astype(str)
        m = np.zeros(N, dtype=bool)
        m[[id_to_pos[x] for x in kid]] = True
        out[name] = m
    return out

raw_masks = load_hard_masks(find_one("hard_sets_by_stable_id.csv"))
cond_masks = load_hard_masks(find_one("conditional_100bins_hard_sets.csv"))

with np.load(find_one("primary_pca_scores.npz"), allow_pickle=False) as p:
    raw_scores = {
        name: np.asarray(p[f"{safe_name(name)}__sse"], dtype=float)
        for name in INVARIANTS
    }

# Detect the conditional-percentile score key for each view.
with np.load(find_one("conditional_100bins_scores.npz"), allow_pickle=False) as p:
    payload_keys = list(p.files)
    conditional_scores = {}
    selected_key = {}
    for name in INVARIANTS:
        prefix = safe_name(name)
        candidates = []
        for key in payload_keys:
            if not key.startswith(prefix):
                continue
            if "norm_bin" in key.lower():
                continue
            arr = np.asarray(p[key])
            if arr.ndim != 1 or len(arr) != N:
                continue
            if np.isfinite(arr).all() and arr.min() >= -1e-12 and arr.max() <= 1 + 1e-12:
                candidates.append(key)
        # Prefer names that explicitly say percentile/conditional/score.
        ranked = sorted(
            candidates,
            key=lambda k: (
                "percent" not in k.lower(),
                "cond" not in k.lower(),
                "score" not in k.lower(),
                k,
            ),
        )
        if not ranked:
            raise RuntimeError(
                f"Could not identify conditional score for {name}. "
                f"Available keys: {[k for k in payload_keys if k.startswith(prefix)]}"
            )
        selected_key[name] = ranked[0]
        conditional_scores[name] = np.asarray(p[ranked[0]], dtype=float)

pd.DataFrame(
    [{"invariant": k, "conditional_score_key": v} for k, v in selected_key.items()]
).to_csv(OUT / "conditional_score_key_map.csv", index=False)

def cutoff_audit(name, score, selected):
    k = int(selected.sum())
    cutoff = float(np.min(score[selected]))
    strict = score > cutoff
    tie = np.isclose(score, cutoff, rtol=0.0, atol=1e-15)
    selected_tie = selected & tie
    slots = k - int(strict.sum())
    tie_ids = np.sort(ids[tie])
    expected = set(tie_ids[:slots])
    actual = set(ids[selected_tie])
    return {
        "invariant": name,
        "n": N,
        "selected_k": k,
        "cutoff": cutoff,
        "n_strictly_above_cutoff": int(strict.sum()),
        "boundary_tie_size": int(tie.sum()),
        "slots_taken_from_boundary_tie": int(slots),
        "selected_from_boundary_tie": int(selected_tie.sum()),
        "unselected_in_boundary_tie": int(tie.sum() - selected_tie.sum()),
        "boundary_is_ambiguous": bool(tie.sum() > slots),
        "stable_id_tie_rule_verified": actual == expected,
    }

rows = []
for name in INVARIANTS:
    rows.append({"score_family": "raw_sse", **cutoff_audit(name, raw_scores[name], raw_masks[name])})
    rows.append({"score_family": "conditional_100", **cutoff_audit(name, conditional_scores[name], cond_masks[name])})
audit = pd.DataFrame(rows)
audit.to_csv(OUT / "one_percent_boundary_tie_audit.csv", index=False)

def randomized_tail(score, observed, rng):
    k = int(observed.sum())
    cutoff = float(np.min(score[observed]))
    strict = score > cutoff
    tie_idx = np.flatnonzero(np.isclose(score, cutoff, rtol=0.0, atol=1e-15))
    slots = k - int(strict.sum())
    out = strict.copy()
    if slots > 0:
        out[rng.choice(tie_idx, size=slots, replace=False)] = True
    return out

def family(mask_dict, names, k):
    c = np.zeros(N, dtype=np.uint8)
    for name in names:
        c += mask_dict[name]
    return c >= k

obs_cond_all5 = family(cond_masks, INVARIANTS, 3)
obs_cond_nokh = family(cond_masks, NO_KHOVANOV, 3)

def jaccard(a, b):
    u = a | b
    return float(np.sum(a & b) / np.sum(u)) if u.any() else np.nan

def phenotype_metrics(mask):
    nonalt = mask & NONALT
    gv = G[nonalt]
    d = KH_DIAG[mask]
    return {
        "n": int(mask.sum()),
        "alternating_prop": float(np.mean(atlas.loc[mask, "is_alternating"])) if mask.any() else np.nan,
        "P_G_gt_0_nonalternating": float(np.mean(gv > 0)) if len(gv) else np.nan,
        "mean_G_nonalternating": float(np.mean(gv)) if len(gv) else np.nan,
        "P_kh_diag_ge_3": float(np.mean(d >= 3)) if len(d) else np.nan,
        "mean_kh_diag": float(np.mean(d)) if len(d) else np.nan,
    }

rng = np.random.default_rng(SEED)
sim_rows = []
for rep in range(N_RANDOM):
    sampled = {
        name: randomized_tail(conditional_scores[name], cond_masks[name], rng)
        for name in INVARIANTS
    }
    for label, names, k, observed in (
        ("all5_ge3", INVARIANTS, 3, obs_cond_all5),
        ("no_khovanov_ge3of4", NO_KHOVANOV, 3, obs_cond_nokh),
    ):
        m = family(sampled, names, k)
        sim_rows.append(
            {
                "replicate": rep,
                "family": label,
                "jaccard_with_deterministic": jaccard(m, observed),
                **phenotype_metrics(m),
            }
        )

sim = pd.DataFrame(sim_rows)
sim.to_parquet(OUT / "randomized_boundary_tie_sensitivity.parquet", index=False)

summary = (
    sim.groupby("family")
    .agg(
        n_min=("n", "min"),
        n_median=("n", "median"),
        n_max=("n", "max"),
        jaccard_q025=("jaccard_with_deterministic", lambda x: x.quantile(.025)),
        jaccard_median=("jaccard_with_deterministic", "median"),
        jaccard_q975=("jaccard_with_deterministic", lambda x: x.quantile(.975)),
        alt_prop_q025=("alternating_prop", lambda x: x.quantile(.025)),
        alt_prop_q975=("alternating_prop", lambda x: x.quantile(.975)),
        kh3_q025=("P_kh_diag_ge_3", lambda x: x.quantile(.025)),
        kh3_q975=("P_kh_diag_ge_3", lambda x: x.quantile(.975)),
        Ginc_q025=("P_G_gt_0_nonalternating", lambda x: x.quantile(.025)),
        Ginc_q975=("P_G_gt_0_nonalternating", lambda x: x.quantile(.975)),
    )
    .reset_index()
)
summary.to_csv(OUT / "randomized_boundary_tie_sensitivity_summary.csv", index=False)

decision = pd.DataFrame([{
    "all_deterministic_rules_verified": bool(audit["stable_id_tie_rule_verified"].all()),
    "n_ambiguous_raw_boundaries": int(
        ((audit["score_family"] == "raw_sse") & audit["boundary_is_ambiguous"]).sum()
    ),
    "n_ambiguous_conditional_boundaries": int(
        ((audit["score_family"] == "conditional_100") & audit["boundary_is_ambiguous"]).sum()
    ),
    "interpretation": (
        "Tie handling is deterministic and audited. Use randomized-boundary "
        "phenotype ranges to decide whether exact membership sensitivity matters."
    ),
}])
decision.to_csv(OUT / "tie_audit_decision.csv", index=False)

print("\n1% boundary audit:")
print(audit.to_string(index=False))
print("\nRandomized-boundary sensitivity:")
print(summary.to_string(index=False))
print("\nSaved Stage 30 to:", OUT)
