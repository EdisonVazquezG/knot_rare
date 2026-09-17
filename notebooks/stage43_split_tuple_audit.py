# %%
"""Stage 43: exact input-tuple overlap across the frozen train/val/test split.

Run from paper_run: %run "/path/to/notebooks/stage43_split_tuple_audit.py"
No fitting, resplitting, or randomization. Reads ORIGINAL source coefficients.
Reports both exact integer tuples and tuples after the pipeline's float32 cast.
Hash buckets are checked against actual rows: hash equality alone is never used.
Stage 44 imports this file's helpers; keep both scripts in the same directory.

Dependencies already used by the project: numpy, pandas, scipy, sklearn, pyarrow.
The legacy Khovanov pickle is loaded as in Stage 35; use Colab's high-RAM runtime.
Temporary coefficient matrices use local disk, not the notebook's X_dict.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile

import numpy as np
import pandas as pd

ID = "knot_id_base"
VIEWS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
SOURCES = {
    "Alexander": ("Alexander_upto17.csv", "A"),
    "Jones": ("Jones_upto17_MIRRORS.csv", "J"),
    "HOMFLY-PT": ("HomflyPt_upto15_MIRRORS.csv", "a"),
    "Theta": ("theta_upto15.csv", "T"),
    "Khovanov": ("even_KH_upto17.pkl", "F_"),
}
META = {"knot_id", "knot_id_clean", ID, "number_of_crossings", "table_number",
        "is_alternating", "signature", "minimum_exponent", "maximum_exponent",
        "s_invariant"}
DEFAULT_DATA = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
FAMILIES = {**{v: (v,) for v in VIEWS},
            "HOMFLY_PT_plus_Theta": ("HOMFLY-PT", "Theta"),
            "four_view_noKh": VIEWS[:4], "all_five": VIEWS}


def safe(s):
    return s.replace("-", "_").replace(" ", "_")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    data = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA)))
    root = Path(os.environ.get("KNOT_OUTPUT_DIR", str(data / "processed_consensus_hardness/corrected_run_20260819")))
    p.add_argument("--root", "--output-dir", dest="root", type=Path, default=root)
    p.add_argument("--data-dir", type=Path, default=data)
    p.add_argument("--atlas", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    p.add_argument("--chunk-rows", type=int, default=4096)
    p.add_argument("--check-only", action="store_true")
    return p


def required(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}. Check --root/--data-dir (Drive mounted?).")
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_npz(path, **payload):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def freeze(out, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"Inputs, code or configuration changed. Choose a new --out; preserving {out}.")
    else:
        if any(out.iterdir()):
            raise RuntimeError(f"Nonempty output has no manifest: {out}. Choose a new --out.")
        atomic_json(path, manifest)


def load_frozen(args):
    atlas_path = required(args.atlas or args.root / "17_final_paper_outputs/final_hard_regime_atlas.parquet")
    split_path = required(args.split or args.root / "07_heldout_ae_target_free/scores/heldout_ae_seed_0.npz")
    atlas = (pd.read_csv(atlas_path) if atlas_path.suffix == ".csv" else pd.read_parquet(atlas_path)).reset_index(drop=True)
    if ID not in atlas or atlas[ID].isna().any():
        raise ValueError("Missing atlas identifiers")
    ids = atlas[ID].astype(str).to_numpy(dtype=str)
    if len(set(ids)) != len(ids) or any("!" in x or x != x.strip() for x in ids):
        raise ValueError("Atlas must contain unique, normalized canonical base IDs")
    with np.load(split_path, allow_pickle=False) as q:
        split = {}
        for key in ("train_idx", "val_idx", "test_idx"):
            value = q[key]
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"Invalid split indices: {key}")
            split[key] = value.astype(np.int64)
        if "test_ids" in q and not np.array_equal(q["test_ids"].astype(str), ids[split["test_idx"]]):
            raise ValueError("Split test IDs do not match atlas row order")
    if any(len(v) == 0 for v in split.values()) or not np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(ids))):
        raise ValueError("Frozen indices must cover the atlas exactly once")
    # Validate notebook row order if a live aligned meta is present; never reuse X_dict.
    try:
        from IPython import get_ipython
        shell = get_ipython()
        live = shell.user_ns.get("meta") if shell else None
        if isinstance(live, pd.DataFrame) and ID in live and not np.array_equal(live[ID].astype(str).to_numpy(), ids):
            raise ValueError("Live notebook meta differs from frozen atlas order; reload the matching alignment or use a fresh session.")
    except ImportError:
        pass
    return atlas, ids, split, [atlas_path, split_path]


@contextmanager
def source_matrix(path, view, ids, work_dir, chunk_rows):
    """Stream the canonical integer coefficients to a temporary local memmap.

    Matches signature-first, unmarked-second representative selection. Missing
    numeric entries become zero as in the published pipeline, with counts logged.
    Fractional or unsafe floating coefficients are rejected, not rounded.
    """
    path = required(path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    lookup = pd.Series(np.arange(len(ids)), index=ids)
    seen = np.full(len(ids), -1, dtype=np.int8)
    columns = None
    invalid_count = 0
    with tempfile.TemporaryDirectory(prefix="knot_coefficients_", dir=work_dir) as tmp:
        X = None
        chunks = pd.read_csv(path, chunksize=chunk_rows, dtype={"knot_id": str}) if path.suffix == ".csv" else [pd.read_pickle(path)]
        for number, raw in enumerate(chunks, 1):
            if "knot_id" not in raw:
                raise ValueError(f"No knot_id in {path}")
            names = raw.knot_id.astype(str).str.strip()
            base = names.str.replace("!", "", regex=False)
            positions = base.map(lookup)
            keep = positions.notna()
            if not keep.any():
                continue
            frame = raw.loc[keep].copy()
            pos = positions[keep].to_numpy(np.int64)
            sig = pd.to_numeric(frame["signature"], errors="coerce") if "signature" in frame else pd.Series(np.nan, index=frame.index)
            priority = 2 * (sig.to_numpy() >= 0).astype(np.int8) + (~names[keep].str.contains("!", regex=False)).to_numpy(np.int8)
            cols = [str(c) for c in frame.columns if c not in META and str(c).startswith(SOURCES[view][1])]
            if columns is None:
                if not cols:
                    raise ValueError(f"No feature coordinates for {view}")
                columns = cols
                X = np.lib.format.open_memmap(Path(tmp) / "coefficients.npy", mode="w+", dtype=np.int64, shape=(len(ids), len(cols)))
            if cols != columns:
                raise ValueError("Feature columns changed while reading")
            values = np.empty((len(frame), len(cols)), dtype=np.int64)
            for j, col in enumerate(cols):
                numeric = pd.to_numeric(frame[col], errors="coerce")
                invalid_count += int(numeric.isna().sum())
                numeric = numeric.fillna(0)
                a = numeric.to_numpy()
                if np.issubdtype(a.dtype, np.integer):
                    if (a > np.iinfo(np.int64).max).any():
                        raise ValueError(f"Integer overflow in {col}")
                elif not np.isfinite(a).all() or (np.abs(a) > 2**53).any() or not np.equal(a, np.trunc(a)).all():
                    raise ValueError(f"Noninteger or unsafe floating source values in {col}")
                values[:, j] = a
            # Process the two mirror priorities independently; source duplicates
            # with the same priority must agree exactly, even across chunks.
            for pr in np.unique(priority):
                indices = np.flatnonzero(priority == pr)
                pp = pos[indices]
                if len(np.unique(pp)) != len(pp):
                    raise ValueError(f"Duplicate source IDs at equal representative priority in {path}")
                tied = seen[pp] == pr
                if tied.any() and not np.array_equal(np.asarray(X[pp[tied]]), values[indices[tied]]):
                    raise ValueError("Conflicting repeated source representatives")
                better = pr > seen[pp]
                X[pp[better]] = values[indices[better]]
                seen[pp[better]] = pr
            if number % 25 == 0:
                print(f"  {view}: scanned {number * chunk_rows:,} rows; aligned {(seen >= 0).sum():,}/{len(ids):,}", flush=True)
        if columns is None or (seen < 0).any():
            raise ValueError(f"Missing canonical source rows: {ids[seen < 0][:10].tolist()}")
        X.flush()
        info = dict(view=view, rows=len(ids), features=len(columns), source_non_numeric_or_missing_entries_filled_zero=invalid_count,
                    representative_rule="nonnegative signature, then unmarked identifier")
        try:
            yield X, columns, info
        finally:
            del X
            gc.collect()


def exact_groups(X, as_float32=False, batch=4096, hashes=None):
    """Hash to find candidates, then partition every duplicate bucket exactly."""
    if hashes is None:
        hashes = np.empty(len(X), dtype=np.uint64)
        for start in range(0, len(X), batch):
            block = np.asarray(X[start:start+batch], dtype=np.float32 if as_float32 else np.int64)
            hashes[start:start+len(block)] = pd.util.hash_pandas_object(pd.DataFrame(block), index=False).to_numpy(np.uint64)
    order = np.argsort(hashes, kind="stable")
    cuts = np.r_[0, 1 + np.flatnonzero(np.diff(hashes[order]) != 0), len(order)]
    labels = np.empty(len(X), dtype=np.int64)
    next_label = 0
    collision_buckets = 0
    for start, stop in zip(cuts[:-1], cuts[1:]):
        ix = order[start:stop]
        if len(ix) == 1:
            labels[ix] = next_label
            next_label += 1
        else:
            block = np.asarray(X[ix], dtype=np.float32 if as_float32 else np.int64)
            unique, inverse = np.unique(block, axis=0, return_inverse=True)
            collision_buckets += int(len(unique) > 1)
            labels[ix] = next_label + inverse
            next_label += len(unique)
    return labels, collision_buckets


def overlap_flags(labels, split):
    size = int(labels.max()) + 1
    counts = {name: np.bincount(labels[idx], minlength=size) for name, idx in split.items()}
    train, val, test = (counts[k] for k in ("train_idx", "val_idx", "test_idx"))
    return counts, dict(seen_in_train=train[labels] > 0, seen_in_validation=val[labels] > 0,
                        seen_in_train_or_validation=(train + val)[labels] > 0)


def read_cohorts(root, test_ids):
    cohorts, paths = [], []
    specs = [
        ("33_view_ablation_and_fibers/heldout_view_ablation_n31_selected_ids.csv", ["selection"]),
        ("40_revision_audit_direct/reconstructed_equal_size_ids.csv", ["family", "method"]),
        ("23_anomaly_score_baselines/score_selected_test_ids.csv", ["family", "method"]),
    ]
    allowed = set(test_ids)
    for rel, fields in specs:
        path = root / rel
        if not path.exists():
            print("Optional cohort file absent:", path)
            continue
        frame = pd.read_csv(path, dtype={ID: str})
        if not set(fields + [ID]) <= set(frame):
            raise ValueError(f"Unexpected cohort columns: {path}")
        for key, part in frame.groupby(fields, dropna=False):
            if part[ID].isna().any() or part[ID].duplicated().any() or not set(part[ID]) <= allowed:
                raise ValueError(f"Invalid or non-test cohort IDs in {path}: {key}")
            cohorts.append((rel + ":" + str(key), set(part[ID])))
        paths.append(path)
    return cohorts, paths


def bundle(out, name):
    archive = out / name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix in {".csv", ".json", ".txt"}:
                z.write(path, path.name)
    print("Review ZIP:", archive)


def main():
    p = parser(__doc__)
    args = p.parse_args()
    if args.chunk_rows < 1:
        p.error("--chunk-rows must be positive")
    root = args.root.expanduser().resolve()
    out = args.out or root / "43_split_tuple_audit"
    atlas, ids, split, inputs = load_frozen(args)
    cohorts, cohort_paths = read_cohorts(root, ids[split["test_idx"]])
    sources = {v: required(args.data_dir / SOURCES[v][0]) for v in VIEWS}
    print("Frozen split:", {k: len(v) for k, v in split.items()}, "cohorts:", len(cohorts))
    print("No fitting. Original integer tuples and pipeline float32 tuples will be audited separately.")
    if args.check_only:
        print("Preflight complete; run without --check-only for the audit.")
        return
    files = inputs + cohort_paths + list(sources.values()) + [Path(__file__)]
    manifest = dict(stage=43, version=1, chunk_rows=args.chunk_rows, numpy=np.__version__, pandas=pd.__version__,
                    python=platform.python_version(), input_sha256={str(f.resolve()): digest(f) for f in files},
                    scope="Exact pre-scaling source tuples and float32 conversion; no near-duplicate or post-scaling audit")
    freeze(out, manifest)
    labels_by_encoding = {"source_integer": {}, "pipeline_float32": {}}
    audits = []
    for view in VIEWS:
        cache = out / (safe(view) + "_groups.npz")
        metadata = out / (safe(view) + "_source_audit.json")
        if cache.exists() and metadata.exists():
            with np.load(cache, allow_pickle=False) as q:
                if not np.array_equal(q["ids"].astype(str), ids):
                    raise ValueError("Cached group ID order mismatch")
                for enc in labels_by_encoding:
                    labels_by_encoding[enc][view] = q[enc].copy()
            audits.append(json.loads(metadata.read_text()))
            print("Reused verified groups:", view)
            continue
        print("Reading and grouping:", view, flush=True)
        with source_matrix(sources[view], view, ids, args.work_dir, args.chunk_rows) as (X, columns, info):
            for enc in labels_by_encoding:
                labels, collisions = exact_groups(X, enc == "pipeline_float32", args.chunk_rows)
                labels_by_encoding[enc][view] = labels
                info[enc + "_hash_collision_buckets_resolved"] = collisions
                info[enc + "_unique_tuples"] = int(labels.max()) + 1
            info["float32_added_collisions"] = info["source_integer_unique_tuples"] - info["pipeline_float32_unique_tuples"]
            info["columns"] = columns
            atomic_npz(cache, ids=ids, **{enc: data[view] for enc, data in labels_by_encoding.items()})
            atomic_json(metadata, info)
            audits.append(info)
    summary, selected_summary, test_records, group_arrays = [], [], [], {}
    te = split["test_idx"]
    for enc, per_view in labels_by_encoding.items():
        for family, views in FAMILIES.items():
            if len(views) == 1:
                labels = per_view[views[0]]
            else:
                _, labels = np.unique(np.column_stack([per_view[v] for v in views]), axis=0, return_inverse=True)
            group_arrays[enc + "__" + safe(family)] = labels
            counts, flags = overlap_flags(labels, split)
            train, val, test = (counts[k] for k in ("train_idx", "val_idx", "test_idx"))
            total = train + val + test
            row = dict(encoding=enc, input_family=family, n_test=len(te), unique_tuples=len(total),
                       nontrivial_groups=int((total > 1).sum()),
                       groups_spanning_partitions=int(((train > 0).astype(int)+(val > 0)+(test > 0) > 1).sum()),
                       validation_seen_in_train_n=int(flags["seen_in_train"][split["val_idx"]].sum()))
            for key, mask in flags.items():
                row["test_" + key + "_n"] = int(mask[te].sum())
                row["test_" + key + "_fraction"] = float(mask[te].mean())
            summary.append(row)
            for cohort, members in cohorts:
                mask = np.isin(ids[te], list(members))
                selected_summary.append(dict(encoding=enc, input_family=family, cohort=cohort, selected_n=int(mask.sum()),
                    **{key + "_n": int(flag[te][mask].sum()) for key, flag in flags.items()}))
            match = flags["seen_in_train_or_validation"][te]
            if match.any():
                pos = te[match]
                test_records.append(pd.DataFrame({ID: ids[pos], "encoding": enc, "input_family": family,
                    "tuple_group": labels[pos], "train_matches": train[labels[pos]], "validation_matches": val[labels[pos]]}))
    pd.DataFrame(summary).to_csv(out / "split_tuple_overlap_summary.csv", index=False)
    pd.DataFrame(selected_summary, columns=["encoding", "input_family", "cohort", "selected_n", "seen_in_train_n", "seen_in_validation_n", "seen_in_train_or_validation_n"]).to_csv(out / "selected_tuple_overlap_summary.csv", index=False)
    (pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame(columns=[ID, "encoding", "input_family", "tuple_group", "train_matches", "validation_matches"])).to_csv(out / "test_tuple_overlap_ids.csv", index=False)
    atomic_npz(out / "all_tuple_group_memberships.npz", ids=ids, **split, **group_arrays)
    pd.DataFrame([{k: v for k, v in r.items() if k != "columns"} for r in audits]).to_csv(out / "source_encoding_audit.csv", index=False)
    (out / "interpretation.txt").write_text(
        "Counts concern identical inputs, not proof of label leakage or generalization failure.\n"
        "Read each cohort against its actual input_family (four_view_noKh, all_five, or its ablation).\n"
        "source_integer is before float32 rounding; pipeline_float32 includes that conversion.\n"
        "Neither tests near duplicates, shared mathematical structure, or additional equality after fitted scaling.\n"
        "No partition is changed. Complete group assignments are saved in all_tuple_group_memberships.npz.\n"
        "The summary ZIP omits the NPZ checkpoints; retain them in Drive.\n")
    print(pd.DataFrame(summary).to_string(index=False))
    bundle(out, "stage43_review.zip")
    print("STAGE 43 COMPLETE:", out)


if __name__ == "__main__":
    main()
