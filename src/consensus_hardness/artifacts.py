from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "scipy")) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def write_run_manifest(
    output_dir: str | Path,
    config: dict,
    input_files: dict[str, str | Path] | None = None,
    extra: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, path in (input_files or {}).items():
        path = Path(path)
        inputs[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = None

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "git_commit": git_commit,
        "config": config,
        "inputs": inputs,
        "extra": extra or {},
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return path


def save_hard_sets_by_id(
    hard_sets: dict[str, set[int]],
    meta: pd.DataFrame,
    path: str | Path,
    id_col: str = "knot_id_base",
) -> Path:
    rows = []
    for invariant, positions in hard_sets.items():
        for position in sorted(positions):
            rows.append(
                {
                    "invariant": invariant,
                    "position": int(position),
                    id_col: str(meta.iloc[position][id_col]),
                }
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def stage_directory(output_dir: str | Path, stage_number: int, name: str) -> Path:
    path = Path(output_dir) / f"{stage_number:02d}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path
