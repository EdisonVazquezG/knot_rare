from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


DEFAULT_INVARIANTS = (
    "Alexander",
    "Jones",
    "HOMFLY-PT",
    "Theta",
    "Khovanov",
)


@dataclass(frozen=True)
class UniverseConfig:
    min_crossings: int = 3
    max_crossings: int = 15
    expected_n: int = 313_230
    identity_id: str = "00_1"
    id_col: str = "knot_id_base"
    expected_s_qc_corrections: int = 1


@dataclass(frozen=True)
class PCAConfig:
    evr_thresholds: tuple[float, ...] = (0.94, 0.99, 0.999)
    primary_evr: float = 0.99
    tail_mass: float = 0.01
    expected_tail_size: int = 3_133
    expected_primary_consensus_size: int = 292
    primary_k: dict[str, int] = field(
        default_factory=lambda: {
            "Alexander": 4,
            "Jones": 10,
            "HOMFLY-PT": 32,
            "Theta": 10,
            "Khovanov": 77,
        }
    )
    sensitivity_k_999: dict[str, int] = field(
        default_factory=lambda: {
            "Alexander": 5,
            "Jones": 13,
            "HOMFLY-PT": 45,
            "Theta": 16,
            "Khovanov": 115,
        }
    )
    fixed_compression_k: dict[str, int] = field(
        default_factory=lambda: {
            "Alexander": 5,
            "Jones": 10,
            "HOMFLY-PT": 20,
            "Theta": 10,
            "Khovanov": 50,
        }
    )


@dataclass(frozen=True)
class RunConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    random_seed: int = 42
    null_reps: int = 1_000
    conditional_null_reps: int = 1_000
    gap_null_reps: int = 5_000
    norm_bins: int = 100
    s_col: str = "s_invariant_qc"

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def canonical_run_config() -> RunConfig:
    """Return the frozen configuration for the corrected 3--15 crossing run."""

    return RunConfig()
