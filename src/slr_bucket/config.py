from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    event_dates: list[str] = field(
        default_factory=lambda: ["2020-04-01", "2021-03-19", "2021-03-31"]
    )
    windows: list[int] = field(default_factory=lambda: [3, 5, 10])
    event_bins: list[tuple[int, int]] = field(
        default_factory=lambda: [(-60, -41), (-40, -21), (-20, -1), (0, 0), (1, 20), (21, 40), (41, 60)]
    )
    dependent_series: list[str] | None = None
    tenor_subset: list[str] | None = None
    total_controls: list[str] = field(default_factory=list)
    direct_controls: list[str] = field(default_factory=lambda: ["sofr", "tgcr", "bgcr"])
    hac_lags: int = 5
    bootstrap_reps: int = 200
    bootstrap_block_size: int = 5
    random_seed: int = 42
    output_root: str = "outputs/summary_pipeline"
    cache_root: str = "outputs/cache"

    def to_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def resolve_run_dir(self, root: Path) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = root / self.output_root / f"{ts}_{self.to_hash()}"
        return run_dir


def as_serializable_dict(config: PipelineConfig) -> dict[str, Any]:
    return asdict(config)
