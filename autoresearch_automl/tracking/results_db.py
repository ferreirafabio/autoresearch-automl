"""Structured results storage — extends beyond results.tsv."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrialRecord:
    """A single trial's complete record."""
    trial_id: int
    backend: str
    generation: int
    config: dict
    budget: float
    val_bpb: float | None
    train_loss: float | None = None
    peak_memory_gb: float | None = None
    wall_time_seconds: float | None = None
    tokens_per_second: float | None = None
    success: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    seed: int = 0
    extra_metrics: dict = field(default_factory=dict)


class ResultsDB:
    """Stores and queries experiment results.

    Uses a JSONL file for append-only storage and provides
    DataFrame-based querying for analysis.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: TrialRecord) -> None:
        """Append a trial record."""
        with open(self.db_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load_all(self) -> list[TrialRecord]:
        """Load all trial records."""
        if not self.db_path.exists():
            return []
        records = []
        for line in self.db_path.read_text().strip().splitlines():
            if line:
                data = json.loads(line)
                records.append(TrialRecord(**data))
        return records

    def to_dataframe(self) -> pd.DataFrame:
        """Load all records as a DataFrame."""
        records = self.load_all()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in records])

    def best_trial(self, objective: str = "val_bpb") -> TrialRecord | None:
        """Return the trial with the best objective value."""
        records = [r for r in self.load_all() if r.success and getattr(r, objective, None) is not None]
        if not records:
            return None
        return min(records, key=lambda r: getattr(r, objective, float("inf")))

    def trials_by_backend(self, backend: str) -> list[TrialRecord]:
        """Filter trials by backend name."""
        return [r for r in self.load_all() if r.backend == backend]

    def trials_by_generation(self, generation: int) -> list[TrialRecord]:
        """Filter trials by structural generation."""
        return [r for r in self.load_all() if r.generation == generation]

    def resume_state(self) -> dict:
        """Extract state needed for resuming a run."""
        records = self.load_all()
        if not records:
            return {"n_completed": 0, "best_val_bpb": float("inf"), "max_generation": 0, "next_trial_id": 0}

        regular = [r for r in records if r.trial_id >= 0]
        best = float("inf")
        for r in records:
            if r.val_bpb is not None and r.val_bpb < best:
                best = r.val_bpb

        return {
            "n_completed": len(records),
            "best_val_bpb": best,
            "max_generation": max((r.generation for r in records), default=0),
            "next_trial_id": max((r.trial_id for r in regular), default=-1) + 1 if regular else 0,
        }

    def replay_history(self) -> list[tuple[dict, float, dict]]:
        """Build replay history as (config, budget, objectives) tuples."""
        records = self.load_all()
        history = []
        for r in sorted(records, key=lambda r: (r.trial_id >= 0, r.trial_id)):
            objectives: dict[str, float] = {}
            if r.val_bpb is not None:
                objectives["val_bpb"] = r.val_bpb
            if r.peak_memory_gb is not None:
                objectives["peak_memory_gb"] = r.peak_memory_gb
            if r.wall_time_seconds is not None:
                objectives["wall_time_seconds"] = r.wall_time_seconds
            history.append((r.config, r.budget, objectives))
        return history

    def summary(self) -> dict:
        """Return a summary of all results."""
        df = self.to_dataframe()
        if df.empty:
            return {"total_trials": 0}
        return {
            "total_trials": len(df),
            "successful_trials": df["success"].sum(),
            "best_val_bpb": df.loc[df["success"], "val_bpb"].min() if df["success"].any() else None,
            "backends": df["backend"].value_counts().to_dict(),
            "generations": df["generation"].nunique(),
            "total_wall_time_hours": df["wall_time_seconds"].sum() / 3600 if "wall_time_seconds" in df else None,
        }
