"""Single experiment execution: modify train.py, run, parse, track."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch_automl.core.config_injector import ConfigInjector
from autoresearch_automl.core.objective import ExperimentResult, ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full configuration for a single experiment."""
    trial_id: int
    hp_config: dict[str, object]
    budget: float = 300.0
    seed: int = 0
    generation: int = 0  # structural generation index
    backend_name: str = "unknown"
    tags: dict[str, str] = field(default_factory=dict)
    source_override: str | None = None  # LLM-generated source (Karpathy agent)


@dataclass
class ExperimentOutcome:
    """Full outcome of a single experiment, including config and result."""
    config: ExperimentConfig
    result: ExperimentResult
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def objectives(self) -> dict[str, float]:
        """Return objectives dict for HPO backend tell()."""
        obj = {}
        if self.result.val_bpb is not None:
            obj["val_bpb"] = self.result.val_bpb
        if self.result.peak_memory_gb is not None:
            obj["peak_memory_gb"] = self.result.peak_memory_gb
        if self.result.wall_time_seconds is not None:
            obj["wall_time_seconds"] = self.result.wall_time_seconds
        return obj


class ExperimentRunner:
    """Orchestrates a single experiment: inject config → run → collect results.

    Creates a per-job copy of train.py (e.g. train_llambo_seed0.py) in the
    same directory to avoid race conditions when multiple Slurm jobs share
    the same project directory. The copy name is deterministic (backend+seed),
    so reruns after preemption overwrite the old copy instead of leaving orphans.
    """

    def __init__(
        self,
        train_py_path: Path,
        run_command: list[str] | None = None,
        gpu_device: int | None = None,
        extra_env: dict[str, str] | None = None,
        job_id: str = "",
    ):
        self.train_py_path = train_py_path
        self.original_source = train_py_path.read_text()

        # Per-job copy: train.py → train_{job_id}.py in the same directory
        # job_id should be deterministic (e.g. "llambo_seed0") so reruns overwrite
        suffix = job_id or str(os.getpid())
        stem = train_py_path.stem
        self.work_train_py = train_py_path.parent / f"{stem}_{suffix}.py"
        self.work_train_py.write_text(self.original_source)
        logger.info("Created working copy at %s", self.work_train_py)

        self.injector = ConfigInjector(self.work_train_py)
        # Run command uses the copy filename, cwd stays in the same directory
        work_run_command = run_command or ["python", self.work_train_py.name]
        self.objective = ObjectiveFunction(
            self.work_train_py,
            run_command=work_run_command,
            gpu_device=gpu_device,
            extra_env=extra_env,
        )

    def cleanup(self) -> None:
        """Remove the per-process working copy."""
        if self.work_train_py.exists():
            self.work_train_py.unlink()
            logger.info("Cleaned up working copy %s", self.work_train_py)

    def run(self, config: ExperimentConfig) -> ExperimentOutcome:
        """Run a single experiment with the given configuration.

        1. Reset working copy to original train.py
        2. Inject HP config values
        3. Run train.py with budget
        4. Return results
        """
        outcome = ExperimentOutcome(config=config, result=ExperimentResult())

        try:
            if config.source_override:
                # Karpathy agent: write LLM-generated source directly
                self.work_train_py.write_text(config.source_override)
                logger.info(
                    "Trial %d: wrote LLM-generated source (%d chars)",
                    config.trial_id, len(config.source_override),
                )
            else:
                # Normal path: reset to original + inject HP values
                self.work_train_py.write_text(self.original_source)
                self.injector.inject_and_write(config.hp_config)
            logger.info(
                "Trial %d: running with %d HPs, budget=%.0fs",
                config.trial_id, len(config.hp_config), config.budget,
            )

            # Run experiment
            outcome.start_time = time.time()
            outcome.result = self.objective.evaluate(budget=config.budget)
            outcome.end_time = time.time()

            if outcome.result.success:
                logger.info(
                    "Trial %d: val_bpb=%.4f (%.1fs)",
                    config.trial_id,
                    outcome.result.val_bpb or float("inf"),
                    outcome.result.wall_time_seconds or 0,
                )
            else:
                logger.warning(
                    "Trial %d: FAILED — %s",
                    config.trial_id, outcome.result.error,
                )

        except Exception:
            logger.exception("Trial %d: unexpected error", config.trial_id)

        return outcome
