"""Single experiment execution: modify train.py, run, parse, track."""

from __future__ import annotations

import logging
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
    """Orchestrates a single experiment: inject config → run → collect results."""

    def __init__(
        self,
        train_py_path: Path,
        run_command: list[str] | None = None,
        gpu_device: int | None = None,
        extra_env: dict[str, str] | None = None,
    ):
        self.train_py_path = train_py_path
        self.injector = ConfigInjector(train_py_path)
        self.objective = ObjectiveFunction(
            train_py_path,
            run_command=run_command,
            gpu_device=gpu_device,
            extra_env=extra_env,
        )

    def run(self, config: ExperimentConfig) -> ExperimentOutcome:
        """Run a single experiment with the given configuration.

        1. Save original train.py
        2. Inject HP config values
        3. Run train.py with budget
        4. Restore original train.py
        5. Return results
        """
        original_source = self.train_py_path.read_text()
        outcome = ExperimentOutcome(config=config, result=ExperimentResult())

        try:
            # Inject config
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

        finally:
            # Always restore original train.py
            self.train_py_path.write_text(original_source)

        return outcome
