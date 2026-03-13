"""Main orchestration loop for HPO experiments."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch_automl.backends.base import HPOBackend
from autoresearch_automl.core.config_injector import ConfigInjector
from autoresearch_automl.core.experiment import ExperimentConfig, ExperimentOutcome, ExperimentRunner
from autoresearch_automl.core.search_space import SearchSpaceBuilder
from autoresearch_automl.tracking.results_db import ResultsDB, TrialRecord

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for an HPO run."""
    train_py_path: Path
    backend: HPOBackend
    n_trials: int = 100
    budget_min: float = 60.0
    budget_max: float = 300.0
    seed: int = 0
    objectives: list[str] = field(default_factory=lambda: ["val_bpb"])
    results_dir: Path = field(default_factory=lambda: Path("results"))
    run_command: list[str] | None = None  # default: ["uv", "run", "train.py"]
    gpu_device: int | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    # Hybrid mode settings
    hybrid_mode: str | None = None  # "A", "B", "C", or None
    structural_interval: int = 20  # trials between structural mutations (Mode B)
    warmstart_configs: int = 5  # number of warm-start configs (Mode C)
    # LLM model for hybrid modules (warmstart, structural mutator)
    llm_model: str | None = None
    # Auto-resume from existing trials.jsonl
    resume: bool = True


class Runner:
    """Main orchestration loop: suggest → inject → run → tell → repeat."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.runner = ExperimentRunner(
            train_py_path=config.train_py_path,
            run_command=config.run_command,
            gpu_device=config.gpu_device,
            extra_env=config.extra_env,
        )
        self.results_db = ResultsDB(config.results_dir / "trials.jsonl")
        self.best_val_bpb = float("inf")
        self.generation = 0

    def run(self) -> dict[str, Any]:
        """Execute the full HPO loop.

        Returns summary dict with best config, val_bpb, etc.
        """
        cfg = self.config

        # Build search space
        space_builder = SearchSpaceBuilder(cfg.train_py_path)
        space = space_builder.build_configspace()
        logger.info("ConfigSpace with %d HPs: %s", len(space), list(space.keys()))

        # Configure backend
        budget_range = (cfg.budget_min, cfg.budget_max) if cfg.backend.supports_multi_fidelity else None
        cfg.backend.configure(
            space=space,
            objectives=cfg.objectives,
            budget_range=budget_range,
            seed=cfg.seed,
        )

        # Resume from checkpoint if enabled
        start_trial_id = 0
        if cfg.resume:
            start_trial_id = self._resume_from_checkpoint()

        # Optional: warm-start with LLM configs (Mode C) — skip if resuming
        if cfg.hybrid_mode == "C" and start_trial_id == 0:
            self._warmstart(space, cfg.backend)

        # Main loop
        start_time = time.time()
        for trial_id in range(start_trial_id, cfg.n_trials):
            logger.info("=== Trial %d/%d ===", trial_id + 1, cfg.n_trials)

            # Mode B: structural mutation at intervals
            if cfg.hybrid_mode == "B" and trial_id > 0 and trial_id % cfg.structural_interval == 0:
                self._structural_mutation()

            # Suggest
            hp_config, budget = cfg.backend.suggest()

            # Run experiment
            exp_config = ExperimentConfig(
                trial_id=trial_id,
                hp_config=hp_config,
                budget=budget,
                seed=cfg.seed,
                generation=self.generation,
                backend_name=cfg.backend.name,
            )
            outcome = self.runner.run(exp_config)

            # Tell backend
            cfg.backend.tell(hp_config, budget, outcome.objectives)

            # Record
            record = TrialRecord(
                trial_id=trial_id,
                backend=cfg.backend.name,
                generation=self.generation,
                config=hp_config,
                budget=budget,
                val_bpb=outcome.result.val_bpb,
                train_loss=outcome.result.train_loss,
                peak_memory_gb=outcome.result.peak_memory_gb,
                wall_time_seconds=outcome.result.wall_time_seconds,
                tokens_per_second=outcome.result.tokens_per_second,
                success=outcome.result.success,
                error=outcome.result.error,
                seed=cfg.seed,
                extra_metrics=outcome.result.extra_metrics,
            )
            self.results_db.add(record)

            # Track best
            if outcome.result.val_bpb is not None and outcome.result.val_bpb < self.best_val_bpb:
                self.best_val_bpb = outcome.result.val_bpb
                logger.info("New best val_bpb: %.4f", self.best_val_bpb)

        total_time = time.time() - start_time
        incumbents = cfg.backend.incumbents()

        summary = {
            "best_val_bpb": self.best_val_bpb,
            "incumbents": incumbents,
            "total_trials": cfg.n_trials,
            "total_time_hours": total_time / 3600,
            "backend": cfg.backend.name,
            "seed": cfg.seed,
            "generations": self.generation,
        }
        logger.info("Run complete: %s", summary)
        return summary

    def _resume_from_checkpoint(self) -> int:
        """Replay completed trials into backend, return the next trial_id.

        Returns 0 if no checkpoint found.
        """
        state = self.results_db.resume_state()
        if state["n_completed"] == 0:
            return 0

        history = self.results_db.replay_history()
        self.config.backend.replay(history)
        self.best_val_bpb = state["best_val_bpb"]
        self.generation = state["max_generation"]

        next_id = state["next_trial_id"]
        logger.info(
            "Resumed: %d completed trials, best_val_bpb=%.4f, generation=%d, resuming from trial %d",
            state["n_completed"], self.best_val_bpb, self.generation, next_id,
        )
        return next_id

    def _warmstart(self, space, backend: HPOBackend) -> None:
        """Warm-start HPO with LLM-generated initial configs."""
        from autoresearch_automl.hybrid.warmstart import WarmStartOracle

        ws_kwargs = {}
        if self.config.llm_model:
            ws_kwargs["model"] = self.config.llm_model
        oracle = WarmStartOracle(**ws_kwargs)
        configs = oracle.generate_initial_configs(
            space, n_configs=self.config.warmstart_configs, budget=self.config.budget_max,
        )
        for i, cfg in enumerate(configs):
            logger.info("Warm-start config %d: %s", i, cfg)
            exp_config = ExperimentConfig(
                trial_id=-(i + 1),  # negative IDs for warm-start
                hp_config=cfg,
                budget=self.config.budget_max,
                seed=self.config.seed,
                generation=0,
                backend_name="warmstart",
            )
            outcome = self.runner.run(exp_config)
            backend.tell(cfg, self.config.budget_max, outcome.objectives)

            if outcome.result.val_bpb is not None and outcome.result.val_bpb < self.best_val_bpb:
                self.best_val_bpb = outcome.result.val_bpb

    def _structural_mutation(self) -> None:
        """Apply a structural mutation via LLM."""
        from autoresearch_automl.hybrid.structural_mutator import StructuralMutator

        mut_kwargs = {}
        if self.config.llm_model:
            mut_kwargs["model"] = self.config.llm_model
        mutator = StructuralMutator(**mut_kwargs)
        mutation = mutator.propose_mutation(
            train_py_path=self.config.train_py_path,
            best_val_bpb=self.best_val_bpb,
            n_trials=self.config.n_trials,
            generation=self.generation,
        )
        mutator.apply_mutation(self.config.train_py_path, mutation)
        self.generation = mutation.generation
        logger.info("Structural mutation applied: gen=%d, %s", self.generation, mutation.description)
