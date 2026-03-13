"""Optuna backend with TPE sampler."""

from __future__ import annotations

import logging
from typing import Any

import optuna
from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend
from autoresearch_automl.core.search_space import configspace_to_optuna

logger = logging.getLogger(__name__)


class OptunaBackend(HPOBackend):
    """Optuna HPO backend with TPE sampler and optional pruning.

    Supports single- and multi-objective optimization.
    Supports Optuna RDBStorage for parallel Slurm execution.
    """

    def __init__(
        self,
        storage: str | None = None,
        study_name: str | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ):
        self._storage = storage
        self._study_name = study_name
        self._sampler = sampler
        self._pruner = pruner
        self._study: optuna.Study | None = None
        self._space: ConfigurationSpace | None = None
        self._optuna_mapping: dict[str, dict] = {}
        self._max_budget: float = 300.0
        self._pending_trial: optuna.Trial | None = None

    @property
    def name(self) -> str:
        return "optuna"

    @property
    def supports_multi_objective(self) -> bool:
        return True

    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self._space = space
        self._objectives = objectives
        self._max_budget = budget_range[1] if budget_range else 300.0
        self._optuna_mapping = configspace_to_optuna(space)

        sampler = self._sampler or optuna.samplers.TPESampler(seed=seed)
        pruner = self._pruner or optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=3,
        )

        directions = ["minimize"] * len(objectives)

        if len(objectives) == 1:
            self._study = optuna.create_study(
                study_name=self._study_name or "autoresearch_automl",
                storage=self._storage,
                sampler=sampler,
                pruner=pruner,
                direction="minimize",
                load_if_exists=True,
            )
        else:
            self._study = optuna.create_study(
                study_name=self._study_name or "autoresearch_automl",
                storage=self._storage,
                sampler=sampler,
                pruner=pruner,
                directions=directions,
                load_if_exists=True,
            )

    def suggest(self) -> tuple[dict[str, Any], float]:
        trial = self._study.ask()
        self._pending_trial = trial

        config = {}
        for hp_name, mapping in self._optuna_mapping.items():
            method = getattr(trial, mapping["method"])
            config[hp_name] = method(**mapping["kwargs"])

        return config, self._max_budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        if self._pending_trial is None:
            logger.warning("tell() called without a pending trial")
            return

        values = [results.get(obj, float("inf")) for obj in self._objectives]

        if any(v == float("inf") for v in values):
            self._study.tell(self._pending_trial, state=optuna.trial.TrialState.FAIL)
        elif len(values) == 1:
            self._study.tell(self._pending_trial, values=values[0])
        else:
            self._study.tell(self._pending_trial, values=values)

        self._pending_trial = None

    def incumbents(self) -> list[dict[str, Any]]:
        if self._study is None:
            return []
        try:
            if len(self._objectives) == 1:
                best = self._study.best_trial
                return [best.params]
            else:
                return [t.params for t in self._study.best_trials]
        except ValueError:
            return []

    def replay(self, history: list[tuple[dict, float, dict]]) -> None:
        """Replay completed trials into the Optuna study via add_trial()."""
        # If using RDB storage with existing trials, skip replay
        if self._storage and self._study and len(self._study.trials) > 0:
            logger.info("Optuna study already has %d trials from storage, skipping replay", len(self._study.trials))
            return

        distributions = self._build_distributions()
        for config, budget, results in history:
            values = [results.get(obj, float("inf")) for obj in self._objectives]
            if any(v == float("inf") for v in values):
                state = optuna.trial.TrialState.FAIL
                trial_values = None
            else:
                state = optuna.trial.TrialState.COMPLETE
                trial_values = values if len(values) > 1 else [values[0]]

            frozen = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                values=trial_values,
                state=state,
            )
            self._study.add_trial(frozen)
        logger.info("Replayed %d trials into Optuna study", len(history))

    def _build_distributions(self) -> dict:
        """Build Optuna distribution objects from the optuna mapping."""
        distributions = {}
        for hp_name, mapping in self._optuna_mapping.items():
            method = mapping["method"]
            kw = mapping["kwargs"]
            if method == "suggest_float":
                distributions[hp_name] = optuna.distributions.FloatDistribution(
                    low=kw["low"], high=kw["high"], log=kw.get("log", False),
                )
            elif method == "suggest_int":
                distributions[hp_name] = optuna.distributions.IntDistribution(
                    low=kw["low"], high=kw["high"], log=kw.get("log", False),
                )
            elif method == "suggest_categorical":
                distributions[hp_name] = optuna.distributions.CategoricalDistribution(
                    choices=kw["choices"],
                )
        return distributions

    @property
    def study(self) -> optuna.Study | None:
        """Expose the Optuna study for advanced usage (plotting, etc.)."""
        return self._study
