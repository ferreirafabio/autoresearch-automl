"""LLAMBO backend via OptunaHub — LLM-enhanced Bayesian optimization."""

from __future__ import annotations

import logging
from typing import Any

import optuna
from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend
from autoresearch_automl.core.search_space import configspace_to_optuna

logger = logging.getLogger(__name__)

DEFAULT_TASK_DESCRIPTION = """\
Optimizing a GPT-2 scale transformer on climbmix-400b-shuffle.
Muon+AdamW optimizer with weight decay and learning rate scheduling.
Goal: minimize val_bpb (validation bits-per-byte).
Training budget per config: {budget}s on H200 GPU.
"""


class LLAMBOBackend(HPOBackend):
    """LLAMBO backend: LLM as surrogate model via OptunaHub.

    LLAMBO uses the LLM for:
    1. Zero-shot warm-starting (domain-aware initial configs)
    2. LLM surrogate model (regression predictions with uncertainty)
    3. Semantic candidate sampling (task-description-aware suggestions)

    Best for <15 trials where TPE lacks data for a good density model.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        task_description: str | None = None,
        n_initial_samples: int = 5,
    ):
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._task_description = task_description
        self._n_initial_samples = n_initial_samples
        self._study: optuna.Study | None = None
        self._space: ConfigurationSpace | None = None
        self._optuna_mapping: dict[str, dict] = {}
        self._max_budget: float = 300.0
        self._pending_trial: optuna.Trial | None = None

    @property
    def name(self) -> str:
        return "llambo"

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

        import optunahub

        module = optunahub.load_module("samplers/llambo")

        task_desc = self._task_description or DEFAULT_TASK_DESCRIPTION.format(
            budget=self._max_budget,
        )

        sampler_kwargs = {
            "custom_task_description": task_desc,
            "model": self._model,
            "n_initial_samples": self._n_initial_samples,
            "sm_mode": "discriminative",
        }

        if self._api_base:
            sampler_kwargs["api_base"] = self._api_base
        if self._api_key:
            sampler_kwargs["api_key"] = self._api_key

        sampler = module.LLAMBOSampler(**sampler_kwargs)

        self._study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
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

        val = results.get(self._objectives[0], float("inf"))
        if val == float("inf"):
            self._study.tell(self._pending_trial, state=optuna.trial.TrialState.FAIL)
        else:
            self._study.tell(self._pending_trial, values=val)

        self._pending_trial = None

    def incumbents(self) -> list[dict[str, Any]]:
        if self._study is None:
            return []
        try:
            return [self._study.best_trial.params]
        except ValueError:
            return []

    def replay(self, history: list[tuple[dict, float, dict]]) -> None:
        """Replay completed trials into the LLAMBO/Optuna study."""
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

        for config, budget, results in history:
            val = results.get(self._objectives[0], float("inf"))
            if val == float("inf"):
                state = optuna.trial.TrialState.FAIL
                trial_values = None
            else:
                state = optuna.trial.TrialState.COMPLETE
                trial_values = [val]

            frozen = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                values=trial_values,
                state=state,
            )
            self._study.add_trial(frozen)
        logger.info("Replayed %d trials into LLAMBO study", len(history))

    @property
    def study(self) -> optuna.Study | None:
        return self._study
