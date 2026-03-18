"""SMAC3 backend — HyperparameterOptimizationFacade (Random Forest surrogate)."""

from __future__ import annotations

import logging
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)


class SMACBackend(HPOBackend):
"""SMAC3 HPO backend using HyperparameterOptimizationFacade (Random Forest surrogate)."""

    # Penalty for failed trials — must be finite (sklearn RF can't handle inf/NaN)
    FAILURE_COST = 100.0

    def __init__(self, multi_fidelity: bool = False, n_workers: int = 1):
        self._multi_fidelity = multi_fidelity
        self._n_workers = n_workers
        self._space: ConfigurationSpace | None = None
        self._objectives: list[str] = []
        self._min_budget: float = 60.0
        self._max_budget: float = 300.0
        self._results: list[tuple[dict, float, dict]] = []
        self._facade = None
        self._intensifier = None

    @property
    def name(self) -> str:
        return "smac" + ("_mf" if self._multi_fidelity else "")

    @property
    def supports_multi_fidelity(self) -> bool:
        return self._multi_fidelity

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
        if budget_range:
            self._min_budget, self._max_budget = budget_range

        try:
            from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario
        except ImportError:
            raise ImportError(
                "SMAC3 is required for SMACBackend. "
                "Install with: pip install 'autoresearch-automl[smac]'"
            )

        scenario = Scenario(
            configspace=space,
            objectives=objectives,
            n_trials=kwargs.get("n_trials", 100),
            seed=seed,
            min_budget=self._min_budget if self._multi_fidelity else None,
            max_budget=self._max_budget if self._multi_fidelity else None,
            n_workers=self._n_workers,
        )

        if self._multi_fidelity:
            self._facade = MultiFidelityFacade(
                scenario=scenario,
                target_function=lambda config, seed, budget: {},  # placeholder
                overwrite=True,
            )
        else:
            self._facade = HyperparameterOptimizationFacade(
                scenario=scenario,
                target_function=lambda config, seed: {},  # placeholder
                overwrite=True,
            )

    def suggest(self) -> tuple[dict[str, Any], float]:
        info = self._facade.ask()
        config = dict(info.config)
        budget = getattr(info, "budget", self._max_budget) or self._max_budget
        self._pending_info = info
        return config, budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        from smac.runhistory import TrialValue
        from smac.runhistory.enumerations import StatusType

        cost = [results.get(o) if results.get(o) is not None else self.FAILURE_COST for o in self._objectives]
        is_failure = any(results.get(o) is None for o in self._objectives)
        status = StatusType.MEMORYOUT if is_failure else StatusType.SUCCESS
        value = TrialValue(cost=cost, status=status)
        self._facade.tell(self._pending_info, value)
        self._results.append((config, budget, results))

    def seed_trial(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        """Inject baseline trial directly into SMAC's runhistory (no suggest needed)."""
        from ConfigSpace import Configuration
        from smac.runhistory.enumerations import StatusType

        cfg = Configuration(self._space, values=config)
        cost = [results.get(o) if results.get(o) is not None else self.FAILURE_COST for o in self._objectives]
        is_failure = any(results.get(o) is None for o in self._objectives)
        status = StatusType.MEMORYOUT if is_failure else StatusType.SUCCESS
        self._facade.runhistory.add(
            config=cfg,
            cost=cost,
            seed=0,
            budget=budget if self._multi_fidelity else None,
            status=status,
        )
        self._results.append((config, budget, results))
        logger.info("Seeded baseline trial into SMAC runhistory")

    def replay(self, history: list[tuple[dict, float, dict]]) -> None:
        """Replay trials into SMAC's runhistory."""
        from ConfigSpace import Configuration
        from smac.runhistory.enumerations import StatusType

        for config_dict, budget, results in history:
            try:
                config = Configuration(self._space, values=config_dict)
                cost = [results.get(o) if results.get(o) is not None else self.FAILURE_COST for o in self._objectives]
                is_failure = any(results.get(o) is None for o in self._objectives)
                status = StatusType.MEMORYOUT if is_failure else StatusType.SUCCESS
                self._facade.runhistory.add(
                    config=config,
                    cost=cost,
                    seed=0,
                    budget=budget if self._multi_fidelity else None,
                    status=status,
                )
            except Exception as e:
                logger.warning("Failed to replay trial into SMAC: %s", e)
        logger.info("Replayed %d trials into SMAC runhistory", len(history))

    def incumbents(self) -> list[dict[str, Any]]:
        if self._facade is None:
            return []
        try:
            inc = self._facade.intensifier.get_incumbents()
            return [dict(c) for c in inc]
        except Exception:
            return []
