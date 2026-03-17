"""SMAC3 backend — Random Forest surrogate with multi-fidelity support."""

from __future__ import annotations

import logging
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)


class SMACBackend(HPOBackend):
    """SMAC3 HPO backend with RF surrogate.

    Supports single-fidelity (BlackBoxFacade) and multi-fidelity
    (MultiFidelityFacade with Successive Halving / Hyperband).
    """

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
            from smac import MultiFidelityFacade, BlackBoxFacade, Scenario
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
            self._facade = BlackBoxFacade(
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
        value = TrialValue(cost=[results.get(o, float("inf")) for o in self._objectives])
        self._facade.tell(self._pending_info, value)
        self._results.append((config, budget, results))

    def replay(self, history: list[tuple[dict, float, dict]]) -> None:
        """Replay trials into SMAC's runhistory."""
        from smac.runhistory import TrialInfo, TrialValue
        from ConfigSpace import Configuration

        for config_dict, budget, results in history:
            try:
                config = Configuration(self._space, values=config_dict)
                cost = [results.get(o, float("inf")) for o in self._objectives]
                info = TrialInfo(config=config, seed=0, budget=budget if self._multi_fidelity else None)
                value = TrialValue(cost=cost)
                self._facade.runhistory.add(info, value)
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
