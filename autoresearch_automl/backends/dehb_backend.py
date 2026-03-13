"""DEHB backend — Differential Evolution + Hyperband."""

from __future__ import annotations

import logging
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)


class DEHBBackend(HPOBackend):
    """DEHB HPO backend with multi-fidelity via Hyperband brackets.

    DEHB combines Differential Evolution mutations with Hyperband's
    multi-fidelity scheduling for efficient hyperparameter optimization.
    """

    def __init__(self, eta: int = 3):
        self._eta = eta
        self._space: ConfigurationSpace | None = None
        self._objectives: list[str] = []
        self._min_budget: float = 60.0
        self._max_budget: float = 300.0
        self._dehb = None
        self._results: list[tuple[dict, float, dict]] = []

    @property
    def name(self) -> str:
        return "dehb"

    @property
    def supports_multi_fidelity(self) -> bool:
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
            from dehb import DEHB
        except ImportError:
            raise ImportError(
                "DEHB is required for DEHBBackend. "
                "Install with: pip install 'autoresearch-automl[dehb]'"
            )

        self._dehb = DEHB(
            cs=space,
            min_fidelity=self._min_budget,
            max_fidelity=self._max_budget,
            eta=self._eta,
            seed=seed,
        )

    def suggest(self) -> tuple[dict[str, Any], float]:
        job_info = self._dehb.ask()
        self._pending_job = job_info
        config = dict(job_info["config"])
        budget = job_info["fidelity"]
        return config, budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        cost = results.get(self._objectives[0], float("inf"))
        result = {
            "fitness": cost,
            "cost": budget,
        }
        self._dehb.tell(self._pending_job, result)
        self._results.append((config, budget, results))

    def incumbents(self) -> list[dict[str, Any]]:
        if self._dehb is None:
            return []
        try:
            inc_config = self._dehb.get_incumbents()
            if isinstance(inc_config, list):
                return [dict(c) for c in inc_config]
            return [dict(inc_config)]
        except Exception:
            if self._results:
                best = min(self._results, key=lambda x: x[2].get(self._objectives[0], float("inf")))
                return [best[0]]
            return []
