"""Random search baseline backend."""

from __future__ import annotations

import random
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend


class RandomBackend(HPOBackend):
    """Random search baseline — samples uniformly from ConfigurationSpace."""

    def __init__(self):
        self._space: ConfigurationSpace | None = None
        self._max_budget: float = 300.0
        self._results: list[tuple[dict, float, dict]] = []

    @property
    def name(self) -> str:
        return "random"

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
        self._rng = random.Random(seed)
        self._space.seed(seed)

    def suggest(self) -> tuple[dict[str, Any], float]:
        config = self._space.sample_configuration()
        return dict(config), self._max_budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        self._results.append((config, budget, results))

    def incumbents(self) -> list[dict[str, Any]]:
        if not self._results:
            return []
        obj = self._objectives[0]
        best = min(self._results, key=lambda x: x[2].get(obj, float("inf")))
        return [best[0]]
