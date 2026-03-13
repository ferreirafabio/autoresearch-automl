"""BOHB backend via HpBandSter — Bayesian Optimization + Hyperband."""

from __future__ import annotations

import logging
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)


class BOHBBackend(HPOBackend):
    """BOHB HPO backend using HpBandSter.

    BOHB combines kernel density estimation (KDE) with Hyperband's
    multi-fidelity scheduling. Uses ask/tell wrapper around HpBandSter.
    """

    def __init__(self, eta: int = 3):
        self._eta = eta
        self._space: ConfigurationSpace | None = None
        self._objectives: list[str] = []
        self._min_budget: float = 60.0
        self._max_budget: float = 300.0
        self._results: list[tuple[dict, float, dict]] = []
        self._configs_queue: list[tuple[dict, float]] = []
        self._sampler = None

    @property
    def name(self) -> str:
        return "bohb"

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
            from hpbandster.optimizers.config_generators.bohb import BOHB as BOHBConfigGen
        except ImportError:
            raise ImportError(
                "HpBandSter is required for BOHBBackend. "
                "Install with: pip install 'autoresearch-automl[bohb]'"
            )

        self._sampler = BOHBConfigGen(
            configspace=space,
            min_bandwidth=1e-3,
            num_samples=64,
            random_fraction=1 / 3,
            bandwidth_factor=3,
        )
        self._budget_levels = self._compute_budget_levels()
        self._budget_idx = 0

    def suggest(self) -> tuple[dict[str, Any], float]:
        budget = self._budget_levels[self._budget_idx % len(self._budget_levels)]
        self._budget_idx += 1

        try:
            config, _info = self._sampler.get_config(budget)
            return dict(config), budget
        except Exception:
            config = self._space.sample_configuration()
            return dict(config), budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        cost = results.get(self._objectives[0], float("inf"))
        job_id = len(self._results)
        try:
            self._sampler.new_result(
                {"config": config, "budget": budget, "config_id": (0, 0, job_id)},
                {"loss": cost, "info": results},
            )
        except Exception as e:
            logger.debug("BOHB new_result failed: %s", e)
        self._results.append((config, budget, results))

    def incumbents(self) -> list[dict[str, Any]]:
        if not self._results:
            return []
        # Filter to max-budget results, then find best
        full_budget = [r for r in self._results if r[1] >= self._max_budget * 0.9]
        pool = full_budget if full_budget else self._results
        best = min(pool, key=lambda x: x[2].get(self._objectives[0], float("inf")))
        return [best[0]]

    def _compute_budget_levels(self) -> list[float]:
        """Compute Successive Halving budget levels."""
        levels = []
        b = self._max_budget
        while b >= self._min_budget:
            levels.append(b)
            b /= self._eta
        return list(reversed(levels))
