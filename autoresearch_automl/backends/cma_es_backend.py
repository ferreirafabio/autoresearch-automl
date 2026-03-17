"""CMA-ES backend via Optuna's CmaESSampler."""

from __future__ import annotations

from typing import Any

import optuna
from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.optuna_backend import OptunaBackend


class CmaESBackend(OptunaBackend):
    """Optuna backend using CMA-ES sampler instead of TPE."""

    @property
    def name(self) -> str:
        return "cma_es"

    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self._sampler = optuna.samplers.CmaEsSampler(seed=seed)
        super().configure(space, objectives, budget_range=budget_range, seed=seed, **kwargs)
