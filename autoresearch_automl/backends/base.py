"""HPO backend protocol: suggest/tell interface for all optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace


class HPOBackend(ABC):
    """Abstract base class for HPO backends.

    All backends implement an ask/tell interface, decoupling the optimizer
    from experiment execution (git ops, GPU scheduling, file I/O).
    """

    @abstractmethod
    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the backend with a search space and objectives.

        Args:
            space: The ConfigurationSpace to optimize over.
            objectives: List of objective names (e.g., ["val_bpb"]).
            budget_range: (min_budget, max_budget) for multi-fidelity, or None.
            seed: Random seed for reproducibility.
        """

    @abstractmethod
    def suggest(self) -> tuple[dict[str, Any], float]:
        """Suggest the next configuration to evaluate.

        Returns:
            Tuple of (config_dict, budget) where config_dict maps HP names
            to values and budget is the time budget in seconds.
        """

    @abstractmethod
    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        """Report the result of an evaluation.

        Args:
            config: The configuration that was evaluated.
            budget: The budget (time) used for evaluation.
            results: Dict mapping objective names to values.
        """

    @abstractmethod
    def incumbents(self) -> list[dict[str, Any]]:
        """Return the best configuration(s) found so far.

        For single-objective: list with one config.
        For multi-objective: Pareto front.
        """

    def seed_trial(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        """Inject a known-good baseline result without going through suggest/tell.

        This allows all backends to start from the same baseline config (e.g.,
        Karpathy's hand-tuned defaults) before their first suggest() call.

        Default implementation calls tell() — backends with ask-before-tell
        requirements should override this.
        """
        self.tell(config, budget, results)

    def replay(self, history: list[tuple[dict[str, Any], float, dict[str, float]]]) -> None:
        """Replay completed trials to reconstruct optimizer state after preemption.

        Default implementation calls tell() for each trial in order.
        Backends with native checkpointing or ask-before-tell requirements
        should override this method.

        Args:
            history: List of (config, budget, results) tuples in chronological order.
        """
        for config, budget, results in history:
            self.tell(config, budget, results)

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier string."""

    @property
    def supports_multi_fidelity(self) -> bool:
        """Whether this backend supports multi-fidelity optimization."""
        return False

    @property
    def supports_multi_objective(self) -> bool:
        """Whether this backend supports multi-objective optimization."""
        return False
