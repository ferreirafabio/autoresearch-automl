"""LLAMBO orchestrator — adapted from original paper code.

Manages the surrogate model and acquisition function to propose
configurations. Does NOT own the evaluation loop (that's the backend's job).

Adapted from: tennisonliu/LLAMBO (llambo.py)
Changes: removed init_f/bbox_eval_f, sample_configuration() replaces optimize(),
added update_observations() for external tell() calls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from autoresearch_automl.backends.llambo_original.acquisition_function import LLM_ACQ
from autoresearch_automl.backends.llambo_original.discriminative_sm import LLM_DIS_SM

logger = logging.getLogger(__name__)


class LLAMBO:
    """LLAMBO optimizer — coordinates acquisition + discriminative surrogate.

    Usage:
        llambo = LLAMBO(task_context, ...)
        llambo.update_observations(config_dict, fval)  # seed baseline
        config = llambo.sample_configuration()  # get next suggestion
        # ... evaluate config externally ...
        llambo.update_observations(config_dict, fval)  # report result
        config = llambo.sample_configuration()  # next suggestion
    """

    def __init__(
        self,
        task_context: dict,
        n_candidates: int = 10,
        n_templates: int = 1,
        n_gens: int = 5,
        alpha: float = -0.2,
        chat_engine: str | None = None,
        client: Any = None,
        max_tokens: int = 500,
        timeout: float = 600.0,
        prompt_setting: str | None = None,
        shuffle_features: bool = False,
        llm_call_logger: Any = None,
    ):
        self.task_context = task_context
        self.lower_is_better = task_context["lower_is_better"]
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = n_gens
        self.alpha = alpha

        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame()

        # Column order (set from first observation)
        self._hp_names: list[str] | None = None

        # Initialize surrogate model (discriminative — uses actual metric values)
        self.surrogate_model = LLM_DIS_SM(
            task_context,
            n_gens,
            self.lower_is_better,
            n_templates=n_templates,
            chat_engine=chat_engine,
            prompt_setting=prompt_setting,
            shuffle_features=shuffle_features,
            client=client,
            max_tokens=max_tokens,
            timeout=timeout,
            llm_call_logger=llm_call_logger,
        )

        # Initialize acquisition function
        self.acq_func = LLM_ACQ(
            task_context,
            n_candidates,
            n_templates,
            self.lower_is_better,
            chat_engine=chat_engine,
            prompt_setting=prompt_setting,
            shuffle_features=shuffle_features,
            client=client,
            max_tokens=max_tokens,
            timeout=timeout,
            llm_call_logger=llm_call_logger,
        )

        logger.info(
            "[LLAMBO] Initialized: n_candidates=%d, n_templates=%d, n_gens=%d, "
            "alpha=%.2f, lower_is_better=%s",
            n_candidates,
            n_templates,
            n_gens,
            alpha,
            self.lower_is_better,
        )

    def update_observations(self, config_dict: dict, fval: float) -> None:
        """Add an observation to the history.

        Args:
            config_dict: HP name → value (must use LLAMBO-internal encoding,
                         e.g. ordinal integers for WINDOW_PATTERN).
            fval: Observed performance value (or penalty for failures).
        """
        if self._hp_names is None:
            self._hp_names = list(config_dict.keys())

        # Ensure consistent column order
        ordered = {k: config_dict[k] for k in self._hp_names}

        new_config = pd.DataFrame([ordered])
        new_fval = pd.DataFrame([{"score": fval}])

        if self.observed_configs.empty:
            self.observed_configs = new_config
        else:
            self.observed_configs = pd.concat(
                [self.observed_configs, new_config], axis=0, ignore_index=True
            )

        if self.observed_fvals.empty:
            self.observed_fvals = new_fval
        else:
            self.observed_fvals = pd.concat(
                [self.observed_fvals, new_fval], axis=0, ignore_index=True
            )

        # Update num_samples in task_context
        self.task_context["num_samples"] = len(self.observed_fvals)

        logger.info(
            "[LLAMBO] Observation added: fval=%.6f (total: %d observations)",
            fval,
            len(self.observed_fvals),
        )

    def sample_configuration(self) -> dict:
        """Sample the next configuration to evaluate.

        Returns:
            dict mapping HP names to values.

        Raises:
            RuntimeError: if acquisition or surrogate fails completely.
        """
        if self.observed_fvals.empty:
            raise RuntimeError("Need at least one observation before sampling")

        # Step 1: Generate candidate points via acquisition function
        candidate_points, acq_time = self.acq_func.get_candidate_points(
            self.observed_configs,
            self.observed_fvals[["score"]],
            alpha=self.alpha,
        )

        logger.info(
            "[LLAMBO] ACQ generated %d candidates in %.1fs",
            candidate_points.shape[0],
            acq_time,
        )

        # Ensure candidates have the same columns as observed configs
        # (ACQ might return different column order or extra/missing columns)
        expected_cols = set(self._hp_names)
        if not candidate_points.empty:
            actual_cols = set(candidate_points.columns)
            if actual_cols != expected_cols:
                logger.warning(
                    "[LLAMBO] Candidate columns mismatch: expected %s, got %s",
                    expected_cols,
                    actual_cols,
                )
                # Keep only columns that exist in both, add missing with defaults
                for col in expected_cols - actual_cols:
                    candidate_points[col] = self.observed_configs[col].median()
                candidate_points = candidate_points[self._hp_names]

        # Pad with random candidates if ACQ didn't produce enough
        min_candidates = max(self.n_candidates, 5)
        if candidate_points.shape[0] < min_candidates:
            n_random = min_candidates - candidate_points.shape[0]
            logger.warning(
                "[LLAMBO] ACQ only produced %d candidates, adding %d random",
                candidate_points.shape[0],
                n_random,
            )
            random_candidates = self._sample_random_candidates(n_random)
            candidate_points = pd.concat(
                [candidate_points, random_candidates], ignore_index=True
            )

        # Step 2: Select best candidate using discriminative surrogate
        sel_candidate, sm_time = self.surrogate_model.select_query_point(
            self.observed_configs,
            self.observed_fvals[["score"]],
            candidate_points,
        )

        logger.info("[LLAMBO] SM selected candidate in %.1fs", sm_time)

        return sel_candidate.to_dict("records")[0]

    def _sample_random_candidates(self, n: int) -> pd.DataFrame:
        """Sample random candidates from HP ranges as fallback."""
        rng = np.random.RandomState()
        constraints = self.task_context["hyperparameter_constraints"]
        samples = []
        for _ in range(n):
            sample = {}
            for hp_name, (hp_type, transform, search_range) in constraints.items():
                if hp_type == "int":
                    if transform == "log":
                        log_val = rng.uniform(
                            np.log(search_range[0]), np.log(search_range[1])
                        )
                        sample[hp_name] = int(round(np.exp(log_val)))
                    else:
                        sample[hp_name] = rng.randint(
                            search_range[0], search_range[1] + 1
                        )
                elif hp_type == "float":
                    if transform == "log":
                        log_val = rng.uniform(
                            np.log(search_range[0] + 1e-10),
                            np.log(search_range[1] + 1e-10),
                        )
                        sample[hp_name] = np.exp(log_val)
                    else:
                        sample[hp_name] = rng.uniform(
                            search_range[0], search_range[1]
                        )
                elif hp_type == "ordinal":
                    sample[hp_name] = rng.choice(search_range)
                else:
                    sample[hp_name] = search_range[0]
            samples.append(sample)
        return pd.DataFrame(samples)
