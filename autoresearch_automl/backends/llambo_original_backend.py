"""LLAMBO Original backend — faithful adaptation of the paper's code.

Key advantages over OptunaHub LLAMBO port:
1. Discriminative SM uses actual metric values (## 0.970 ##), not binary 0/1
2. WINDOW_PATTERN included as ordinal in LLM prompts (not delegated to random)
3. Failed trials get penalty values visible to surrogate (not silently dropped)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)

# Ordinal encoding for WINDOW_PATTERN
WINDOW_PATTERN_ENCODING = {
    "SSSL": 0,
    "SSLL": 1,
    "SLSL": 2,
    "LLLL": 3,
    "SSSS": 4,
    "LSSL": 5,
}
WINDOW_PATTERN_DECODING = {v: k for k, v in WINDOW_PATTERN_ENCODING.items()}

DEFAULT_TASK_DESCRIPTION = """\
Optimizing a GPT-2 scale transformer for language modeling on climbmix-400b-shuffle.
The model uses a Muon+AdamW optimizer with weight decay and learning rate scheduling.
Architecture hyperparameters control model depth, width (via aspect ratio and head dim), \
and attention patterns. Optimization hyperparameters control learning rates for different \
parameter groups (embedding, unembedding, matrix, scalar), weight decay, and batch sizes.
Goal: minimize val_bpb (validation bits-per-byte).
Training runs on a single H200 GPU (141GB VRAM, {available_vram} available after vLLM) \
with a fixed time budget of {budget}s.
VRAM is a soft constraint — large models with high DEPTH and large DEVICE_BATCH_SIZE can OOM.
WINDOW_PATTERN is encoded as ordinal: 0=SSSL, 1=SSLL, 2=SLSL, 3=LLLL, 4=SSSS, 5=LSSL \
(controls sliding window vs full attention pattern per layer).\
"""


class LLMCallLogger:
    """Logs every LLM call (surrogate + acquisition) to per-trial JSONL files."""

    def __init__(self, log_dir: Path, backend_name: str = "llambo_original", seed: int = 0):
        self._log_dir = log_dir
        self._backend_name = backend_name
        self._seed = seed
        self.trial_id: int = 0

    def _trial_path(self) -> Path:
        return (
            self._log_dir
            / f"{self._backend_name}_seed{self._seed}_trial{self.trial_id}.jsonl"
        )

    def _thinking_path(self) -> Path:
        return (
            self._log_dir
            / f"{self._backend_name}_seed{self._seed}_trial{self.trial_id}_thinking.txt"
        )

    def log_call(
        self,
        component: str,
        messages: list[dict],
        responses: list[str],
        thinking: list[str],
        elapsed: float,
    ) -> None:
        """Log an LLM call to the trial's JSONL file."""
        record = {
            "timestamp": time.time(),
            "elapsed_s": round(elapsed, 3),
            "pid": os.getpid(),
            "component": component,
            "trial_id": self.trial_id,
            "messages": [
                {"role": m.get("role", ""), "content": m.get("content", "")}
                for m in messages
            ],
            "responses": responses,
        }
        with open(self._trial_path(), "a") as f:
            f.write(json.dumps(record) + "\n")

        if thinking:
            with open(self._thinking_path(), "a") as f:
                for trace in thinking:
                    f.write(f"\n--- {component} call ---\n")
                    f.write(trace)
                    f.write("\n")

    def log_trial_result(
        self, trial_id: int, val_bpb: float | None, error: str | None
    ) -> None:
        """Append trial result summary."""
        record = {
            "timestamp": time.time(),
            "type": "trial_result",
            "trial_id": trial_id,
            "val_bpb": val_bpb,
            "error": error,
        }
        with open(self._trial_path(), "a") as f:
            f.write(json.dumps(record) + "\n")


def _configspace_to_constraints(
    cs: ConfigurationSpace,
) -> dict[str, tuple[str, str, list]]:
    """Convert ConfigSpace to LLAMBO's hyperparameter_constraints format.

    Returns dict mapping HP name → (type, transform, range):
        - int/float: ("int"/"float", "log"/"linear", [low, high])
        - ordinal (WINDOW_PATTERN): ("ordinal", "ordinal", [0, 1, 2, ...])
    """
    constraints = {}
    for hp in cs.values():
        if isinstance(hp, CategoricalHyperparameter):
            if hp.name == "WINDOW_PATTERN":
                # Encode as ordinal
                constraints[hp.name] = (
                    "ordinal",
                    "ordinal",
                    sorted(WINDOW_PATTERN_ENCODING.values()),
                )
            else:
                logger.warning(
                    "Skipping categorical HP %s (not supported in original LLAMBO)",
                    hp.name,
                )
        elif isinstance(hp, UniformIntegerHyperparameter):
            transform = "log" if hp.log else "linear"
            constraints[hp.name] = ("int", transform, [hp.lower, hp.upper])
        elif isinstance(hp, UniformFloatHyperparameter):
            transform = "log" if hp.log else "linear"
            constraints[hp.name] = ("float", transform, [hp.lower, hp.upper])
        else:
            logger.warning("Unknown HP type for %s: %s", hp.name, type(hp))
    return constraints


class LLAMBOOriginalBackend(HPOBackend):
    """Original LLAMBO backend: faithful adaptation of the paper's code.

    Uses discriminative surrogate with actual metric values in few-shot prompts.
    Handles all HPs (including categoricals as ordinals) in the LLM context.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        log_dir: Path | None = None,
        max_tokens: int = 2048,
        timeout: float = 600.0,
        n_candidates: int = 10,
        n_templates: int = 1,
        n_gens: int = 10,
        alpha: float = -0.2,
        task_description: str | None = None,
    ):
        self._model = model
        self._log_dir = log_dir
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._n_candidates = n_candidates
        self._n_templates = n_templates
        self._n_gens = n_gens
        self._alpha = alpha
        self._task_description = task_description
        self._space: ConfigurationSpace | None = None
        self._max_budget: float = 300.0
        self._objectives: list[str] = []
        self._llm_logger: LLMCallLogger | None = None
        self._llambo = None
        self._trial_id: int = 0
        self._observed_fvals_list: list[float] = []  # raw fvals for penalty calc
        self._failure_indices: set[int] = set()  # track which observations are penalties
        self.FAILURE_PENALTY: float = 2.0  # updated dynamically

    @property
    def name(self) -> str:
        return "llambo_original"

    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        from openai import OpenAI

        self._space = space
        self._objectives = objectives
        self._max_budget = budget_range[1] if budget_range else 300.0

        # Create OpenAI client (reads OPENAI_BASE_URL + OPENAI_API_KEY from env)
        client = OpenAI()


        # Setup logging
        if self._log_dir is not None:
            self._llm_logger = LLMCallLogger(
                self._log_dir, backend_name="llambo_original", seed=seed
            )

        # Convert ConfigSpace to LLAMBO format
        constraints = _configspace_to_constraints(space)

        # Count features
        n_cat = sum(1 for v in constraints.values() if v[0] == "ordinal")
        n_num = sum(1 for v in constraints.values() if v[0] in ("int", "float"))
        tot_feats = n_cat + n_num

        # Build task description
        available_vram = os.environ.get("AVAILABLE_VRAM", "~120GB")
        task_desc = self._task_description or DEFAULT_TASK_DESCRIPTION.format(
            budget=self._max_budget,
            available_vram=available_vram,
        )

        task_context = {
            "model": "GPT-2 scale transformer",
            "task": "regression",
            "tot_feats": tot_feats,
            "cat_feats": n_cat,
            "num_feats": n_num,
            "n_classes": 0,
            "num_samples": 0,  # updated as observations come in
            "metric": "val_bpb",
            "lower_is_better": True,
            "hyperparameter_constraints": constraints,
            "custom_task_description": task_desc,
        }

        # Create LLAMBO instance
        from autoresearch_automl.backends.llambo_original.llambo import LLAMBO

        self._llambo = LLAMBO(
            task_context=task_context,
            n_candidates=self._n_candidates,
            n_templates=self._n_templates,
            n_gens=self._n_gens,
            alpha=self._alpha,
            chat_engine=self._model,
            client=client,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            prompt_setting="full_context",
            shuffle_features=False,
            llm_call_logger=self._llm_logger,
        )

        logger.info(
            "LLAMBO Original configured: %d HPs, model=%s, n_gens=%d, max_tokens=%d",
            len(constraints),
            self._model,
            self._n_gens,
            self._max_tokens,
        )

    def _encode_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Encode config for LLAMBO internal use (WINDOW_PATTERN → ordinal)."""
        encoded = {}
        constraints = self._llambo.task_context["hyperparameter_constraints"]
        for hp_name in constraints:
            if hp_name not in config:
                continue
            if hp_name == "WINDOW_PATTERN":
                val = config[hp_name]
                if isinstance(val, str):
                    encoded[hp_name] = WINDOW_PATTERN_ENCODING.get(val, 0)
                else:
                    encoded[hp_name] = int(val)
            else:
                encoded[hp_name] = config[hp_name]
        return encoded

    def _decode_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Decode config from LLAMBO internal (ordinal → WINDOW_PATTERN string)."""
        decoded = dict(config)
        if "WINDOW_PATTERN" in decoded:
            val = decoded["WINDOW_PATTERN"]
            nearest = min(
                WINDOW_PATTERN_DECODING.keys(), key=lambda x: abs(x - float(val))
            )
            decoded["WINDOW_PATTERN"] = WINDOW_PATTERN_DECODING[nearest]
        # Cast int HPs
        constraints = self._llambo.task_context["hyperparameter_constraints"]
        for hp_name, constraint in constraints.items():
            if hp_name in decoded and constraint[0] == "int":
                decoded[hp_name] = int(round(decoded[hp_name]))
        return decoded

    def suggest(self) -> tuple[dict[str, Any], float]:
        from autoresearch_automl.core.search_space import POWER_OF_2_HPS, snap_to_power_of_2

        # Set trial_id on logger
        if self._llm_logger is not None:
            self._llm_logger.trial_id = self._trial_id

        try:
            config = self._llambo.sample_configuration()
            config = self._decode_config(config)
        except Exception as e:
            logger.warning(
                "LLAMBO sampling failed (%s), falling back to random", e
            )
            sample = self._space.sample_configuration()
            config = dict(sample)

        # Snap power-of-2 HPs
        config = snap_to_power_of_2(config)

        # Ensure all space HPs are present with valid values
        config = self._clamp_to_space(config)

        return config, self._max_budget

    def tell(
        self, config: dict[str, Any], budget: float, results: dict[str, float]
    ) -> None:
        val = results.get(self._objectives[0], float("inf"))

        obs_idx = len(self._llambo.observed_fvals)  # index of next observation

        if val == float("inf"):
            # Assign penalty so surrogate sees failures as "very bad"
            if self._observed_fvals_list:
                self.FAILURE_PENALTY = max(self._observed_fvals_list) + 0.5
            fval = self.FAILURE_PENALTY
            self._failure_indices.add(obs_idx)
            logger.info(
                "Trial %d failed, using penalty=%.4f", self._trial_id, fval
            )
        else:
            fval = val
            self._observed_fvals_list.append(fval)

        encoded = self._encode_config(config)
        self._llambo.update_observations(encoded, fval)

        # Log trial result
        if self._llm_logger is not None:
            val_bpb = val if val != float("inf") else None
            error = results.get("_error") if val == float("inf") else None
            self._llm_logger.log_trial_result(self._trial_id, val_bpb, error)

        self._trial_id += 1

    def seed_trial(
        self, config: dict[str, Any], budget: float, results: dict[str, float]
    ) -> None:
        """Inject baseline as an observation (bypasses suggest/tell cycle)."""
        self.tell(config, budget, results)
        logger.info("Seeded baseline trial into LLAMBO Original: %s", results)

    def replay(
        self, history: list[tuple[dict[str, Any], float, dict[str, float]]]
    ) -> None:
        """Replay completed trials to reconstruct state after preemption."""
        for config, budget, results in history:
            val = results.get(self._objectives[0], float("inf"))
            obs_idx = len(self._llambo.observed_fvals)

            if val == float("inf"):
                if self._observed_fvals_list:
                    self.FAILURE_PENALTY = max(self._observed_fvals_list) + 0.5
                fval = self.FAILURE_PENALTY
                self._failure_indices.add(obs_idx)
            else:
                fval = val
                self._observed_fvals_list.append(fval)

            encoded = self._encode_config(config)
            self._llambo.update_observations(encoded, fval)
            self._trial_id += 1

        logger.info(
            "Replayed %d trials into LLAMBO Original, trial_id=%d",
            len(history),
            self._trial_id,
        )

    def incumbents(self) -> list[dict[str, Any]]:
        if not self._observed_fvals_list:
            return []
        # Find best observed non-failure config
        best_idx = None
        best_val = float("inf")
        obs_fvals = self._llambo.observed_fvals
        obs_configs = self._llambo.observed_configs
        for i in range(len(obs_fvals)):
            if i in self._failure_indices:
                continue
            fval = obs_fvals.iloc[i]["score"]
            if fval < best_val:
                best_val = fval
                best_idx = i
        if best_idx is not None:
            config = obs_configs.iloc[best_idx].to_dict()
            return [self._decode_config(config)]
        return []

    def _clamp_to_space(self, config: dict) -> dict:
        """Ensure values fall within ConfigSpace bounds."""
        clamped = {}
        for hp in self._space.values():
            if hp.name not in config:
                clamped[hp.name] = hp.default_value
                continue
            val = config[hp.name]
            if isinstance(hp, UniformIntegerHyperparameter):
                val = int(round(float(val)))
                val = max(hp.lower, min(hp.upper, val))
            elif isinstance(hp, UniformFloatHyperparameter):
                val = float(val)
                val = max(hp.lower, min(hp.upper, val))
            elif isinstance(hp, CategoricalHyperparameter):
                if val not in hp.choices:
                    val = hp.default_value
            clamped[hp.name] = val
        return clamped
