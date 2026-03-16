"""LLAMBO backend via OptunaHub — LLM-enhanced Bayesian optimization."""

from __future__ import annotations

import json
import os
import logging
import time
from pathlib import Path
from typing import Any

import optuna
from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend
from autoresearch_automl.core.search_space import configspace_to_optuna

logger = logging.getLogger(__name__)


class LLMCallLogger:
    """Logs every LLM call by patching ``OpenAI_interface.ask_base``.

    All LLAMBO LLM calls (surrogate, acquisition, generative) funnel through
    ``ask_base``, so a single class-level patch captures everything — no need
    to chase lazily-created OpenAI clients.
    """

    def __init__(self, log_dir: Path, backend_name: str = "llambo", seed: int = 0):
        self._log_dir = log_dir
        self._backend_name = backend_name
        self._seed = seed
        self.trial_id: int = 0  # set by backend before each suggest()

    def _trial_path(self) -> Path:
        return self._log_dir / f"{self._backend_name}_seed{self._seed}_trial{self.trial_id}.jsonl"

    def _thinking_path(self) -> Path:
        return self._log_dir / f"{self._backend_name}_seed{self._seed}_trial{self.trial_id}_thinking.txt"

    def install(self, openai_interface_cls: type) -> None:
        """Monkey-patch ``ask_base`` on the class to log all LLM calls.

        Also patches the OpenAI client's create method to capture
        reasoning_content (thinking traces) from vLLM responses.
        """
        original = openai_interface_cls.ask_base

        if getattr(original, "_llambo_patched", False):
            return  # already installed

        logger_ref = self  # capture for trial_id + log_dir access

        def _logged_ask_base(
            self_inner: Any,
            messages: list,
            ret_dict: dict | None = None,
        ) -> tuple:
            # Patch client.create to capture the last completion's thinking
            client = getattr(self_inner, "client", None)
            thinking_content = []
            if client and not getattr(client, "_thinking_patched", False):
                orig_create = client.chat.completions.create

                def _create_with_thinking(*args: Any, **kwargs: Any) -> Any:
                    completion = orig_create(*args, **kwargs)
                    msg = completion.choices[0].message if completion.choices else None
                    # vLLM 0.17+ with --reasoning-parser: thinking in model_extra["reasoning"]
                    extra = (getattr(msg, "model_extra", {}) or {}) if msg else {}
                    thinking = extra.get("reasoning") or getattr(msg, "reasoning_content", None) if msg else None
                    if thinking:
                        thinking_content.append(thinking)
                    return completion

                client.chat.completions.create = _create_with_thinking
                client._thinking_patched = True  # type: ignore[attr-defined]

            t0 = time.time()
            result = original(self_inner, messages, ret_dict)
            elapsed = time.time() - t0

            record = {
                "timestamp": t0,
                "elapsed_s": round(elapsed, 3),
                "pid": os.getpid(),
                "model": getattr(self_inner, "model", ""),
                "trial_id": logger_ref.trial_id,
                "messages": [
                    {"role": m.get("role", ""), "content": m.get("content", "")}
                    for m in messages
                ],
                "response": result[0] if result else None,
            }
            with open(logger_ref._trial_path(), "a") as f:
                f.write(json.dumps(record) + "\n")

            # Write thinking trace to separate file (append, multiple calls per trial)
            if thinking_content:
                with open(logger_ref._thinking_path(), "a") as f:
                    f.write("\n--- LLM call ---\n")
                    f.write(thinking_content[-1])
                    f.write("\n")
                thinking_content.clear()

            return result

        _logged_ask_base._llambo_patched = True  # type: ignore[attr-defined]
        openai_interface_cls.ask_base = _logged_ask_base
        logger.info("Patched OpenAI_interface.ask_base → %s", self._log_dir)

    def log_trial_result(self, trial_id: int, val_bpb: float | None, error: str | None) -> None:
        """Append a summary entry with the trial's result to the trial's log file."""
        record = {
            "timestamp": time.time(),
            "type": "trial_result",
            "trial_id": trial_id,
            "val_bpb": val_bpb,
            "error": error,
        }
        with open(self._trial_path(), "a") as f:
            f.write(json.dumps(record) + "\n")

DEFAULT_TASK_DESCRIPTION = """\
Optimizing a GPT-2 scale transformer for language modeling on climbmix-400b-shuffle.
The model uses a Muon+AdamW optimizer with weight decay and learning rate scheduling.
Architecture hyperparameters control model depth, width (via aspect ratio and head dim), \
and attention patterns. Optimization hyperparameters control learning rates for different \
parameter groups (embedding, unembedding, matrix, scalar), weight decay, and batch sizes.
Goal: minimize val_bpb (validation bits-per-byte).
Training runs on a single H200 GPU (141GB VRAM, {available_vram} available after vLLM) with a fixed time budget of {budget}s.
VRAM is a soft constraint — large models with high DEPTH and large DEVICE_BATCH_SIZE can OOM.
"""

# Maximum number of failed configs to include in the task description prompt
MAX_FAILURE_CONTEXT = 10


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
        n_initial_samples: int = 2,
        log_dir: Path | None = None,
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
        self._log_dir = log_dir
        self._llm_logger: LLMCallLogger | None = None
        self._failed_configs: list[dict[str, Any]] = []
        self._base_task_desc: str | None = None
        self._trial_id: int = 0

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

        import sys

        # Find the llm module optunahub loaded (full path varies)
        llm_mod = next(
            mod for name, mod in sys.modules.items()
            if name.endswith("llambo.llm") and mod is not None
        )

        # Patch OpenAI client creation to always inject max_tokens=16384
        # (OptunaHub LLAMBO doesn't set max_tokens, which is fine for
        # non-thinking models but we want consistency across all backends)
        self._patch_max_tokens(llm_mod.OpenAI_interface)

        # Patch ask_base for LLM call logging
        if self._log_dir is not None:
            self._llm_logger = LLMCallLogger(self._log_dir, backend_name="llambo", seed=seed)
            self._llm_logger.install(llm_mod.OpenAI_interface)

        available_vram = os.environ.get("AVAILABLE_VRAM", "~120GB")
        task_desc = self._task_description or DEFAULT_TASK_DESCRIPTION.format(
            budget=self._max_budget,
            available_vram=available_vram,
        )
        self._base_task_desc = task_desc

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

    @staticmethod
    def _patch_max_tokens(openai_interface_cls: type) -> None:
        """Patch OpenAI_interface to inject max_tokens=16384 into all LLM calls."""
        original_init = openai_interface_cls.__init__
        if getattr(original_init, "_max_tokens_patched", False):
            return

        def _patched_init(self_inner: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self_inner, *args, **kwargs)
            # Wrap client.create to inject max_tokens
            client = getattr(self_inner, "client", None)
            if client and not getattr(client, "_max_tokens_patched", False):
                orig_create = client.chat.completions.create

                def _create_with_max_tokens(*a: Any, **kw: Any) -> Any:
                    if "max_tokens" not in kw:
                        kw["max_tokens"] = 16384
                    return orig_create(*a, **kw)

                client.chat.completions.create = _create_with_max_tokens
                client._max_tokens_patched = True  # type: ignore[attr-defined]

        _patched_init._max_tokens_patched = True  # type: ignore[attr-defined]
        openai_interface_cls.__init__ = _patched_init
        logger.info("Patched OpenAI_interface to inject max_tokens=16384")

    def _build_task_description(self) -> str:
        """Build task description with failure context appended."""
        desc = self._base_task_desc
        if not self._failed_configs:
            return desc
        # Use last N failures to avoid prompt bloat
        recent = self._failed_configs[-MAX_FAILURE_CONTEXT:]
        lines = []
        for entry in recent:
            params = ", ".join(f"{k}={v}" for k, v in entry["config"].items())
            error = entry.get("error", "unknown error")
            lines.append(f"- {params} → {error}")
        desc += (
            "\nThe following configurations failed during evaluation:\n"
            + "\n".join(lines)
            + "\nAvoid similar configurations.\n"
        )
        return desc

    def suggest(self) -> tuple[dict[str, Any], float]:
        from autoresearch_automl.core.search_space import POWER_OF_2_HPS, snap_to_power_of_2

        # Set trial_id on the logger so individual LLM calls get tagged
        if self._llm_logger is not None:
            self._llm_logger.trial_id = self._trial_id

        # Update task description with failure context before each suggestion.
        # Must update both the sampler's attribute (used when LLAMBO_instance is
        # first created) AND the live LLAMBO_instance's task_context dict (used
        # by surrogate/acquisition prompts on every call).
        if self._failed_configs and self._study is not None:
            desc = self._build_task_description()
            sampler = self._study.sampler
            sampler.custom_task_description = desc
            if getattr(sampler, "LLAMBO_instance", None) is not None:
                sampler.LLAMBO_instance.task_context["custom_task_description"] = desc

        try:
            trial = self._study.ask()
        except ValueError as e:
            # LLAMBO can produce NaN suggestions when the surrogate degrades;
            # fall back to a fresh random trial.
            logger.warning("study.ask() failed (%s), falling back to random trial", e)
            trial = self._study.ask()
        self._pending_trial = trial

        config = {}
        for hp_name, mapping in self._optuna_mapping.items():
            method = getattr(trial, mapping["method"])
            try:
                config[hp_name] = method(**mapping["kwargs"])
            except (ValueError, KeyError):
                # NaN from LLAMBO sampler — use midpoint of the range
                kw = mapping["kwargs"]
                if mapping["method"] in ("suggest_float", "suggest_int"):
                    config[hp_name] = (kw["low"] + kw["high"]) / 2
                    if mapping["method"] == "suggest_int":
                        config[hp_name] = int(config[hp_name])
                elif mapping["method"] == "suggest_categorical":
                    config[hp_name] = kw["choices"][0]
                else:
                    config[hp_name] = 0
                logger.warning("NaN for %s, using fallback %s", hp_name, config[hp_name])

        # Snap power-of-2 HPs and update Optuna's stored values so
        # LLAMBO's prompt history matches what actually runs
        snapped = snap_to_power_of_2(config)
        for hp_name in POWER_OF_2_HPS:
            if hp_name in snapped and snapped[hp_name] != config.get(hp_name):
                self._study._storage.set_trial_param(
                    trial._trial_id, hp_name, snapped[hp_name],
                    trial.distributions[hp_name],
                )
        config = snapped

        return config, self._max_budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        if self._pending_trial is None:
            logger.warning("tell() called without a pending trial")
            return

        val = results.get(self._objectives[0], float("inf"))
        if val == float("inf"):
            # Report as FAIL so the surrogate only sees real performance values
            self._study.tell(
                self._pending_trial,
                state=optuna.trial.TrialState.FAIL,
            )
            # Track the failure with error context for prompt injection
            error_msg = results.get("_error", "unknown error")
            # Shorten common OOM errors
            if "out of memory" in str(error_msg).lower():
                error_msg = "GPU out-of-memory"
            self._failed_configs.append({"config": config, "error": error_msg})
        else:
            self._study.tell(self._pending_trial, values=val)

        self._pending_trial = None

        # Log trial result summary
        if self._llm_logger is not None:
            val_bpb = results.get(self._objectives[0]) if val != float("inf") else None
            error = results.get("_error") if val == float("inf") else None
            self._llm_logger.log_trial_result(self._trial_id, val_bpb, error)
        self._trial_id += 1

    def incumbents(self) -> list[dict[str, Any]]:
        if self._study is None:
            return []
        try:
            return [self._study.best_trial.params]
        except ValueError:
            return []

    def seed_trial(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        """Inject baseline as a completed frozen trial (bypasses ask-before-tell)."""
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
        self._trial_id += 1  # advance past baseline so LLM logs get correct trial_id
        logger.info("Seeded baseline trial into LLAMBO study: %s", results)

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
                # Track failure for prompt injection
                error_msg = results.get("_error", "unknown error")
                if "out of memory" in str(error_msg).lower():
                    error_msg = "GPU out-of-memory"
                self._failed_configs.append({"config": config, "error": error_msg})
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
        # Sync trial_id so new trials get correct numbers after resume
        self._trial_id = len(history)
        logger.info(
            "Replayed %d trials into LLAMBO study (%d failures tracked), trial_id=%d",
            len(history), len(self._failed_configs), self._trial_id,
        )

    @property
    def study(self) -> optuna.Study | None:
        return self._study
