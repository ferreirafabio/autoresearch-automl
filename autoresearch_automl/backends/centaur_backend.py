"""Centaur backend: CMA-ES guided LLM optimization (CMA-ES as critic, LLM as actor)."""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import optuna
from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend
from autoresearch_automl.core.search_space import configspace_to_optuna

logger = logging.getLogger(__name__)

SUGGEST_PROMPT = """\
You are optimizing hyperparameters for a GPT-2 scale transformer trained on climbmix-400b-shuffle.
Goal: minimize val_bpb. GPU: H200 with {available_vram} available.

Search space:
{space_description}

{cma_analysis}

Previous evaluations (last 20):
{history}

{incumbent_info}

Use CMA-ES's analysis as guidance:
- If CMA-ES is converging (low sigma), exploit near its mean
- If sigma is large, CMA-ES is still exploring — help by using your transformer knowledge
- You can diverge from CMA-ES if you see patterns it might miss (e.g., OOM-prone regions, known good ratios)

Respond ONLY with a valid JSON object mapping HP names to values."""


class CentaurBackend(HPOBackend):
    """Centaur: CMA-ES guided LLM optimization.

    CMA-ES (critic) learns the optimization landscape — covariance structure,
    convergence direction. LLM (actor) uses CMA-ES's insights + transformer
    domain knowledge to suggest configs. CMA-ES runs every trial (always
    ask/tell) and learns from all results, including LLM-suggested ones.
    """

    FAILURE_PENALTY = 100.0

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        log_dir: Path | None = None,
        llm_ratio: float = 0.3,
        llm_warmup: int = 10,
    ):
        self._model = model
        self._log_dir = log_dir
        self._llm_ratio = llm_ratio
        self._llm_warmup = llm_warmup

        self._llm_client = None
        self._study: optuna.Study | None = None
        self._space: ConfigurationSpace | None = None
        self._optuna_mapping: dict[str, dict] = {}
        self._results: list[tuple[dict, float, dict]] = []
        self._pending_trial: optuna.Trial | None = None
        self._trial_count: int = 0
        self._active_source: str = "cma"
        self._pending_llm_call: dict | None = None
        self._objectives: list[str] = []
        self._max_budget: float = 300.0
        self._seed: int = 0
        self._replaying: bool = False

    @property
    def name(self) -> str:
        return "centaur"

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
        self._seed = seed
        self._optuna_mapping = configspace_to_optuna(space)

        sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)
        self._study = optuna.create_study(
            study_name="centaur",
            sampler=sampler,
            direction="minimize",
        )

    def suggest(self) -> tuple[dict[str, Any], float]:
        trial = self._study.ask()
        self._pending_trial = trial

        # Get CMA-ES suggested config
        cma_config = {}
        for hp_name, mapping in self._optuna_mapping.items():
            method = getattr(trial, mapping["method"])
            cma_config[hp_name] = method(**mapping["kwargs"])

        # Decide if this is an LLM trial
        use_llm = self._should_use_llm()

        if not use_llm:
            self._active_source = "cma"
            self._trial_count += 1
            return cma_config, self._max_budget

        # LLM trial: get CMA-ES analysis and build prompt
        self._active_source = "llm"
        try:
            config = self._suggest_llm(cma_config)
            # Override the pending trial's params so CMA-ES sees what was actually evaluated
            self._override_trial_params(trial, config)
            self._trial_count += 1
            return config, self._max_budget
        except Exception as e:
            # Fail HARD on EVERY exception. Any silent fallback to pure-CMA
            # would log CMA-only trials as if they were Centaur,
            # indistinguishable in trials.jsonl downstream. Previous policy
            # tolerated non-transport exceptions (bad JSON, schema mismatch,
            # value errors) via a CMA fallback path; that tolerated path
            # corrupted ~580 trials in
            # gemini31pro_benchmark/centaur_gemini_3_1_pro_preview/seed_3,4
            # on 2026-05-08 when a 400 API_KEY_INVALID slipped through as a
            # non-standard exception, and we don't want a repeat from
            # Claude-Code 5h/7d rate-limit errors either. The rule is now:
            # no fallback, ever; let the orchestrator log the trial as
            # failed and move on.
            logger.error(
                "LLM call failed hard (%s: %s): refusing to fall back to "
                "pure CMA (would silently pollute results).",
                type(e).__name__, e,
            )
            raise

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        if self._pending_trial is None:
            logger.warning("tell() called without a pending trial")
            return

        values = [results.get(obj, float("inf")) for obj in self._objectives]
        if any(v == float("inf") for v in values):
            self._study.tell(self._pending_trial, values=self.FAILURE_PENALTY)
        else:
            self._study.tell(self._pending_trial, values=values[0])

        self._results.append((config, budget, dict(results)))
        self._pending_trial = None

        if not self._replaying:
            self._flush_llm_log(results)

    def incumbents(self) -> list[dict[str, Any]]:
        if not self._results:
            return []
        obj = self._objectives[0]
        best = min(self._results, key=lambda x: x[2].get(obj, float("inf")))
        return [best[0]]

    def seed_trial(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        distributions = self._build_distributions()
        values = [results.get(obj, float("inf")) for obj in self._objectives]
        if any(v == float("inf") for v in values):
            state = optuna.trial.TrialState.FAIL
            trial_values = None
        else:
            state = optuna.trial.TrialState.COMPLETE
            trial_values = [values[0]]

        frozen = optuna.trial.create_trial(
            params=config,
            distributions=distributions,
            values=trial_values,
            state=state,
        )
        self._study.add_trial(frozen)
        self._results.append((config, budget, dict(results)))
        self._trial_count += 1
        logger.info("Seeded baseline trial into Centaur: %s", results)

    def replay(self, history: list[tuple[dict, float, dict]]) -> None:
        self._replaying = True
        distributions = self._build_distributions()
        for config, budget, results in history:
            values = [results.get(obj, float("inf")) for obj in self._objectives]
            if any(v == float("inf") for v in values):
                state = optuna.trial.TrialState.FAIL
                trial_values = None
            else:
                state = optuna.trial.TrialState.COMPLETE
                trial_values = [values[0]]

            frozen = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                values=trial_values,
                state=state,
            )
            self._study.add_trial(frozen)
            self._results.append((config, budget, dict(results)))
            self._trial_count += 1
        self._replaying = False
        logger.info("Replayed %d trials into Centaur study", len(history))

    # ── Private helpers ──────────────────────────────────────────────

    def _should_use_llm(self) -> bool:
        if self._trial_count < self._llm_warmup:
            return False
        if self._llm_ratio <= 0:
            return False
        interval = max(1, round(1.0 / self._llm_ratio))
        return self._trial_count % interval == 0

    def _suggest_llm(self, cma_config: dict) -> dict[str, Any]:
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(timeout=1800.0)  # 30 min; Kimi K2.6 decode can take >10min for long outputs

        cma_state = self._extract_cma_state()
        cma_analysis = self._format_cma_analysis(cma_state, cma_config)

        available_vram = os.environ.get("AVAILABLE_VRAM", "~120GB")
        prompt = SUGGEST_PROMPT.format(
            space_description=self._format_space(),
            cma_analysis=cma_analysis,
            history=self._format_history() or "(no evaluations yet)",
            incumbent_info=self._format_incumbent(),
            available_vram=available_vram,
        )

        messages = [{"role": "user", "content": prompt}]

        t0 = time.time()
        # Use 16384 max_tokens: Gemini's OpenAI-compatible endpoint counts
        # thinking tokens against max_tokens, so 2048 truncates responses
        # when the prompt is large (covariance matrix, history, etc.)
        response = self._llm_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=16384,
        )
        elapsed = time.time() - t0

        msg = response.choices[0].message
        extra = getattr(msg, "model_extra", {}) or {}
        thinking = extra.get("reasoning") or getattr(msg, "reasoning_content", None) or ""
        text = (msg.content or "").strip()
        if not text and thinking:
            text = thinking.strip()
            thinking = ""

        self._pending_llm_call = {
            "timestamp": t0,
            "elapsed_s": round(elapsed, 3),
            "pid": os.getpid(),
            "model": self._model,
            "trial_id": self._trial_count,
            "source": "llm",
            "messages": messages,
            "response": text,
            "thinking": thinking,
        }

        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        config = json.loads(text)

        return self._clamp_to_space(config)

    def _extract_cma_state(self) -> dict | None:
        """Extract CMA-ES optimizer internals from Optuna's internal storage."""
        try:
            trials = self._study.trials
            if not trials:
                return None

            # Find the CMA optimizer object by concatenating split parts.
            # Optuna's CmaEsSampler splits the pickled optimizer across multiple
            # system_attrs keys: cma:optimizer:0, cma:optimizer:1, ...
            for trial in reversed(trials):
                parts = {}
                for key, val in trial.system_attrs.items():
                    if key.startswith("cma:optimizer:"):
                        idx = int(key.split(":")[-1])
                        parts[idx] = val
                if not parts:
                    continue
                combined = "".join(parts[i] for i in sorted(parts.keys()))
                optimizer = pickle.loads(bytes.fromhex(combined))
                mean = optimizer._mean.tolist()
                sigma = float(optimizer._sigma)

                # Get top completed trials
                completed = [
                    t for t in trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                completed.sort(key=lambda t: t.value if t.value is not None else float("inf"))
                top_trials = [
                    {"params": t.params, "value": t.value}
                    for t in completed[:5]
                ]

                # Covariance matrix C (indices = alphabetically sorted HP names)
                cov = None
                if hasattr(optimizer, "_C"):
                    cov = optimizer._C.tolist()

                return {
                    "mean_raw": mean,
                    "sigma": sigma,
                    "cov": cov,
                    "top_trials": top_trials,
                    "n_trials": len(trials),
                }
            return None
        except Exception as e:
            logger.debug("Could not extract CMA state: %s", e)
            return None

    def _format_cma_analysis(self, cma_state: dict | None, cma_config: dict) -> str:
        native_cma = self._to_native_types(cma_config)
        if cma_state is None:
            return (
                "## CMA-ES Advisor\n"
                "CMA-ES is still initializing (not enough trials for covariance estimation).\n\n"
                f"CMA-ES's next suggestion: {json.dumps(native_cma, indent=2)}"
            )

        lines = ["## CMA-ES Advisor Analysis"]
        lines.append(
            f"CMA-ES has run {cma_state['n_trials']} trials "
            f"(sigma={cma_state['sigma']:.4f}, smaller=more focused)."
        )
        lines.append(f"\nCMA-ES's next suggestion: {json.dumps(native_cma, indent=2)}")

        if cma_state["top_trials"]:
            lines.append("\nTop 5 configs found by CMA-ES:")
            for i, t in enumerate(cma_state["top_trials"], 1):
                native_params = self._to_native_types(t["params"])
                lines.append(f"  {i}. val_bpb={t['value']:.4f}: {json.dumps(native_params)}")

        # Full covariance matrix (rows/cols = alphabetically sorted continuous HP names)
        if cma_state.get("cov") is not None:
            import numpy as np
            # CMA-ES only handles continuous HPs; categoricals are sampled independently
            hp_names = sorted(
                name for name, m in self._optuna_mapping.items()
                if m["method"] != "suggest_categorical"
            )
            C = np.array(cma_state["cov"])
            if len(hp_names) == C.shape[0]:
                lines.append(
                    f"\nCMA-ES covariance matrix ({C.shape[0]}x{C.shape[0]}). "
                    "This matrix describes how the hyperparameters co-vary in the learned search distribution. "
                    "Positive off-diagonal entries mean those HPs tend to increase together in good regions; "
                    "negative entries mean they trade off. "
                    f"Row/column order: {', '.join(f'{i}={name}' for i, name in enumerate(hp_names))}."
                )
                lines.append(np.array2string(C, precision=4, suppress_small=True))

        return "\n".join(lines)

    def _override_trial_params(self, trial: optuna.Trial, config: dict) -> None:
        """Override the pending trial's params so CMA-ES sees the LLM config."""
        distributions = self._build_distributions()
        storage = self._study._storage
        trial_id = trial._trial_id

        for hp_name, val in config.items():
            if hp_name in distributions:
                dist = distributions[hp_name]
                # For categoricals, Optuna stores the index internally
                if isinstance(dist, optuna.distributions.CategoricalDistribution):
                    if val in dist.choices:
                        internal_val = dist.choices.index(val)
                    else:
                        continue  # skip invalid categorical values
                    storage.set_trial_param(trial_id, hp_name, internal_val, dist)
                else:
                    storage.set_trial_param(trial_id, hp_name, val, dist)

    def _build_distributions(self) -> dict:
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
        return distributions

    def _format_space(self) -> str:
        lines = []
        for hp in self._space.values():
            lines.append(f"  {hp.name}: {hp}")
        return "\n".join(lines)

    def _format_history(self) -> str:
        if not self._results:
            return ""
        lines = []
        for config, _budget, results in self._results[-20:]:
            native_config = self._to_native_types(config)
            obj_val = results.get(self._objectives[0])
            error = results.get("_error")
            if obj_val is not None:
                lines.append(f"  {json.dumps(native_config)} → val_bpb={obj_val}")
            elif error:
                lines.append(f"  {json.dumps(native_config)} → CRASHED ({error})")
            else:
                lines.append(f"  {json.dumps(native_config)} → FAILED")
        return "\n".join(lines)

    def _format_incumbent(self) -> str:
        if not self._results:
            return ""
        best = min(self._results, key=lambda x: x[2].get(self._objectives[0], float("inf")))
        obj_val = best[2].get(self._objectives[0], "?")
        native = self._to_native_types(best[0])
        return f"Current best config (val_bpb={obj_val}):\n{json.dumps(native, indent=2)}"

    def _clamp_to_space(self, config: dict) -> dict:
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

        clamped = {}
        for hp in self._space.values():
            if hp.name not in config:
                clamped[hp.name] = hp.default_value
                continue
            val = config[hp.name]
            if isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter)):
                val = type(hp.default_value)(val)
                val = max(hp.lower, min(hp.upper, val))
            clamped[hp.name] = val
        return clamped

    @staticmethod
    def _to_native_types(config: dict) -> dict:
        import numpy as np
        native = {}
        for k, v in config.items():
            if isinstance(v, np.integer):
                native[k] = int(v)
            elif isinstance(v, np.floating):
                native[k] = float(v)
            elif isinstance(v, (np.str_, np.bytes_)):
                native[k] = str(v)
            else:
                native[k] = v
        return native

    def _flush_llm_log(self, results: dict[str, float]) -> None:
        if self._log_dir is None or self._pending_llm_call is None:
            return
        record = self._pending_llm_call
        thinking = record.pop("thinking", "")
        record["val_bpb"] = results.get(self._objectives[0])
        record["error"] = results.get("_error")

        prefix = f"centaur_seed{self._seed}_trial{self._trial_count}"

        with open(self._log_dir / f"{prefix}.jsonl", "w") as f:
            f.write(json.dumps(record, indent=2) + "\n")

        if thinking:
            with open(self._log_dir / f"{prefix}_thinking.txt", "w") as f:
                f.write(thinking)

        self._pending_llm_call = None
