"""LLM greedy hill-climbing backend — Karpathy's original approach as a baseline."""

from __future__ import annotations

import json
import logging
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)

SUGGEST_PROMPT = """\
You are optimizing hyperparameters for a GPT-2 scale transformer trained on climbmix-400b-shuffle.
The goal is to minimize val_bpb (validation bits-per-byte).
Training runs on a single GPU with limited VRAM — configs with very high DEPTH, large HEAD_DIM,
or large DEVICE_BATCH_SIZE may cause OOM (out of memory) crashes.

Current search space with bounds:
{space_description}

Previous evaluations (config → result):
{history}

{incumbent_info}

Suggest the next configuration to try. Learn from both successful runs AND failures (OOM crashes
mean the model was too large for the GPU). Respond with a JSON object mapping HP names to values.
Only include HPs from the search space. Be creative but principled — consider what you know
about transformer training dynamics.

Respond ONLY with a valid JSON object, no explanation."""


class LLMGreedyBackend(HPOBackend):
    """LLM-based greedy hill-climbing, mimicking Karpathy's approach.

    Uses an LLM to suggest configs based on the full evaluation history.
    This is the baseline we compare AutoML backends against.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self._llm_client = llm_client
        self._model = model
        self._space: ConfigurationSpace | None = None
        self._max_budget: float = 300.0
        self._results: list[tuple[dict, float, dict]] = []
        self._objectives: list[str] = []

    @property
    def name(self) -> str:
        return "llm_greedy"

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

        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI()

    def suggest(self) -> tuple[dict[str, Any], float]:
        space_desc = self._format_space()
        history = self._format_history()
        incumbent_info = ""
        if self._results:
            best = min(self._results, key=lambda x: x[2].get(self._objectives[0], float("inf")))
            incumbent_info = f"Current best config (val_bpb={best[2].get(self._objectives[0], '?')}):\n{json.dumps(best[0], indent=2)}"

        prompt = SUGGEST_PROMPT.format(
            space_description=space_desc,
            history=history if history else "(no evaluations yet)",
            incumbent_info=incumbent_info,
        )

        try:
            response = self._llm_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            config = json.loads(text)

            # Validate against space bounds
            config = self._clamp_to_space(config)
            return config, self._max_budget

        except Exception as e:
            logger.warning("LLM suggestion failed (%s), falling back to random", e)
            sample = self._space.sample_configuration()
            return self._to_native_types(dict(sample)), self._max_budget

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        self._results.append((config, budget, dict(results)))

    def incumbents(self) -> list[dict[str, Any]]:
        if not self._results:
            return []
        obj = self._objectives[0]
        best = min(self._results, key=lambda x: x[2].get(obj, float("inf")))
        return [best[0]]

    def _format_space(self) -> str:
        lines = []
        for hp in self._space.values():
            lines.append(f"  {hp.name}: {hp}")
        return "\n".join(lines)

    def _format_history(self) -> str:
        if not self._results:
            return ""
        lines = []
        for config, _budget, results in self._results[-20:]:  # last 20 to fit context
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

    @staticmethod
    def _to_native_types(config: dict) -> dict:
        """Convert numpy types to native Python for JSON compatibility."""
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

    def _clamp_to_space(self, config: dict) -> dict:
        """Ensure LLM-suggested values fall within ConfigSpace bounds."""
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
