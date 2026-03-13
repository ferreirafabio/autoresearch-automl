"""LLM generates initial configs to seed HPO (warm-start oracle)."""

from __future__ import annotations

import json
import logging
from typing import Any

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

logger = logging.getLogger(__name__)

WARMSTART_PROMPT = """\
You are an expert ML researcher. Suggest {n_configs} diverse initial configurations
for hyperparameter optimization of a GPT-2 scale transformer trained on climbmix-400b-shuffle.

Search space:
{space_description}

Goal: minimize val_bpb (validation bits-per-byte).
Training budget: {budget}s per config on H200 GPU.

Suggest {n_configs} diverse configs that cover different strategies:
1. A conservative baseline (known-good defaults)
2. An aggressive high-capacity config
3-{n_configs}. Creative variations exploring different trade-offs

Respond with a JSON array of {n_configs} objects, each mapping HP names to values.
Only use HP names from the search space above. Respond ONLY with valid JSON."""


class WarmStartOracle:
    """Uses LLM to generate smart initial configs for HPO warm-starting."""

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self._llm_client = llm_client
        self._model = model

    def _get_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI()
        return self._llm_client

    def generate_initial_configs(
        self,
        space: ConfigurationSpace,
        n_configs: int = 5,
        budget: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Generate n_configs initial configurations using LLM domain knowledge."""
        space_desc = self._format_space(space)

        prompt = WARMSTART_PROMPT.format(
            n_configs=n_configs,
            space_description=space_desc,
            budget=budget,
        )

        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        configs = json.loads(text)

        # Validate and clamp to space bounds
        validated = []
        for cfg in configs:
            clamped = self._clamp_config(cfg, space)
            if clamped:
                validated.append(clamped)

        logger.info("Generated %d warm-start configs (requested %d)", len(validated), n_configs)
        return validated

    @staticmethod
    def _format_space(space: ConfigurationSpace) -> str:
        lines = []
        for hp in space.values():
            lines.append(f"  {hp.name}: {hp}")
        return "\n".join(lines)

    @staticmethod
    def _clamp_config(config: dict, space: ConfigurationSpace) -> dict[str, Any] | None:
        """Validate and clamp config values to space bounds."""
        clamped = {}
        for hp in space.values():
            if hp.name not in config:
                clamped[hp.name] = hp.default_value
                continue
            val = config[hp.name]
            if isinstance(hp, UniformIntegerHyperparameter):
                val = int(round(val))
                val = max(hp.lower, min(hp.upper, val))
            elif isinstance(hp, UniformFloatHyperparameter):
                val = float(val)
                val = max(hp.lower, min(hp.upper, val))
            clamped[hp.name] = val
        return clamped
