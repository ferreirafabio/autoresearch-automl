"""LLM proposes/refines ConfigurationSpace from train.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

logger = logging.getLogger(__name__)

PROPOSE_SPACE_PROMPT = """\
You are an expert ML researcher. Analyze the following training script and propose
a ConfigurationSpace for hyperparameter optimization.

For each hyperparameter found as ALL_CAPS = value assignments, specify:
- name: the variable name
- type: "int", "float", or "cat"
- For int/float: low, high, log (boolean)
- For cat: choices (list of values)

Focus on hyperparameters that most affect val_bpb (validation bits-per-byte).
Ignore path/device/logging variables.

train.py:
```python
{train_py_source}
```

{history_context}

Respond with a JSON array of objects, each with keys: name, type, low, high, log, choices.
Only include keys relevant to the type. Respond ONLY with valid JSON, no explanation."""

REFINE_SPACE_PROMPT = """\
You are refining a hyperparameter search space based on optimization results.

Current search space:
{current_space}

Results from {n_trials} trials:
Best val_bpb: {best_val_bpb}
Top 5 configs:
{top_configs}

HP importance (estimated):
{hp_importance}

Suggest refinements to the search space:
1. Zoom in on promising regions
2. Drop unimportant HPs
3. Add new HPs that might help
4. Adjust bounds based on where good configs cluster

Respond with the same JSON array format as before."""


class SpaceProposer:
    """Uses an LLM to propose and refine ConfigurationSpaces from train.py."""

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self._llm_client = llm_client
        self._model = model

    def _get_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI()
        return self._llm_client

    def propose_space(
        self,
        train_py_path: Path,
        history: list[dict] | None = None,
    ) -> ConfigurationSpace:
        """LLM reads train.py and proposes a ConfigurationSpace."""
        source = train_py_path.read_text()
        history_context = ""
        if history:
            history_context = f"Previous optimization history ({len(history)} trials available)."

        prompt = PROPOSE_SPACE_PROMPT.format(
            train_py_source=source[:8000],  # truncate if very long
            history_context=history_context,
        )

        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        hp_specs = json.loads(text)
        return self._specs_to_configspace(hp_specs)

    def refine_space(
        self,
        current_space: ConfigurationSpace,
        results: list[tuple[dict, dict]],
        hp_importance: dict[str, float] | None = None,
    ) -> ConfigurationSpace:
        """LLM reviews results and refines the search space."""
        # Sort by val_bpb
        sorted_results = sorted(results, key=lambda x: x[1].get("val_bpb", float("inf")))
        top_5 = sorted_results[:5]

        top_configs_str = "\n".join(
            f"  {json.dumps(cfg)} → val_bpb={res.get('val_bpb', '?')}"
            for cfg, res in top_5
        )
        importance_str = json.dumps(hp_importance or {}, indent=2)

        prompt = REFINE_SPACE_PROMPT.format(
            current_space=str(current_space),
            n_trials=len(results),
            best_val_bpb=top_5[0][1].get("val_bpb", "?") if top_5 else "?",
            top_configs=top_configs_str,
            hp_importance=importance_str,
        )

        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        hp_specs = json.loads(text)
        return self._specs_to_configspace(hp_specs)

    @staticmethod
    def _specs_to_configspace(specs: list[dict]) -> ConfigurationSpace:
        """Convert LLM-generated HP specs to ConfigurationSpace."""
        cs = ConfigurationSpace(seed=42)
        for spec in specs:
            name = spec["name"]
            hp_type = spec.get("type", "float")
            try:
                if hp_type == "cat":
                    cs.add(Categorical(name, items=spec["choices"]))
                elif hp_type == "int":
                    cs.add(Integer(
                        name,
                        bounds=(spec["low"], spec["high"]),
                        log=spec.get("log", False),
                    ))
                elif hp_type == "float":
                    cs.add(Float(
                        name,
                        bounds=(spec["low"], spec["high"]),
                        log=spec.get("log", False),
                    ))
            except Exception as e:
                logger.warning("Failed to add HP %s: %s", name, e)
        return cs
