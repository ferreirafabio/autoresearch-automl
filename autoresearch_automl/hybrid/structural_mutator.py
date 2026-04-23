"""LLM proposes code-level architectural changes between HPO phases."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MUTATION_PROMPT = """\
You are an expert ML researcher optimizing a GPT-2 scale transformer.
The current architecture is defined in train.py (below).

Current best val_bpb: {best_val_bpb}
Optimization history summary: {n_trials} trials across {n_generations} structural generations.
{generation_history}

Propose ONE structural change to the architecture that could improve val_bpb.
Examples of structural changes:
- Replace attention mechanism (e.g., standard → flash attention, multi-head → GQA)
- Change activation function (e.g., GELU → SwiGLU)
- Modify normalization (e.g., LayerNorm → RMSNorm)
- Add/remove architectural components (e.g., rotary embeddings, MoE layers)
- Change optimizer (e.g., AdamW → Muon, add Nesterov momentum)

train.py:
```python
{train_py_source}
```

Respond with:
1. A brief description of the change (1-2 sentences)
2. The modified train.py source code

Format your response as:
DESCRIPTION: <your description>
```python
<full modified train.py>
```"""


@dataclass
class StructuralMutation:
    """A proposed structural change to train.py."""
    description: str
    new_source: str
    generation: int


class StructuralMutator:
    """Uses LLM to propose code-level architectural changes."""

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self._llm_client = llm_client
        self._model = model

    def _get_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI()
        return self._llm_client

    def propose_mutation(
        self,
        train_py_path: Path,
        best_val_bpb: float,
        n_trials: int,
        generation: int,
        generation_history: str = "",
    ) -> StructuralMutation:
        """Propose a structural mutation to train.py."""
        source = train_py_path.read_text()

        prompt = MUTATION_PROMPT.format(
            best_val_bpb=f"{best_val_bpb:.4f}" if best_val_bpb < float("inf") else "N/A",
            n_trials=n_trials,
            n_generations=generation,
            generation_history=generation_history or "(first generation)",
            train_py_source=source[:12000],  # truncate for context
        )

        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192,
        )
        text = response.choices[0].message.content.strip()

        # Parse description and code
        description = "Unknown structural change"
        new_source = source  # fallback

        if "DESCRIPTION:" in text:
            desc_part = text.split("DESCRIPTION:")[1]
            if "```" in desc_part:
                description = desc_part.split("```")[0].strip()
            else:
                description = desc_part.split("\n")[0].strip()

        if "```python" in text:
            code_part = text.split("```python")[1]
            if "```" in code_part:
                new_source = code_part.split("```")[0].strip()
            else:
                new_source = code_part.strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                new_source = parts[1].strip()
                if new_source.startswith("python\n"):
                    new_source = new_source[7:]

        return StructuralMutation(
            description=description,
            new_source=new_source,
            generation=generation + 1,
        )

    def apply_mutation(self, train_py_path: Path, mutation: StructuralMutation) -> None:
        """Apply a structural mutation to train.py."""
        train_py_path.write_text(mutation.new_source)
        logger.info(
            "Applied structural mutation (gen %d): %s",
            mutation.generation, mutation.description,
        )
