"""Karpathy agent backend — LLM directly edits train.py code (no fixed search space)."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.base import HPOBackend

logger = logging.getLogger(__name__)

AGENT_PROMPT = """\
You are an autonomous ML researcher. Your goal: minimize val_bpb (validation bits-per-byte) \
for a GPT-2 scale transformer trained on climbmix-400b-shuffle.

Current train.py (the version that achieved the best result so far):
```python
{current_source}
```

Previous experiments:
{history}

Current best val_bpb: {best_val_bpb}

Rules:
- You can change ANYTHING in train.py: architecture, optimizer, hyperparameters, training loop, model size.
- Do NOT modify imports from prepare (prepare.py is read-only).
- The script runs for exactly 5 minutes on a single GPU with {available_vram} VRAM available.
- Configs that use too much memory will crash (OOM) — be mindful of DEPTH, HEAD_DIM, DEVICE_BATCH_SIZE.
- Simplicity wins: a small improvement from deleting code is better than a large improvement from adding complexity.
- Think about what previous experiments tell you — learn from both successes and failures.

Respond in EXACTLY this format (no extra text before or after):

DESCRIPTION: <one line explaining what you changed and why>
```python
<the complete modified train.py — must be valid, runnable Python>
```"""


class KarpathyAgentBackend(HPOBackend):
    """Karpathy's original autoresearch approach: LLM directly edits train.py.

    Unlike HP-based backends that pick values from a fixed search space,
    this backend shows the LLM the full train.py source and asks it to
    propose code modifications — architecture, optimizer, hyperparameters,
    anything. Uses hill-climbing: keep improvements, revert failures.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini", log_dir: Path | None = None):
        self._llm_client = llm_client
        self._model = model
        self._log_dir = log_dir
        self._seed: int = 0
        self._trial_id = 0
        self._objectives: list[str] = []
        self._max_budget: float = 300.0

        # Source code state (hill-climbing)
        self._original_source: str | None = None
        self._current_best_source: str | None = None
        self._best_val_bpb: float = float("inf")

        # History for prompt
        self._results: list[dict[str, Any]] = []

        # Pending suggestion
        self._pending_source: str | None = None
        self._pending_description: str = ""
        self._pending_llm_call: dict | None = None
        self._replaying = False

    @property
    def name(self) -> str:
        return "karpathy_agent"

    def set_source(self, source: str) -> None:
        """Set the original train.py source code. Called by runner before suggest()."""
        self._original_source = source
        if self._current_best_source is None:
            self._current_best_source = source

    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self._objectives = objectives
        self._max_budget = budget_range[1] if budget_range else 300.0
        self._seed = seed

        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI()

    def suggest(self) -> tuple[dict[str, Any], float]:
        """Suggest next config by generating modified train.py source.

        Returns a config dict (extracted HP values for logging) and budget.
        The actual source code is available via get_source_override().
        """
        if self._current_best_source is None:
            raise RuntimeError("Must call set_source() before suggest()")

        available_vram = os.environ.get("AVAILABLE_VRAM", "~120GB")
        prompt = AGENT_PROMPT.format(
            current_source=self._current_best_source,
            history=self._format_history() or "(first experiment — this is the baseline run)",
            best_val_bpb=f"{self._best_val_bpb:.4f}" if self._best_val_bpb < float("inf") else "unknown (baseline)",
            available_vram=available_vram,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            t0 = time.time()
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
                "trial_id": self._trial_id,
                "messages": [{"role": "user", "content": "(prompt omitted — see source)"}],
                "response_length": len(text),
                "thinking_length": len(thinking),
                "thinking": thinking,
            }

            description, source = self._parse_response(text)

            # Validate: must be parseable Python
            try:
                ast.parse(source)
            except SyntaxError as e:
                logger.warning("LLM generated invalid Python: %s. Using current best.", e)
                source = self._current_best_source
                description = f"SYNTAX ERROR (reverted): {description}"

            self._pending_source = source
            self._pending_description = description

            config = self._extract_config(source)
            return config, self._max_budget

        except Exception as e:
            logger.warning("Karpathy agent LLM call failed: %s. Using current best source.", e)
            self._pending_source = self._current_best_source
            self._pending_description = f"LLM FAILED: {e}"
            config = self._extract_config(self._current_best_source)
            return config, self._max_budget

    def get_source_override(self) -> str | None:
        """Return the pending source code for the runner to write directly."""
        return self._pending_source

    def seed_trial(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        """Inject baseline result."""
        val_bpb = results.get(self._objectives[0]) if self._objectives else None
        if val_bpb is not None and val_bpb < self._best_val_bpb:
            self._best_val_bpb = val_bpb
        self._results.append({
            "description": "baseline (Karpathy defaults)",
            "val_bpb": val_bpb,
            "success": val_bpb is not None,
            "error": results.get("_error"),
            "kept": True,
        })
        self._trial_id += 1
        logger.info("Seeded baseline into Karpathy agent: val_bpb=%s", val_bpb)

    def replay(self, history: list[tuple[dict, float, dict]]) -> None:
        """Replay completed trials to reconstruct history."""
        self._replaying = True
        for config, budget, results in history:
            self.tell(config, budget, results)
        self._replaying = False
        # Load best source from checkpoint file if available
        if self._log_dir:
            best_source_path = self._log_dir / "karpathy_best_source.py"
            if best_source_path.exists():
                self._current_best_source = best_source_path.read_text()
                logger.info("Restored best source from %s", best_source_path)
        logger.info("Replayed %d trials into Karpathy agent", len(history))

    def tell(self, config: dict[str, Any], budget: float, results: dict[str, float]) -> None:
        val_bpb = results.get(self._objectives[0]) if self._objectives else None
        error = results.get("_error")
        success = val_bpb is not None and error is None

        # Hill-climbing: keep if improved
        kept = False
        if success and val_bpb is not None and val_bpb < self._best_val_bpb:
            old_best = self._best_val_bpb
            self._best_val_bpb = val_bpb
            if self._pending_source and not self._replaying:
                self._current_best_source = self._pending_source
                # Save best source for resume
                if self._log_dir:
                    best_path = self._log_dir / "karpathy_best_source.py"
                    best_path.write_text(self._current_best_source)
            kept = True
            logger.info(
                "Karpathy agent: KEEP (%.4f → %.4f) — %s",
                old_best, val_bpb, self._pending_description,
            )
        else:
            logger.info(
                "Karpathy agent: REVERT (val_bpb=%s, best=%.4f) — %s",
                val_bpb, self._best_val_bpb, self._pending_description,
            )

        self._results.append({
            "description": self._pending_description or "(replayed)",
            "val_bpb": val_bpb,
            "success": success,
            "error": error,
            "kept": kept,
        })

        if not self._replaying:
            self._flush_llm_log(results)

        self._pending_source = None
        self._pending_description = ""
        self._trial_id += 1

    def incumbents(self) -> list[dict[str, Any]]:
        if self._current_best_source:
            return [self._extract_config(self._current_best_source)]
        return []

    def _format_history(self) -> str:
        if not self._results:
            return ""
        lines = []
        for r in self._results[-15:]:
            status_tag = "KEPT" if r["kept"] else "REVERTED"
            if r["success"]:
                lines.append(f"  [{status_tag}] {r['description']} → val_bpb={r['val_bpb']:.4f}")
            else:
                lines.append(f"  [REVERTED] {r['description']} → CRASHED ({r['error']})")
        return "\n".join(lines)

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Parse LLM response into (description, source_code)."""
        # Extract description
        description = "unknown change"
        desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", text)
        if desc_match:
            description = desc_match.group(1).strip()

        # Extract Python code block
        code_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if code_match:
            source = code_match.group(1).strip()
            return description, source

        # Fallback: try to find any code block
        code_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if code_match:
            source = code_match.group(1).strip()
            return description, source

        # Last resort: if the response looks like Python code, use it all
        if "import " in text and "def " in text:
            logger.warning("No code block found, using full response as source")
            # Strip the DESCRIPTION line if present
            source = re.sub(r"^DESCRIPTION:.*\n?", "", text).strip()
            return description, source

        logger.warning("Could not parse source code from LLM response")
        return description, self._current_best_source

    def _extract_config(self, source: str) -> dict[str, Any]:
        """Extract HP values from source code via AST for logging."""
        config = {}
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if not name.isupper():
                    continue
                try:
                    value = ast.literal_eval(node.value)
                    config[name] = value
                except (ValueError, TypeError):
                    pass
        except SyntaxError:
            pass
        return config

    def _flush_llm_log(self, results: dict[str, float]) -> None:
        """Write LLM call log to per-trial file."""
        if self._log_dir is None or self._pending_llm_call is None:
            return
        record = self._pending_llm_call
        record["description"] = self._pending_description
        record["val_bpb"] = results.get(self._objectives[0]) if self._objectives else None
        record["error"] = results.get("_error")
        record["kept"] = (
            record["val_bpb"] is not None
            and record["val_bpb"] < self._best_val_bpb
        )

        prefix = f"karpathy_agent_seed{self._seed}_trial{self._trial_id}"
        thinking = record.pop("thinking", "")

        with open(self._log_dir / f"{prefix}.jsonl", "w") as f:
            f.write(json.dumps(record, indent=2) + "\n")

        if thinking:
            with open(self._log_dir / f"{prefix}_thinking.txt", "w") as f:
                f.write(thinking)

        self._pending_llm_call = None
