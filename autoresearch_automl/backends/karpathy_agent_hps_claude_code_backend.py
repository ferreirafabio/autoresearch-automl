"""KarpathyAgent (14 HPs, fixed search space) backend that uses Claude Code
subscription via the claude-agent-sdk.

Same SUGGEST_PROMPT, history formatting, space clamping, and JSON extraction as
KarpathyAgentHPSBackend. Only the LLM call path changes.

Apples-to-apples with Qwen3.5-27B and Gemini 3.1 Pro Preview variants:
  - Identical prompt (inherited SUGGEST_PROMPT)
  - No system prompt (system_prompt="" replaces Claude Code's default preset)
  - Extended thinking hard-disabled server-side and client-side
  - Same VRAM cap (CUDA_MEM_FRACTION=0.543 -> 76 GB) set by the sbatch wrapper

Requires:
    uv pip install claude-agent-sdk

Auth:
    Run `claude auth login` once so the SDK picks up the subscription OAuth
    token. Calls bill against the Claude Max quota rather than a raw API key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.karpathy_agent_hps_backend import (
    KarpathyAgentHPSBackend,
    SUGGEST_PROMPT,
)

logger = logging.getLogger(__name__)


class KarpathyAgentHPSClaudeCodeBackend(KarpathyAgentHPSBackend):
    """KarpathyAgentHPSBackend variant that calls Claude Code SDK (subscription auth).

    Uses the exact same SUGGEST_PROMPT, history formatting, and clamp-to-space
    logic as the parent. Only the LLM call path changes.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        log_dir: Path | None = None,
    ):
        # Parent accepts llm_client that we never use.
        super().__init__(llm_client=None, model=model, log_dir=log_dir)
        self._sdk_ready = False

    @property
    def name(self) -> str:
        return "karpathy_agent_hps_claude_code"

    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        # Intentionally do NOT call super().configure(): the parent tries to
        # instantiate an OpenAI client here, which we don't need or want.
        self._space = space
        self._objectives = objectives
        self._max_budget = budget_range[1] if budget_range else 300.0
        self._seed = seed

        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "claude-agent-sdk is required for KarpathyAgentHPSClaudeCodeBackend. "
                "Install with `uv pip install claude-agent-sdk`, then run "
                "`claude auth login` once to authenticate with the subscription."
            ) from e
        self._sdk_ready = True

    # ------------------------------------------------------------------
    # LLM call via Claude Agent SDK (subscription auth)
    # ------------------------------------------------------------------
    async def _call_claude_async(self, prompt: str) -> str:
        """Single-turn Claude Code SDK call. Returns assistant text content only."""
        from claude_agent_sdk import query, ClaudeAgentOptions  # type: ignore

        options = ClaudeAgentOptions(
            model=self._model,
            max_turns=1,
            permission_mode="bypassPermissions",
            tools=[],                            # load NO tools
            system_prompt="",                    # disable Claude Code default preset
            thinking={"type": "disabled"},       # server-side thinking off
            max_thinking_tokens=0,               # defensive
        )

        text_parts: list[str] = []
        async for message in query(prompt=prompt, options=options):
            content = getattr(message, "content", None)
            if content is None:
                continue
            if isinstance(content, list):
                for block in content:
                    btype = getattr(block, "type", None) or type(block).__name__.lower()
                    if "thinking" in str(btype).lower():
                        continue
                    text = getattr(block, "text", None)
                    if text:
                        text_parts.append(text)
            elif isinstance(content, str):
                text_parts.append(content)
        return "".join(text_parts).strip()

    def _call_claude(self, prompt: str) -> str:
        """Sync wrapper around the async SDK call."""
        try:
            return asyncio.run(self._call_claude_async(prompt))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._call_claude_async(prompt))
            finally:
                loop.close()

    # ------------------------------------------------------------------
    # Override suggest() to swap the LLM call
    # ------------------------------------------------------------------
    def suggest(self) -> tuple[dict[str, Any], float]:
        space_desc = self._format_space()
        history = self._format_history()
        incumbent_info = ""
        if self._results:
            best = min(self._results, key=lambda x: x[2].get(self._objectives[0], float("inf")))
            incumbent_info = f"Current best config (val_bpb={best[2].get(self._objectives[0], '?')}):\n{json.dumps(best[0], indent=2)}"

        available_vram = os.environ.get("AVAILABLE_VRAM", "~120GB")
        prompt = SUGGEST_PROMPT.format(
            space_description=space_desc,
            history=history if history else "(no evaluations yet)",
            incumbent_info=incumbent_info,
            available_vram=available_vram,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            t0 = time.time()
            text = self._call_claude(prompt)
            elapsed = time.time() - t0

            self._pending_llm_call = {
                "timestamp": t0,
                "elapsed_s": round(elapsed, 3),
                "pid": os.getpid(),
                "model": self._model,
                "trial_id": self._trial_id,
                "messages": messages,
                "response": text,
                "thinking": "",
            }

            # Extract JSON from response (handle markdown code blocks)
            clean = text
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            config = json.loads(clean)

            # Validate against space bounds
            config = self._clamp_to_space(config)
            return config, self._max_budget

        except Exception as e:
            logger.warning("Claude Code HPS backend LLM call failed (%s), falling back to random", e)
            sample = self._space.sample_configuration()
            return self._to_native_types(dict(sample)), self._max_budget
