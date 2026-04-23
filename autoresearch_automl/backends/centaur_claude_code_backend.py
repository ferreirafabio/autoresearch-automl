"""Centaur (CMA-ES + LLM) backend that uses Claude Code subscription via the
claude-agent-sdk for the LLM side of the hybrid.

Same CMA-ES handling, same SUGGEST_PROMPT, same JSON extraction as
CentaurBackend. Only the LLM call path changes.

Apples-to-apples with Qwen3.5-27B and Gemini 3.1 Pro Preview Centaur variants:
  - Identical prompt (inherited SUGGEST_PROMPT)
  - No system prompt (system_prompt="" replaces Claude Code's default preset)
  - Extended thinking hard-disabled server-side and client-side
  - Same VRAM cap (CUDA_MEM_FRACTION=0.543 → 76 GB) set by the sbatch wrapper

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

from autoresearch_automl.backends.centaur_backend import (
    CentaurBackend,
    SUGGEST_PROMPT,
)

logger = logging.getLogger(__name__)


class CentaurClaudeCodeBackend(CentaurBackend):
    """CentaurBackend variant that calls Claude Code SDK (subscription auth).

    Uses the exact same SUGGEST_PROMPT, CMA state formatting, history, and
    clamp-to-space logic as the parent. Only the LLM call is swapped.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        log_dir: Path | None = None,
        llm_ratio: float = 0.3,
        llm_warmup: int = 10,
    ):
        super().__init__(
            model=model,
            log_dir=log_dir,
            llm_ratio=llm_ratio,
            llm_warmup=llm_warmup,
        )
        self._sdk_ready = False

    @property
    def name(self) -> str:
        return "centaur_claude_code"

    def configure(
        self,
        space: ConfigurationSpace,
        objectives: list[str],
        budget_range: tuple[float, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        # Parent configure() builds the CMA-ES Optuna study and wires up the
        # replay state. Do that first, then verify the SDK is available.
        super().configure(space, objectives, budget_range, seed, **kwargs)

        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "claude-agent-sdk is required for CentaurClaudeCodeBackend. "
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

        # See karpathy_agent_claude_code_backend.py for the rationale behind
        # each of these options. Short version: disable everything the SDK
        # adds by default so Claude sees ONLY our user message, identical to
        # what Qwen/Gemini received via their OpenAI-compatible endpoints.
        options = ClaudeAgentOptions(
            model=self._model,
            max_turns=1,
            permission_mode="bypassPermissions",
            tools=[],                            # load NO tools (not just an empty allow-filter)
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
    # Override only the LLM call method; everything else is inherited.
    # ------------------------------------------------------------------
    def _suggest_llm(self, cma_config: dict) -> dict[str, Any]:
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
        text = self._call_claude(prompt)
        elapsed = time.time() - t0

        self._pending_llm_call = {
            "timestamp": t0,
            "elapsed_s": round(elapsed, 3),
            "pid": os.getpid(),
            "model": self._model,
            "trial_id": self._trial_count,
            "source": "llm",
            "messages": messages,
            "response": text,
            "thinking": "",
        }

        # Extract JSON from response (same logic as parent)
        clean = text
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        config = json.loads(clean)

        return self._clamp_to_space(config)
