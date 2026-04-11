"""Karpathy Agent (Code) backend that uses Claude Code subscription via the
claude-agent-sdk, so LLM calls bill against the Max subscription quota instead
of a raw API key.

Same prompt, same hill-climbing, same output parsing as KarpathyAgentBackend.
Only the LLM call itself changes. Thinking is disabled so the comparison with
Qwen3.5-27B and Gemini 3.1 Pro Preview (also with thinking disabled) is fair.

Requires:
    uv pip install claude-agent-sdk

Auth:
    Run `claude auth login` once so the SDK picks up the subscription OAuth
    token. Calls will then bill against the Claude Max 20x quota.
"""

from __future__ import annotations

import asyncio
import ast
import logging
import os
import time
from pathlib import Path
from typing import Any

from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.karpathy_agent_backend import (
    AGENT_PROMPT,
    KarpathyAgentBackend,
)

logger = logging.getLogger(__name__)


class KarpathyAgentClaudeCodeBackend(KarpathyAgentBackend):
    """KarpathyAgentBackend variant that calls Claude Code SDK (subscription auth).

    Uses the exact same AGENT_PROMPT, history formatting, hill-climbing, and
    response parser as the parent. Only the LLM call path changes.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        log_dir: Path | None = None,
        max_turns: int = 1,
        permission_mode: str = "bypassPermissions",
    ):
        # Parent __init__ accepts an OpenAI-compatible llm_client that we never use.
        super().__init__(llm_client=None, model=model, log_dir=log_dir)
        self._max_turns = max_turns
        self._permission_mode = permission_mode
        # Lazy-import check is deferred to configure() so the module can load
        # without the SDK being installed (e.g., for CLI registry discovery).
        self._sdk_ready = False

    @property
    def name(self) -> str:
        return "karpathy_agent_claude_code"

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
        self._objectives = objectives
        self._max_budget = budget_range[1] if budget_range else 300.0
        self._seed = seed

        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "claude-agent-sdk is required for KarpathyAgentClaudeCodeBackend. "
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
            max_turns=self._max_turns,
            permission_mode=self._permission_mode,
            allowed_tools=[],  # pure completion; no tool use
        )

        text_parts: list[str] = []
        async for message in query(prompt=prompt, options=options):
            # AssistantMessage carries a list of content blocks; we only keep
            # TextBlock content. ThinkingBlocks (if present) are dropped so
            # parsing is consistent with the Qwen / Gemini baselines.
            content = getattr(message, "content", None)
            if content is None:
                continue
            if isinstance(content, list):
                for block in content:
                    btype = getattr(block, "type", None) or type(block).__name__.lower()
                    if "thinking" in btype:
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
            # If we're already inside an event loop (rare in our sync runner),
            # fall back to a dedicated loop.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._call_claude_async(prompt))
            finally:
                loop.close()

    # ------------------------------------------------------------------
    # Override suggest() to swap the LLM call
    # ------------------------------------------------------------------
    def suggest(self) -> tuple[dict[str, Any], float]:
        if self._current_best_source is None:
            raise RuntimeError("Must call set_source() before suggest()")

        available_vram = os.environ.get("AVAILABLE_VRAM", "~120GB")
        prompt = AGENT_PROMPT.format(
            current_source=self._current_best_source,
            history=self._format_history() or "(first experiment — this is the baseline run)",
            best_val_bpb=(
                f"{self._best_val_bpb:.4f}"
                if self._best_val_bpb < float("inf")
                else "unknown (baseline)"
            ),
            available_vram=available_vram,
        )

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
                "messages": [{"role": "user", "content": "(prompt omitted — see source)"}],
                "response_length": len(text),
                "thinking_length": 0,
                "thinking": "",
            }

            description, source = self._parse_response(text)

            try:
                ast.parse(source)
            except SyntaxError as e:
                logger.warning("Claude generated invalid Python: %s. Using current best.", e)
                source = self._current_best_source
                description = f"SYNTAX ERROR (reverted): {description}"

            self._pending_source = source
            self._pending_description = description
            config = self._extract_config(source)
            return config, self._max_budget

        except Exception as e:
            logger.warning("Claude Code backend LLM call failed: %s. Using current best source.", e)
            self._pending_source = self._current_best_source
            self._pending_description = f"LLM FAILED: {e}"
            config = self._extract_config(self._current_best_source)
            return config, self._max_budget
