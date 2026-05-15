---
name: Claude Agent SDK gotchas for apples-to-apples LLM-as-optimizer comparison
description: Critical SDK options required when using claude-agent-sdk as a drop-in replacement for OpenAI-compatible LLM calls — learned from integrating Opus 4.6 into the autoresearch benchmark
type: feedback
---

When plugging `claude-agent-sdk` into a benchmark that expects the LLM to behave as a pure text-completion endpoint (no tools, no thinking, no system prompt), four `ClaudeAgentOptions` fields MUST be set explicitly. The defaults will silently break a fair comparison.

**Why:** The SDK is designed for agent workflows by default, which means it injects a system prompt, enables tools, and may use extended thinking. A benchmark comparison against Qwen/Gemini (both via OpenAI-compatible APIs with only a user message) needs none of that.

**How to apply:**

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-opus-4-6",
    max_turns=1,                         # single-shot, no multi-turn loop
    permission_mode="bypassPermissions",
    tools=[],                            # critical: see below
    system_prompt="",                    # critical: see below
    thinking={"type": "disabled"},       # critical: see below
    max_thinking_tokens=0,               # belt-and-suspenders
)
```

## Four gotchas, ranked by how easy they are to miss

### 1. `tools=[]` vs `allowed_tools=[]` — NOT the same

- `allowed_tools=[]` is a **filter**, not a disable switch. With an empty allow list the SDK still loads the default tool set, and Opus can still invoke `Task` (the agent-spawning tool). This causes multi-turn interleaving (`TaskStartedMessage`, `TaskProgressMessage`, ...) and will time out on long prompts.
- `tools=[]` fully disables tool loading so the model runs as pure completion.

Symptom if wrong: `Fatal error in message reader: Command failed with exit code 1` on long prompts, zero progress, subprocess hang.

### 2. `system_prompt=""` replaces the default preset; `None` uses it

- `system_prompt=None` (the default) injects Claude Code's full preset system prompt: tool instructions, working directory context, git status, auto-memory, CLI commands, permission rules. None of this is present in a vanilla OpenAI `messages.create` call.
- `system_prompt=""` replaces the preset with an empty string, giving the model only the user message — exactly matching what Qwen/Gemini received via their OpenAI-compatible endpoints.

Symptom if wrong: model gets more context than baseline methods, breaks "apples to apples" fairness, and may respond with tool-using behavior.

### 3. `thinking={"type":"disabled"}` — explicit disable required for Opus

- Opus models enable extended thinking by default at the server side. Even if the client-side code filters out `ThinkingBlock` from the response, tokens are still consumed and the model's reasoning path differs from a thinking-off run.
- Use `ThinkingConfigDisabled` (a `TypedDict` = `{"type": "disabled"}`) to hard-disable server-side.

### 4. `max_thinking_tokens=0` as a belt-and-suspenders

Separate knob that limits thinking budget regardless of the `thinking` config. Safe to set alongside `thinking={"type":"disabled"}`.

## Verification checklist

After configuring, run a minimal test and confirm:
- Message stream contains only `SystemMessage`, `AssistantMessage`, `RateLimitEvent`, `ResultMessage`.
- **No** `TaskStartedMessage`, `TaskProgressMessage`, `UserMessage` (those are tool-driven sub-turns).
- Assistant content blocks are `TextBlock` only, no `ThinkingBlock`.
- Response arrives in a single `AssistantMessage`, not split across many turns.
- Total elapsed time is on the order of the prompt length, not indefinite.

## Subprocess / auth notes

- The SDK spawns the bundled `claude` CLI as a subprocess; use `stderr=callable` on `ClaudeAgentOptions` to capture subprocess stderr during debugging.
- Claude Code auth is OAuth-based, not API-key. The SDK reads the token from `~/.claude/credentials.json`, so `claude auth login` must be run once on a host that shares `$HOME` with the execution environment (on NFS-mounted clusters this is automatic across nodes).
- Subscription billing (e.g., Claude Max 20x) applies automatically; there is no separate API key path.

## Project context where this was learned

This was discovered during integration of Claude Opus 4.6 into the autoresearch-automl benchmark as `karpathy_agent_claude_code` and `centaur_claude_code` backends. Without these options, the first attempt produced `Task`-spawning interleaved messages and the subprocess died on long `AGENT_PROMPT` inputs (~27 kB). With the correct options, the SDK returns a clean 26k-char code block in ~135 s on Opus 4.6, matching the behavior of the Qwen / Gemini OpenAI-compatible calls.
