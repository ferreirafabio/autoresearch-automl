---
name: Claude Code Opus 4.6 backend integration
description: Two new backends that run KA Code and Centaur via the Claude Code SDK (subscription auth); critical SDK gotchas, VRAM cap, auth setup, and sbatch scripts
type: project
---

# Claude Code Opus 4.6 integration (added 2026-04-11)

Two new backends call **Claude Opus 4.6** via the `claude-agent-sdk`, billed against the user's Max 20x subscription. The goal is to add `Karpathy Agent (Code) [Opus 4.6]` and `Centaur [Opus 4.6]` variants to the benchmark for direct comparison with Qwen3.5-27B and Gemini 3.1 Pro Preview.

## Files

- `autoresearch_automl/backends/karpathy_agent_claude_code_backend.py` — subclass of `KarpathyAgentBackend`, overrides only the LLM call path. Backend name: `karpathy_agent_claude_code`.
- `autoresearch_automl/backends/centaur_claude_code_backend.py` — subclass of `CentaurBackend`, overrides only `_suggest_llm()`. Backend name: `centaur_claude_code`.
- `autoresearch_automl/cli.py` — both backends registered in `BACKEND_REGISTRY` and allowlisted for the `--llm-model`, `--log-dir`, `--llm-ratio` kwargs.
- `slurm/exp2_karpathy_agent_claude_code.sh` — sbatch wrapper. Usage: `sbatch slurm/exp2_karpathy_agent_claude_code.sh <seed> [smoke]`
- `slurm/exp2_centaur_claude_code.sh` — sbatch wrapper. Same signature.

Both sbatch scripts skip vLLM entirely (no LLM server needed — SDK calls go out via the Claude Code subprocess to Anthropic). They set `CUDA_MEM_FRACTION=0.543` and `AVAILABLE_VRAM="76GB"` to match the Qwen/Gemini runs.

## Critical SDK options (must set, non-negotiable for fair comparison)

These options live in `ClaudeAgentOptions` inside each backend's `_call_claude_async`:

```python
options = ClaudeAgentOptions(
    model="claude-opus-4-6",
    max_turns=1,
    permission_mode="bypassPermissions",
    tools=[],                             # NOT allowed_tools=[]
    system_prompt="",                     # NOT None
    thinking={"type": "disabled"},
    max_thinking_tokens=0,
)
```

**Why each matters:**
- `tools=[]` — fully disables tool loading so Opus cannot spawn Task/agent subprocesses. `allowed_tools=[]` is WRONG; that is a filter that still permits default tools including Task, which causes multi-turn interleaving (`TaskStartedMessage`, `TaskProgressMessage`, etc.) and timeouts on long prompts.
- `system_prompt=""` — replaces Claude Code's default preset system prompt (tool instructions, working dir context, git status, auto-memory). `system_prompt=None` (the default) silently injects the preset, giving Opus MORE context than Qwen/Gemini received via their OpenAI-compatible APIs (which used only a user message, no system message).
- `thinking={"type":"disabled"}` + `max_thinking_tokens=0` — hard-disable extended thinking. Required for fairness with the Qwen (thinking off) and Gemini (thinking off) runs.

## Auth (one-time, already done)

```bash
uv pip install claude-agent-sdk     # installs into the project venv
claude auth login                   # OAuth via browser, on login node kis3bat2
claude auth status                  # verify Max subscription
```

$HOME is NFS-mounted across alldlc2_gpu-h200 compute nodes, so the OAuth token propagates. The sbatch script runs `claude auth status | head` as a sanity check at the start of each job.

## Apples-to-apples guarantees

Identical to Qwen3.5-27B and Gemini 3.1 Pro Preview variants:
- **Prompt**: inherited `AGENT_PROMPT` / `SUGGEST_PROMPT` from the parent backends, same format() placeholders
- **No system message**: enforced via `system_prompt=""`
- **No tool use**: enforced via `tools=[]`
- **No thinking**: enforced via `thinking={"type":"disabled"}` + `max_thinking_tokens=0`
- **VRAM cap**: 76 GB via `CUDA_MEM_FRACTION=0.543` (matches every run since 2026-03-17)
- **Budget**: 300s per trial, 24h total training time (`--budget-max 300 --time-budget 86400`)
- **Output parser**: inherited `_parse_response` (KA Code) and JSON-in-```` ``` ```` extraction (Centaur)

## Wall-time vs training time

LLM overhead per trial is ~2 min for KA Code (long train.py context + ~26k-char response) and ~30s for Centaur (smaller JSON output). This means:
- KA Code: 24h training budget ≈ 30-36h SLURM wall time
- Centaur: 24h training budget ≈ 26-30h SLURM wall time

Since the `alldlc2_gpu-h200` partition caps at 24h per job, we **chain two jobs** per seed via `--dependency=afterany:<job1>`. Both jobs use the same results dir and `--resume`, so the second one picks up where the first left off and exits naturally once the 24h training budget is hit.

## Known idiosyncrasies

- Claude CLI subprocess calls are noisy: expect several `SystemMessage`, `RateLimitEvent`, `ResultMessage` wrappers around the one `AssistantMessage` that carries the actual text. The backend filters these to keep only `TextBlock` content, dropping any `ThinkingBlock` defensively.
- Opus tends to be conversational (wraps the `DESCRIPTION:` / code block with extra framing text). The inherited `_parse_response` regex is tolerant enough to find the first ```` ```python ```` block regardless.
- SLURM `--requeue` on these backends is safe because `--resume` makes the runner replay `trials.jsonl` history from disk.

## Results directories

- KA Code: `/work/dlclarge1/ferreira-autoresearch-automl/results/gemini31pro_benchmark/karpathy_agent_claude_opus_4_6/seed_{0,1,2}/`
- Centaur: `/work/dlclarge1/ferreira-autoresearch-automl/results/gemini31pro_benchmark/centaur_claude_opus_4_6/seed_{0,1,2}/`

Co-located with the existing Gemini 3.1 Pro results in the same `gemini31pro_benchmark/` directory.
