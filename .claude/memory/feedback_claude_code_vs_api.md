---
name: Claude Code (subscription) vs Anthropic API — don't conflate
description: All *_claude_code backends consume Claude Code subscription quota (7d cap), NOT Anthropic API credits. The cron monitor watches Claude Code usage, not API spend.
type: feedback
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
Don't refer to "Claude credits" or "API credits" when talking about the Opus 4.6 / 4.7 runs or any other *_claude_code backend. Those backends use **Claude Code** (the SDK driven by the user's Claude Max subscription), not the paid Anthropic API.

**Why:** different billing model, different rate limit. Claude Code has a rolling 7-day usage cap (the one the cron monitor at `scripts/claude_usage_monitor.sh` watches via the OAuth `oauth/usage` endpoint). The Anthropic API would meter spend in dollars/tokens against the user's API key, which is not what's happening here.

**How to apply:** When jobs stall or get throttled, the cause is Claude Code 7-day quota, not API credits. Phrasing: "Claude Code weekly cap" or "Claude Code 7d quota". The monitor's THRESHOLD (currently 60%) is fraction of that weekly quota, not dollars.
