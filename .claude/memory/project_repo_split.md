---
name: Public/private repo split
description: Three repos (private, public, paper), cluster name restrictions, push workflow
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## Repos
- **Private**: https://github.com/ferreirafabio/autoresearch-automl-private (remote: `origin`)
- **Public**: https://github.com/ferreirafabio/autoresearch-automl (remote: `public`)
- **Paper**: https://github.com/ferreirafabio/autoresearch-automl-paper (Overleaf-synced)

## Public repo rules
NEVER push files or strings containing:
- `kislurm`, `horeka`, `HoreKa`, `hkfs`, `scc.kit.edu`, `aadlogin`
- `dlclarge`, `dlclarge1`, `dlclarge2`, `fr_ff1042`
- `/work/dlclarge*/`, `/hkfs/work/`
- SLURM account IDs, internal job IDs, node/partition names
- slurm/ directory, RESULTS.md, experiments/EXPERIMENTS.md

## Push workflow
- `git push origin main` = private only
- `git push public main` = public (may need `--force` after history rewrites)
- **The GitHub Pages demo (https://ferreirafabio.github.io/autoresearch-automl/) deploys ONLY from the `public` repo.** Demo edits won't appear after `git push origin` — must also `git push public main`. Easy to forget.
- Always push to both for non-cluster-specific changes
- Paper repo: `git pull` first (Overleaf may have pushed), then `git push`
- Always compile LaTeX locally before pushing paper repo
- Include main.bbl for Overleaf reference resolution

## Name mapping (original results -> repo results)
- `llm_greedy` -> `karpathy_agent_hps`
- `llm_greedy_Qwen3_5_27B_nothink` -> `karpathy_agent_hps_Qwen3_5_27B`
- `llambo_Qwen3_5_27B_nothink` -> `llambo_Qwen3_5_27B`
- `llambo_original_Qwen3_5_27B_nothink` -> `llambo_original_Qwen3_5_27B`
