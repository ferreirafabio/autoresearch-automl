# Memory Index

## Project
- [project_overview.md](project_overview.md) — HPO into autoresearch: constraints, current backend list, dataset, experiment setup
- [project_repo_split.md](project_repo_split.md) — Three repos (private/public/paper), cluster restrictions, name mapping, push workflow
- [project_exp2_benchmark.md](project_exp2_benchmark.md) — Exp2 design: methods × seeds, 24h budget, slurm script arg orders
- [project_dir_naming.md](project_dir_naming.md) — Results dir → display-name map AND the 2026-05-15 de-aliasing (llm_greedy → karpathy_agent_hps, tpe → optuna)
- [project_27b_followup_jobs.md](project_27b_followup_jobs.md) — 27B methods need multiple chained 24h rounds to reach 24h training budget
- [project_h200_throttling.md](project_h200_throttling.md) — H200 power throttling resolved: healthy GPUs at ~1750K tok/s, baseline ~0.991
- [project_vram_cap_reruns.md](project_vram_cap_reruns.md) — 76GB VRAM cap: CUDA_MEM_FRACTION=0.543, all methods verified equal
- [project_thinking_disabled.md](project_thinking_disabled.md) — Thinking disabled for all Qwen3.5 models; max_tokens=2048 (8192 for karpathy_agent)
- [project_llambo_categoricals.md](project_llambo_categoricals.md) — LLAMBO Optuna delegates categoricals to random sampling, causing high OOM
- [project_llambo_patches.md](project_llambo_patches.md) — Three patches needed on cached OptunaHub LLAMBO files; must reapply if cache cleared
- [project_llm_logging.md](project_llm_logging.md) — Per-trial LLM logging with thinking traces
- [project_centaur_ablations.md](project_centaur_ablations.md) — Centaur ratio ablations (0.1, 0.2, 0.5, 0.8) for 0.8B and 27B
- [project_gemini_pro_paths.md](project_gemini_pro_paths.md) — Gemini 3.1 Pro Preview paths: split across gemini31pro_benchmark and gemini31_benchmark
- [project_claude_code_integration.md](project_claude_code_integration.md) — Opus 4.6 backends (karpathy_agent_claude_code, centaur_claude_code): files, SDK options, auth, sbatch wrappers
- [project_opus47_followup.md](project_opus47_followup.md) — Opus 4.7 follow-up: 9 jobs launched 2026-04-18 for a future paper; partial state preserved
- [project_kimi_k26_setup.md](project_kimi_k26_setup.md) — Kimi K2.6 self-host recipe: .venv-kimi, 3 required patches, single-node TP=8
- [project_live_benchmark_tab.md](project_live_benchmark_tab.md) — Demo's Live Benchmark tab architecture (build_tracker_hero.py, sections A/B/C/D, how to add a new Claude release)
- [project_claude_usage_monitor.md](project_claude_usage_monitor.md) — Cron-based Claude Code 7d-usage gate; auto-pauses *claude_code* slurm jobs across the weekly reset
- [project_fairness_audit.md](project_fairness_audit.md) — Fairness audit decisions (information leakage, default-seed handling)
- [project_observations_caveat.md](project_observations_caveat.md) — README "Key Observations" claims based on partial data; re-verify when runs complete

## Feedback / Rules
- [feedback_paper_writing_rules.md](feedback_paper_writing_rules.md) — Comprehensive paper writing rules: Frank's rules + user style + data integrity checklist
- [feedback_paper_style.md](feedback_paper_style.md) — Additional paper style preferences
- [feedback_search_space_framing.md](feedback_search_space_framing.md) — How to frame search space in paper text
- [feedback_no_em_dashes_anywhere.md](feedback_no_em_dashes_anywhere.md) — HARD RULE: never use em dashes (—) in ANY text for the user
- [feedback_claude_code_vs_api.md](feedback_claude_code_vs_api.md) — Don't conflate Claude Code (subscription, 7d cap) with Anthropic API (paid credits); *_claude_code backends are subscription-only
- [feedback_no_custom_hacks.md](feedback_no_custom_hacks.md) — Avoid custom hacks/workarounds; prefer principled fixes
- [feedback_claude_agent_sdk_gotchas.md](feedback_claude_agent_sdk_gotchas.md) — Critical ClaudeAgentOptions settings for pure-completion use: tools=[], system_prompt="", thinking disabled
- [feedback_24h_cap.md](feedback_24h_cap.md) — Hard 24h cap on all results, forward-fill interpolation, 95% budget filter, refresh checklist
- [feedback_no_coauthor.md](feedback_no_coauthor.md) — Never add Claude co-authorship to commits
- [feedback_git_push.md](feedback_git_push.md) — Allowed to git push without asking
- [feedback_check_paths.md](feedback_check_paths.md) — Never confuse Flash-Lite and Pro Preview paths, always verify exact dir names
- [feedback_plot_axes.md](feedback_plot_axes.md) — Primary plots: cumulative wall-time x-axis; secondary/appendix: trial number x-axis
- [feedback_status_format.md](feedback_status_format.md) — Status updates: per-seed columns with % budget + best val_bpb
- [feedback_budget_percentage.md](feedback_budget_percentage.md) — Runs exceeding 24h are cropped and should show 100%, not 99.7%
- [feedback_budget_calc.md](feedback_budget_calc.md) — Budget % must sum wall_time of ALL trials (success + failure), not just successful
- [feedback_time_fields.md](feedback_time_fields.md) — wall_time_seconds in trials.jsonl = training time; elapsed_s in per-trial files = LLM call time only
- [feedback_overleaf_conflicts.md](feedback_overleaf_conflicts.md) — Always fetch Overleaf before editing or pushing; never auto-resolve conflicts
- [feedback_optimizer_optimizee.md](feedback_optimizer_optimizee.md) — Use optimizer/optimizee to distinguish Qwen3.5 from the 50M model being trained

## References
- [reference_paper_repo.md](reference_paper_repo.md) — Paper LaTeX repo: github.com/ferreirafabio/autoresearch-automl-paper
- [reference_horeka.md](reference_horeka.md) — How to run jobs on HoreKa from kislurm (SSH proxy, OTP, partitions)
