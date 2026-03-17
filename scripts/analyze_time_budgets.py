#!/usr/bin/env python3
"""Analyze training-time budgets across all exp2 backends/seeds."""

import json
from pathlib import Path

RESULTS_DIR = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark")
BUDGET_24H = 86400.0  # 24 hours in seconds

BACKENDS = [
    "optuna",
    "random",
    "smac",
    "cma_es",
    "llambo_Qwen3_5_27B_nothink",
    "llambo_original_Qwen3_5_27B_nothink",
    "llm_greedy_Qwen3_5_27B_nothink",
]
SEEDS = [0, 1, 2]


def analyze_trials(jsonl_path: Path) -> dict | None:
    if not jsonl_path.exists():
        return None
    records = []
    for line in jsonl_path.read_text().strip().splitlines():
        if line:
            records.append(json.loads(line))
    if not records:
        return None

    n_trials = len(records)
    total_train_time = sum(r.get("wall_time_seconds") or 0 for r in records)
    timestamps = [r.get("timestamp", 0) for r in records if r.get("timestamp")]
    wall_clock_span = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0
    successful = sum(1 for r in records if r.get("success"))
    best_bpb = min((r["val_bpb"] for r in records if r.get("val_bpb") is not None), default=None)

    return {
        "n_trials": n_trials,
        "successful": successful,
        "total_train_time_s": total_train_time,
        "total_train_time_h": total_train_time / 3600,
        "wall_clock_span_h": wall_clock_span / 3600,
        "budget_met": total_train_time >= BUDGET_24H,
        "remaining_h": max(0, (BUDGET_24H - total_train_time) / 3600),
        "best_val_bpb": best_bpb,
    }


def find_cutoff_trial(jsonl_path: Path, budget: float = BUDGET_24H) -> dict | None:
    """Find the trial at which the cumulative training time exceeds the budget."""
    if not jsonl_path.exists():
        return None
    records = []
    for line in jsonl_path.read_text().strip().splitlines():
        if line:
            records.append(json.loads(line))
    if not records:
        return None

    cumulative = 0.0
    for i, r in enumerate(records):
        cumulative += r.get("wall_time_seconds") or 0
        if cumulative >= budget:
            return {
                "cutoff_trial_idx": i,
                "cutoff_trial_id": r.get("trial_id"),
                "cumulative_at_cutoff_h": cumulative / 3600,
                "total_trials": len(records),
            }
    return None  # budget not reached


def main():
    print(f"{'Backend':<40} {'Seed':>4} {'Trials':>6} {'OK':>4} {'Train(h)':>9} {'Clock(h)':>9} {'Best BPB':>9} {'24h?':>5} {'Remaining':>9} {'Cutoff':>10}")
    print("-" * 125)

    for backend in BACKENDS:
        for seed in SEEDS:
            path = RESULTS_DIR / backend / f"seed_{seed}" / "trials.jsonl"
            info = analyze_trials(path)
            if info is None:
                print(f"{backend:<40} {seed:>4}   -- no data --")
                continue
            status = "YES" if info["budget_met"] else "NO"
            bpb_str = f"{info['best_val_bpb']:.4f}" if info["best_val_bpb"] is not None else "N/A"
            remaining = f"{info['remaining_h']:.2f}h" if not info["budget_met"] else "-"

            # Find cutoff for over-budget runs
            cutoff_str = "-"
            if info["budget_met"]:
                cutoff = find_cutoff_trial(path)
                if cutoff:
                    cutoff_str = f"@{cutoff['cutoff_trial_id']} ({cutoff['cumulative_at_cutoff_h']:.2f}h)"

            print(
                f"{backend:<40} {seed:>4} {info['n_trials']:>6} {info['successful']:>4} "
                f"{info['total_train_time_h']:>8.2f}h {info['wall_clock_span_h']:>8.2f}h "
                f"{bpb_str:>9} {status:>5} {remaining:>9} {cutoff_str:>10}"
            )
        print()


if __name__ == "__main__":
    main()
