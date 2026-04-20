"""Convergence plot comparing Opus 4.6 vs Opus 4.7 across Centaur / KA HPs / KA Code,
with TPE (best classical method) as reference.

Custom: legend is grouped by Opus version (Classical / Opus 4.6 / Opus 4.7) rather
than sorted by performance. Within each group, methods keep the order Centaur,
KA HPs, KA Code.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/results")
OPUS46_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/opus46_benchmark")
OPUS47_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/opus47_benchmark")


def load_trials(jsonl_path: Path) -> list[dict]:
    trials = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trials


def cumulative_walltime(trials: list[dict], cap_hours: float = 24.0):
    cum_time = 0.0
    best = float("inf")
    times = []
    values = []
    for t in trials:
        wt = t.get("wall_time_seconds") or 0.0
        cum_time += wt
        if cum_time / 3600 > cap_hours:
            break
        if t["success"] and t["val_bpb"] is not None:
            best = min(best, t["val_bpb"])
        times.append(cum_time / 3600)
        values.append(best)
    return times, values


def build_backends() -> dict[str, dict]:
    """Color encodes method family; linestyle encodes model version.

    Group keys: 'classical' / 'opus46' / 'opus47' — used for legend grouping.
    """
    return {
        # Classical reference
        "optuna": {
            "label": "TPE",
            "group": "classical",
            "color": "#E69F00",
            "linestyle": ":",
            "path": str(RESULTS_BASE / "optuna"),
        },
        # Opus 4.6 block
        "centaur_claude_opus_4_6": {
            "label": "Centaur",
            "group": "opus46",
            "color": "#C62828",
            "linestyle": "-",
            "path": str(OPUS46_BASE / "centaur_claude_opus_4_6"),
        },
        "karpathy_agent_hps_claude_opus_4_6": {
            "label": "KA HPs",
            "group": "opus46",
            "color": "#00897B",
            "linestyle": "-",
            "path": str(OPUS46_BASE / "karpathy_agent_hps_claude_opus_4_6"),
        },
        "karpathy_agent_claude_opus_4_6": {
            "label": "KA Code",
            "group": "opus46",
            "color": "#1565C0",
            "linestyle": "-",
            "path": str(OPUS46_BASE / "karpathy_agent_claude_opus_4_6"),
        },
        # Opus 4.7 block
        "centaur_claude_opus_4_7": {
            "label": "Centaur",
            "group": "opus47",
            "color": "#C62828",
            "linestyle": "--",
            "path": str(OPUS47_BASE / "centaur_claude_opus_4_7"),
        },
        "karpathy_agent_hps_claude_opus_4_7": {
            "label": "KA HPs",
            "group": "opus47",
            "color": "#00897B",
            "linestyle": "--",
            "path": str(OPUS47_BASE / "karpathy_agent_hps_claude_opus_4_7"),
        },
        "karpathy_agent_claude_opus_4_7": {
            "label": "KA Code",
            "group": "opus47",
            "color": "#1565C0",
            "linestyle": "--",
            "path": str(OPUS47_BASE / "karpathy_agent_claude_opus_4_7"),
        },
    }


GROUP_HEADERS = {
    "classical": "Classical",
    "opus46":    "Opus 4.6",
    "opus47":    "Opus 4.7",
}


def plot(backends: dict[str, dict], output_path: Path, title: str,
         ylim=(0.973, 0.993), max_bands: int = 7):
    fig, ax = plt.subplots(figsize=(10, 6))

    MIN_TRIALS = 100
    MIN_BUDGET_FRAC = 0.90
    BUDGET_SECONDS = 86400
    INTERP_HOURS = np.linspace(0, 24, 1000)

    # Collect per-backend mean/std, preserving insertion order
    entries = []
    for key, style in backends.items():
        base = Path(style["path"])
        seed_interps = []
        for seed in range(3):
            jsonl = base / f"seed_{seed}" / "trials.jsonl"
            if not jsonl.exists():
                continue
            trials = load_trials(jsonl)
            if len(trials) < MIN_TRIALS:
                continue
            total_s = sum(t.get("wall_time_seconds") or 0.0 for t in trials)
            if total_s < BUDGET_SECONDS * MIN_BUDGET_FRAC:
                continue
            times, values = cumulative_walltime(trials)
            if not times or values[0] == float("inf"):
                continue
            right_val = values[-1] if total_s >= BUDGET_SECONDS * 0.98 else np.nan
            seed_interps.append(np.interp(INTERP_HOURS, times, values, left=np.nan, right=right_val))

        if not seed_interps:
            continue

        aligned = np.array(seed_interps)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)

        line, = ax.plot(INTERP_HOURS, mean, color=style["color"], linewidth=2,
                        linestyle=style.get("linestyle", "-"))
        entries.append({
            "group": style["group"],
            "label": style["label"],
            "handle": line,
            "color": style["color"],
            "mean": mean,
            "std": std,
            "best": np.nanmin(mean),
        })

    # Bands for top methods (sorted by best val)
    for e in sorted(entries, key=lambda x: x["best"])[:max_bands]:
        ax.fill_between(INTERP_HOURS, e["mean"] - e["std"], e["mean"] + e["std"],
                        color=e["color"], alpha=0.08)

    ax.set_xlabel("Cumulative training time (hours)", fontsize=12)
    ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(True, alpha=0.3)

    # Grouped legend: one header line per group, then its entries in insertion order
    if entries:
        from matplotlib.lines import Line2D
        max_label_len = max(len(e["label"]) for e in entries)
        spacer = Line2D([], [], color="none", marker="", linestyle="none")

        handles, labels = [], []
        for group_key in ("classical", "opus46", "opus47"):
            group_entries = [e for e in entries if e["group"] == group_key]
            if not group_entries:
                continue
            # Header row (italic-ish via bold font-weight in monospace)
            handles.append(spacer)
            labels.append(f"— {GROUP_HEADERS[group_key]} —")
            for e in group_entries:
                handles.append(e["handle"])
                labels.append(f"{e['label'].ljust(max_label_len + 2)}{e['best']:.4f}")

        ax.legend(handles, labels, fontsize=8, loc="upper right",
                  prop={"family": "monospace", "size": 8})

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    output_path = Path(
        "/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/"
        "assets/exp3_opus46_vs_opus47.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot(
        backends=build_backends(),
        output_path=output_path,
        title="autoresearch Opus 4.6 vs. Opus 4.7",
    )


if __name__ == "__main__":
    main()
