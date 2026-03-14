"""Plot convergence curves for TPE vs LLAMBO (and Exp1 model size comparison)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trials(jsonl_path: Path) -> list[dict]:
    trials = []
    with open(jsonl_path) as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def best_so_far(trials: list[dict]) -> list[float]:
    """Compute incumbent (best val_bpb so far) at each trial."""
    best = float("inf")
    curve = []
    for t in trials:
        if t["success"] and t["val_bpb"] is not None:
            best = min(best, t["val_bpb"])
        curve.append(best)
    return curve


def plot_exp1(results_dir: Path, output_path: Path):
    """Exp1: LLM model size comparison (0.8B vs 9B)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Qwen3.5-0.8B": "#2196F3", "Qwen3.5-9B": "#FF5722"}

    for model_name in ["Qwen3.5-0.8B", "Qwen3.5-9B"]:
        jsonl = results_dir / "exp1_model_size" / model_name / "trials.jsonl"
        if not jsonl.exists():
            continue
        trials = load_trials(jsonl)
        curve = best_so_far(trials)
        # Only plot from first successful trial
        first_valid = next((i for i, v in enumerate(curve) if v < float("inf")), None)
        if first_valid is None:
            continue
        ax.plot(
            range(first_valid, len(curve)),
            curve[first_valid:],
            label=model_name,
            color=colors[model_name],
            linewidth=2,
        )

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Best val_bpb", fontsize=12)
    ax.set_title("Exp1: LLM Model Size for HP Suggestions", fontsize=13)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_exp2(results_dir: Path, output_path: Path):
    """Exp2: TPE vs LLAMBO convergence with seed bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    backends = {
        "optuna": {"label": "TPE", "color": "#2196F3"},
        "llambo": {"label": "LLAMBO", "color": "#9C27B0"},
    }

    for backend_name, style in backends.items():
        seed_curves = []
        for seed in range(3):
            jsonl = results_dir / "exp2_benchmark" / backend_name / f"seed_{seed}" / "trials.jsonl"
            if not jsonl.exists():
                continue
            trials = load_trials(jsonl)
            curve = best_so_far(trials)
            seed_curves.append(curve)

        if not seed_curves:
            continue

        # Align to shortest curve
        min_len = min(len(c) for c in seed_curves)
        aligned = np.array([c[:min_len] for c in seed_curves])

        # Replace inf with nan for plotting
        aligned[aligned == float("inf")] = np.nan

        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)

        x = np.arange(min_len)
        ax.plot(x, mean, label=style["label"], color=style["color"], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15)

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Best val_bpb", fontsize=12)
    ax.set_title("Exp2: TPE vs LLAMBO", fontsize=13)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    results_dir = Path("/work/dlclarge1/ferreira-autoresearch-automl/results")
    assets_dir = Path("/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/assets")
    assets_dir.mkdir(exist_ok=True)

    plot_exp1(results_dir, assets_dir / "exp1_model_size.png")
    plot_exp2(results_dir, assets_dir / "exp2_tpe_vs_llambo.png")
