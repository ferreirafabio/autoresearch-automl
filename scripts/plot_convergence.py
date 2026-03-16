"""Plot convergence curves for Exp2 benchmark: TPE vs LLAMBO vs LLM Greedy."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def best_so_far(trials: list[dict]) -> list[float]:
    """Compute incumbent (best val_bpb so far) at each trial."""
    best = float("inf")
    curve = []
    for t in trials:
        if t["success"] and t["val_bpb"] is not None:
            best = min(best, t["val_bpb"])
        curve.append(best)
    return curve


def plot_convergence_multi(
    results_dir: Path,
    backends: dict[str, dict],
    output_path: Path,
    title: str = "Convergence",
    ylim: tuple[float, float] = (0.97, 1.05),
    xlim: tuple[int, int] | None = None,
):
    """Plot convergence with mean +/- std across seeds for multiple backends."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for backend_dir, style in backends.items():
        seed_curves = []
        for seed in range(3):
            jsonl = results_dir / backend_dir / f"seed_{seed}" / "trials.jsonl"
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
        aligned[aligned == float("inf")] = np.nan

        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)

        x = np.arange(min_len)
        n_seeds = len(seed_curves)
        label = f"{style['label']} ({n_seeds} seeds, {min_len} trials)"
        ax.plot(x, mean, label=label, color=style["color"], linewidth=2,
                linestyle=style.get("linestyle", "-"))
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.12)

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_exp2_0_8b(results_dir: Path, output_path: Path):
    """Exp2: 0.8B model — TPE vs LLAMBO (OptunaHub) vs LLM Greedy."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3"},
        "llambo": {"label": "LLAMBO (OptunaHub) 0.8B", "color": "#9C27B0"},
        "llm_greedy": {"label": "LLM Greedy 0.8B", "color": "#FF9800"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Exp2: HPO Backend Comparison (Qwen3.5-0.8B)",
        ylim=(0.97, 1.05),
        xlim=(0, 200),
    )


def plot_exp2_27b_think(results_dir: Path, output_path: Path):
    """Exp2: 27B thinking — all backends."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3"},
        "llm_greedy_Qwen3_5_27B": {"label": "LLM Greedy 27B think", "color": "#FF9800"},
        "llambo_Qwen3_5_27B": {"label": "LLAMBO (OptunaHub) 27B think", "color": "#9C27B0"},
        "llambo_original_Qwen3_5_27B": {"label": "LLAMBO (Original) 27B think", "color": "#E91E63"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Exp2: HPO Backend Comparison (Qwen3.5-27B, thinking ON)",
        ylim=(0.97, 1.05),
        xlim=(0, 200),
    )


def plot_exp2_27b_nothink(results_dir: Path, output_path: Path):
    """Exp2: 27B no thinking — all backends."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "LLM Greedy 27B nothink", "color": "#FF9800"},
        "llambo_Qwen3_5_27B_nothink": {"label": "LLAMBO (OptunaHub) 27B nothink", "color": "#9C27B0"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Original) 27B nothink", "color": "#E91E63"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Exp2: HPO Backend Comparison (Qwen3.5-27B, thinking OFF)",
        ylim=(0.97, 1.05),
        xlim=(0, 200),
    )


def plot_exp2_model_comparison(results_dir: Path, output_path: Path):
    """Compare 0.8B vs 27B (think vs nothink) for each LLM backend."""
    backends = {
        "llm_greedy": {"label": "LLM Greedy 0.8B", "color": "#FF9800", "linestyle": "-"},
        "llm_greedy_Qwen3_5_27B": {"label": "LLM Greedy 27B think", "color": "#FF9800", "linestyle": "--"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "LLM Greedy 27B nothink", "color": "#FF9800", "linestyle": ":"},
        "llambo": {"label": "LLAMBO (OH) 0.8B", "color": "#9C27B0", "linestyle": "-"},
        "llambo_Qwen3_5_27B": {"label": "LLAMBO (OH) 27B think", "color": "#9C27B0", "linestyle": "--"},
        "llambo_Qwen3_5_27B_nothink": {"label": "LLAMBO (OH) 27B nothink", "color": "#9C27B0", "linestyle": ":"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Exp2: Model Size Comparison (0.8B vs 27B)",
        ylim=(0.97, 1.05),
        xlim=(0, 200),
    )


def plot_progress(results_dir: Path, output_path: Path):
    """Progress plot: scatter + running best staircase for available backends."""
    bench_dir = results_dir / "exp2_benchmark"

    # Find all backends with data
    all_backends = []
    for d in sorted(bench_dir.iterdir()):
        if d.is_dir() and (d / "seed_0" / "trials.jsonl").exists():
            all_backends.append(d.name)

    if not all_backends:
        print("No data for progress plot")
        return

    colors = {
        "optuna": "#2196F3",
        "llambo": "#9C27B0",
        "llm_greedy": "#FF9800",
        "llambo_original": "#E91E63",
    }

    n = len(all_backends)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True, squeeze=False)

    for idx, backend_name in enumerate(all_backends):
        ax = axes[idx // cols][idx % cols]
        jsonl = bench_dir / backend_name / "seed_0" / "trials.jsonl"
        trials = load_trials(jsonl)
        if not trials:
            ax.set_title(f"{backend_name} (no data)")
            continue

        # Pick color based on base name
        color = "#666666"
        for key, c in colors.items():
            if backend_name.startswith(key):
                color = c
                break

        val_bpbs = []
        for t in trials:
            val_bpbs.append(t["val_bpb"] if t["success"] and t["val_bpb"] is not None else None)

        success_x = [i for i, v in enumerate(val_bpbs) if v is not None]
        success_y = [v for v in val_bpbs if v is not None]

        best = float("inf")
        kept_x, kept_y = [], []
        disc_x, disc_y = [], []
        for i, v in zip(success_x, success_y):
            if v < best:
                best = v
                kept_x.append(i)
                kept_y.append(v)
            else:
                disc_x.append(i)
                disc_y.append(v)

        ax.scatter(disc_x, disc_y, c="#cccccc", s=12, alpha=0.5, zorder=2)
        ax.scatter(kept_x, kept_y, c=color, s=50, zorder=4,
                   edgecolors="black", linewidths=0.5, label="Kept")

        curve = best_so_far(trials)
        valid_curve = [(i, v) for i, v in enumerate(curve) if v < float("inf")]
        if valid_curve:
            stair_x, stair_y = zip(*valid_curve)
            ax.step(stair_x, stair_y, where="post", color=color,
                    linewidth=2, alpha=0.7, zorder=3)

        n_success = len(success_x)
        n_fail = len(trials) - n_success
        best_str = f"{best:.4f}" if best < float("inf") else "N/A"
        ax.set_title(f"{backend_name}\n{len(trials)} trials, {n_fail} failed, best={best_str}", fontsize=10)
        ax.set_xlabel("Trial #", fontsize=10)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    axes[0][0].set_ylabel("val_bpb (lower is better)", fontsize=11)
    fig.suptitle("Exp2 Progress (seed 0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    results_dir = Path("/work/dlclarge1/ferreira-autoresearch-automl/results")
    assets_dir = Path("/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/assets")
    assets_dir.mkdir(exist_ok=True)

    plot_exp2_0_8b(results_dir, assets_dir / "exp2_0.8b_convergence.png")
    plot_exp2_27b_think(results_dir, assets_dir / "exp2_27b_think_convergence.png")
    plot_exp2_27b_nothink(results_dir, assets_dir / "exp2_27b_nothink_convergence.png")
    plot_exp2_model_comparison(results_dir, assets_dir / "exp2_model_comparison.png")
    plot_progress(results_dir, assets_dir / "exp2_progress.png")
