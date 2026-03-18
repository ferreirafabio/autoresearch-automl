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
    ylim: tuple[float, float] = (0.975, 1.012),
    xlim: tuple[int, int] | None = None,
):
    """Plot convergence with mean +/- std across seeds for multiple backends."""
    fig, ax = plt.subplots(figsize=(10, 6))

    max_trials = 0
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
        max_trials = max(max_trials, min_len)
        aligned = np.array([c[:min_len] for c in seed_curves])
        aligned[aligned == float("inf")] = np.nan

        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)

        x = np.arange(min_len)
        best_val = np.nanmin(mean)
        label = f"{style['label']} (best={best_val:.4f})"
        ax.plot(x, mean, label=label, color=style["color"], linewidth=2,
                linestyle=style.get("linestyle", "-"))
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.12)

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, max_trials)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_exp2_0_8b(results_dir: Path, output_path: Path):
    """Exp2: 0.8B model — all backends."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3"},
        "llambo": {"label": "LLAMBO (Optuna) 0.8B", "color": "#9C27B0"},
        "llambo_original": {"label": "LLAMBO (Original) 0.8B", "color": "#E91E63"},
        "llm_greedy": {"label": "LLM Greedy 0.8B", "color": "#FF9800"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Karpathy's Autoresearch: HPO Convergence (0.8B Optimizer)",
    )


def plot_exp2_27b(results_dir: Path, output_path: Path):
    """Exp2: 27B — all backends."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3"},
        "cma_es": {"label": "CMA-ES", "color": "#00796B"},
        "centaur_Qwen3_5_27B": {"label": "Centaur 27B", "color": "#D32F2F"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "LLM Greedy 27B", "color": "#FF9800"},
        "llambo_Qwen3_5_27B_nothink": {"label": "LLAMBO (Optuna) 27B", "color": "#9C27B0"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Original) 27B", "color": "#E91E63"},
        "random": {"label": "Random", "color": "#607D8B"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Karpathy's Autoresearch: HPO Convergence (27B + Classical)",
    )


def plot_exp2_all(results_dir: Path, output_path: Path):
    """All backends on one plot."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3", "linestyle": "-"},
        "cma_es": {"label": "CMA-ES", "color": "#00796B", "linestyle": "-"},
        "centaur_Qwen3_5_27B": {"label": "Centaur 27B", "color": "#D32F2F", "linestyle": "-"},
        "random": {"label": "Random", "color": "#607D8B", "linestyle": "-"},
        "smac": {"label": "SMAC", "color": "#795548", "linestyle": "-"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "LLM Greedy 27B", "color": "#FF9800", "linestyle": "-"},
        "llambo_Qwen3_5_27B_nothink": {"label": "LLAMBO (Optuna) 27B", "color": "#9C27B0", "linestyle": "-"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Orig) 27B", "color": "#E91E63", "linestyle": "-"},
        "karpathy_agent_Qwen3_5_27B": {"label": "Karpathy Agent 27B", "color": "#4CAF50", "linestyle": "-"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Karpathy's Autoresearch: All HPO Methods",
    )


def plot_exp2_model_size(results_dir: Path, output_path: Path):
    """Compare 0.8B vs 27B for each LLM backend."""
    backends = {
        "optuna": {"label": "TPE (Optuna)", "color": "#2196F3", "linestyle": "-"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "LLM Greedy 27B", "color": "#FF9800", "linestyle": "-"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Orig) 27B", "color": "#E91E63", "linestyle": "-"},
        "karpathy_agent_Qwen3_5_0_8B": {"label": "Karpathy Agent 0.8B", "color": "#4CAF50", "linestyle": "--"},
        "karpathy_agent_Qwen3_5_27B": {"label": "Karpathy Agent 27B", "color": "#4CAF50", "linestyle": "-"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Karpathy's Autoresearch: Does Optimizer LLM Size Matter?",
    )


HP_SHORT = {
    "ASPECT_RATIO": "AR",
    "DEPTH": "D",
    "DEVICE_BATCH_SIZE": "DBS",
    "EMBEDDING_LR": "emb_lr",
    "FINAL_LR_FRAC": "final_lr",
    "HEAD_DIM": "HD",
    "MATRIX_LR": "mat_lr",
    "SCALAR_LR": "scl_lr",
    "TOTAL_BATCH_SIZE": "TBS",
    "UNEMBEDDING_LR": "uemb_lr",
    "WARMDOWN_RATIO": "warmdown",
    "WARMUP_RATIO": "warmup",
    "WEIGHT_DECAY": "wd",
    "WINDOW_PATTERN": "WP",
}


def _format_val(v):
    """Format HP value concisely."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 0.01:
            return f"{v:.4f}"
        return f"{v:g}"
    if isinstance(v, int) and v >= 10000:
        return f"{v // 1000}K"
    return str(v)


def _config_diff(prev: dict, curr: dict, max_diffs: int = 2) -> str:
    """Karpathy-style HP change description: 'warmdown 0.5→0.7, TBS 524K→262K'."""
    diffs = []
    for hp in curr:
        if hp not in prev or prev[hp] != curr[hp]:
            short = HP_SHORT.get(hp, hp)
            if hp in prev:
                diffs.append(f"{short} {_format_val(prev[hp])}→{_format_val(curr[hp])}")
            else:
                diffs.append(f"{short}={_format_val(curr[hp])}")
    if not diffs:
        return "baseline"
    if len(diffs) > max_diffs:
        return ", ".join(diffs[:max_diffs]) + f" +{len(diffs) - max_diffs}"
    return ", ".join(diffs)


def plot_progress_single(
    bench_dir: Path,
    backend_name: str,
    display_name: str,
    color: str,
    output_path: Path,
):
    """Karpathy-style progress plot: scatter + annotated Pareto front."""
    jsonl = bench_dir / backend_name / "seed_0" / "trials.jsonl"
    if not jsonl.exists():
        print(f"No data for {backend_name}")
        return

    trials = load_trials(jsonl)
    if not trials:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    val_bpbs = []
    for t in trials:
        val_bpbs.append(t["val_bpb"] if t["success"] and t["val_bpb"] is not None else None)

    success_x = [i for i, v in enumerate(val_bpbs) if v is not None]
    success_y = [v for v in val_bpbs if v is not None]

    # Find incumbents (Pareto front)
    best = float("inf")
    incumbents = []  # (trial_idx, val_bpb, config)
    disc_x, disc_y = [], []
    for i, v in zip(success_x, success_y):
        if v < best:
            best = v
            incumbents.append((i, v, trials[i]["config"]))
        else:
            disc_x.append(i)
            disc_y.append(v)

    # Scatter discarded
    ax.scatter(disc_x, disc_y, c="#cccccc", s=15, alpha=0.4, zorder=2, label="Discarded")

    # Scatter + staircase for incumbents
    inc_x = [p[0] for p in incumbents]
    inc_y = [p[1] for p in incumbents]
    ax.scatter(inc_x, inc_y, c=color, s=60, zorder=4,
               edgecolors="black", linewidths=0.5, label="Kept")

    curve = best_so_far(trials)
    valid_curve = [(i, v) for i, v in enumerate(curve) if v < float("inf")]
    if valid_curve:
        stair_x, stair_y = zip(*valid_curve)
        ax.step(stair_x, stair_y, where="post", color=color,
                linewidth=2, alpha=0.6, zorder=3, label="Running best")

    # Annotate incumbents — Karpathy style
    prev_config = None
    for idx, (trial_i, val, config) in enumerate(incumbents):
        if prev_config is None:
            label = "baseline"
        else:
            label = _config_diff(prev_config, config)
            if len(label) > 45:
                label = label[:42] + "..."
        prev_config = config
        ax.annotate(
            label, xy=(trial_i, val), xytext=(6, 6),
            textcoords="offset points", fontsize=8, color="#1a7a3a",
            alpha=0.9, rotation=30, ha="left", va="bottom",
            annotation_clip=True,
        )

    n_fail = sum(1 for t in trials if not t.get("success", False))
    ax.set_title(
        f"Karpathy's Autoresearch: {display_name}\n"
        f"{len(trials)} trials, {len(incumbents)} improvements, {n_fail} failed",
        fontsize=13,
    )
    ax.set_xlabel("Trial #", fontsize=12)
    ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
    ax.set_xlim(0, len(trials))
    ax.set_ylim(0.975, 1.012)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_progress(results_dir: Path, assets_dir: Path):
    """Generate per-backend Karpathy-style progress plots."""
    bench_dir = results_dir / "exp2_benchmark"

    backends_to_plot = [
        ("optuna", "TPE (Optuna)", "#2196F3"),
        ("llm_greedy", "LLM Greedy 0.8B", "#FF9800"),
        ("llm_greedy_Qwen3_5_27B_nothink", "LLM Greedy 27B", "#FF9800"),
        ("llambo_original", "LLAMBO (Original) 0.8B", "#E91E63"),
        ("llambo_original_Qwen3_5_27B_nothink", "LLAMBO (Original) 27B", "#E91E63"),
        ("llambo", "LLAMBO (Optuna) 0.8B", "#9C27B0"),
        ("llambo_Qwen3_5_27B_nothink", "LLAMBO (Optuna) 27B", "#9C27B0"),
    ]

    for backend_name, display_name, color in backends_to_plot:
        safe_name = backend_name.replace("/", "_")
        plot_progress_single(
            bench_dir, backend_name, display_name, color,
            assets_dir / f"progress_{safe_name}.png",
        )


def plot_progress_subplot(ax, bench_dir, backend_name, display_name, color, descriptions=None):
    """Draw a single Pareto-front progress plot on the given axes."""
    jsonl = bench_dir / backend_name / "seed_0" / "trials.jsonl"
    if not jsonl.exists():
        ax.set_title(f"{display_name}\n(no data)", fontsize=10)
        return

    trials = load_trials(jsonl)
    if not trials:
        return

    val_bpbs = []
    for t in trials:
        val_bpbs.append(t["val_bpb"] if t["success"] and t["val_bpb"] is not None else None)

    success_x = [i for i, v in enumerate(val_bpbs) if v is not None]
    success_y = [v for v in val_bpbs if v is not None]

    best = float("inf")
    incumbents = []
    disc_x, disc_y = [], []
    for i, v in zip(success_x, success_y):
        if v < best:
            best = v
            incumbents.append((i, v, trials[i]["config"]))
        else:
            disc_x.append(i)
            disc_y.append(v)

    ax.scatter(disc_x, disc_y, c="#cccccc", s=8, alpha=0.4, zorder=2)

    inc_x = [p[0] for p in incumbents]
    inc_y = [p[1] for p in incumbents]
    ax.scatter(inc_x, inc_y, c=color, s=30, zorder=4,
               edgecolors="black", linewidths=0.5)

    curve = best_so_far(trials)
    valid_curve = [(i, v) for i, v in enumerate(curve) if v < float("inf")]
    if valid_curve:
        stair_x, stair_y = zip(*valid_curve)
        ax.step(stair_x, stair_y, where="post", color=color,
                linewidth=1.5, alpha=0.6, zorder=3)

    # Annotate incumbents — use LLM descriptions if available, else auto-diff
    # descriptions: dict mapping trial_idx -> description string
    desc_map = {}
    if descriptions:
        for d in descriptions:
            desc_map[d["trial"]] = d["description"]

    prev_config = None
    for idx, (trial_i, val, config) in enumerate(incumbents):
        if trial_i in desc_map:
            label = desc_map[trial_i]
        elif prev_config is None:
            label = "baseline"
        else:
            label = _config_diff(prev_config, config)
        if len(label) > 35:
            label = label[:32] + "..."
        prev_config = config
        ax.annotate(
            label, xy=(trial_i, val), xytext=(6, 6),
            textcoords="offset points", fontsize=5.5, color="#1a7a3a",
            alpha=0.9, rotation=30, ha="left", va="bottom",
            annotation_clip=True,
        )

    n_fail = sum(1 for t in trials if not t.get("success", False))
    ax.set_title(f"{display_name}\n{len(trials)} trials, {len(incumbents)} impr., {n_fail} fail",
                 fontsize=9)
    ax.set_xlim(0, len(trials))
    ax.set_ylim(0.975, 1.012)
    ax.grid(True, alpha=0.2)


def plot_progress_combined(results_dir: Path, output_path: Path):
    """Combined incumbent traces plot: 2x2 grid, 27B backends."""
    bench_dir = results_dir / "exp2_benchmark"

    # Load cached LLM descriptions if available
    desc_path = Path(__file__).parent.parent / "assets" / "incumbent_descriptions.json"
    all_descriptions = {}
    if desc_path.exists():
        all_descriptions = json.loads(desc_path.read_text())

    backends = [
        ("optuna", "TPE (Optuna)", "#2196F3"),
        ("cma_es", "CMA-ES", "#00796B"),
        ("centaur_Qwen3_5_27B", "Centaur 27B", "#D32F2F"),
        ("llambo_original_Qwen3_5_27B_nothink", "LLAMBO (Original) 27B", "#E91E63"),
        ("llm_greedy_Qwen3_5_27B_nothink", "LLM Greedy 27B", "#FF9800"),
        ("random", "Random", "#607D8B"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (backend, name, color) in enumerate(backends):
        row, col = divmod(idx, 3)
        descs = all_descriptions.get(backend)
        plot_progress_subplot(axes[row, col], bench_dir, backend, name, color, descriptions=descs)

    for ax in axes[1, :]:
        ax.set_xlabel("Trial #", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("val_bpb", fontsize=10)

    fig.suptitle("Incumbent Traces (seed 0)", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    results_dir = Path("/work/dlclarge1/ferreira-autoresearch-automl/results")
    assets_dir = Path("/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/assets")
    assets_dir.mkdir(exist_ok=True)

    plot_exp2_27b(results_dir, assets_dir / "exp2_27b_convergence.png")
    plot_exp2_all(results_dir, assets_dir / "exp2_all_convergence.png")
    plot_exp2_model_size(results_dir, assets_dir / "exp2_model_size.png")
    plot_progress(results_dir, assets_dir)
    plot_progress_combined(results_dir, assets_dir / "exp2_pareto_fronts.png")
