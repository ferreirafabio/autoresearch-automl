"""Plot convergence curves for Exp2 benchmark: classical vs LLM-based vs hybrid HPO."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


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
    ranking = []  # (best_val, label, color)
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

        # Align to longest curve (pad shorter ones with their last value)
        max_len = max(len(c) for c in seed_curves)
        max_trials = max(max_trials, max_len)
        padded = []
        for c in seed_curves:
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))
            padded.append(c)
        aligned = np.array([c[:max_len] for c in padded])
        aligned[aligned == float("inf")] = np.nan

        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)

        x = np.arange(max_len)
        best_val = np.nanmin(mean)
        line, = ax.plot(x, mean, color=style["color"], linewidth=2,
                        linestyle=style.get("linestyle", "-"))
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.12)
        ranking.append((best_val, line, style["label"]))

    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, max_trials)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # Sorted monospace legend with header and right-aligned best scores
    if ranking:
        ranking.sort(key=lambda r: r[0])
        max_name_len = max(len(r[2]) for r in ranking)
        # Header row (invisible handle)
        header_handle, = ax.plot([], [], color="none", marker="none", linestyle="none")
        handles = [header_handle]
        labels = [f"{'Method'.ljust(max_name_len + 2)}{'Best':>6}"]
        for val, handle, name in ranking:
            labels.append(f"{name.ljust(max_name_len + 2)}{val:.4f}")
            handles.append(handle)
        ax.legend(handles, labels, fontsize=8, loc="upper right",
                  prop={"family": "monospace", "size": 8})

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_exp2_0_8b(results_dir: Path, output_path: Path):
    """Exp2: 0.8B model — all backends."""
    backends = {
        "optuna": {"label": "TPE", "color": "#2196F3"},
        "llambo": {"label": "LLAMBO (Optuna) [0.8B]", "color": "#9C27B0"},
        "llambo_original": {"label": "LLAMBO (Paper) [0.8B]", "color": "#00BCD4"},
        "llm_greedy": {"label": "Karpathy Agent (14 HPs) [0.8B]", "color": "#FF9800"},
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
        "optuna": {"label": "TPE", "color": "#2196F3"},
        "cma_es": {"label": "CMA-ES", "color": "#00796B"},
        "centaur_Qwen3_5_27B": {"label": "Centaur [27B]", "color": "#D32F2F"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "Karpathy Agent (14 HPs) [27B]", "color": "#FF9800"},
        "llambo_Qwen3_5_27B_nothink": {"label": "LLAMBO (Optuna) [27B]", "color": "#9C27B0"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Paper) [27B]", "color": "#00BCD4"},
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
        "optuna": {"label": "TPE", "color": "#2196F3", "linestyle": "-"},
        "cma_es": {"label": "CMA-ES", "color": "#00796B", "linestyle": "-"},
        "centaur_Qwen3_5_27B": {"label": "Centaur [27B]", "color": "#D32F2F", "linestyle": "-"},
        "random": {"label": "Random", "color": "#607D8B", "linestyle": "-"},
        "smac": {"label": "SMAC", "color": "#8BC34A", "linestyle": "-"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "Karpathy Agent (14 HPs) [27B]", "color": "#FF9800", "linestyle": "-"},
        "llambo_Qwen3_5_27B_nothink": {"label": "LLAMBO (Optuna) [27B]", "color": "#9C27B0", "linestyle": "-"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Paper) [27B]", "color": "#00BCD4", "linestyle": "-"},
        "karpathy_agent_Qwen3_5_27B": {"label": "Karpathy Agent (Code) [27B]", "color": "#795548", "linestyle": "-"},
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
        "optuna": {"label": "TPE", "color": "#2196F3", "linestyle": "-"},
        "llm_greedy_Qwen3_5_27B_nothink": {"label": "Karpathy Agent (14 HPs) [27B]", "color": "#FF9800", "linestyle": "-"},
        "llambo_original_Qwen3_5_27B_nothink": {"label": "LLAMBO (Paper) [27B]", "color": "#00BCD4", "linestyle": "-"},
        "karpathy_agent_Qwen3_5_0_8B": {"label": "Karpathy Agent (Code) [0.8B]", "color": "#795548", "linestyle": "--"},
        "karpathy_agent_Qwen3_5_27B": {"label": "Karpathy Agent (Code) [27B]", "color": "#795548", "linestyle": "-"},
    }
    plot_convergence_multi(
        results_dir / "exp2_benchmark",
        backends,
        output_path,
        title="Karpathy's Autoresearch: Does Optimizer LLM Size Matter?",
    )


HP_HUMAN = {
    "ASPECT_RATIO": "aspect ratio",
    "DEPTH": "depth",
    "DEVICE_BATCH_SIZE": "batch size",
    "EMBEDDING_LR": "emb lr",
    "FINAL_LR_FRAC": "final lr",
    "HEAD_DIM": "head dim",
    "MATRIX_LR": "matrix lr",
    "SCALAR_LR": "scalar lr",
    "TOTAL_BATCH_SIZE": "total batch",
    "UNEMBEDDING_LR": "unemb lr",
    "WARMDOWN_RATIO": "warmdown",
    "WARMUP_RATIO": "warmup",
    "WEIGHT_DECAY": "weight decay",
    "WINDOW_PATTERN": "window pattern",
}


def _format_val(v):
    """Format HP value concisely (3-4 significant digits, no sci notation)."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if abs(v) >= 0.001:
            # Trim trailing zeros: 0.040 -> 0.04, 0.225 -> 0.225
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int) and v >= 10000:
        return f"{v // 1000}K"
    return str(v)


def _describe_change(old_v, new_v):
    """Return a human-readable verb for the direction of change."""
    try:
        o, n = float(old_v), float(new_v)
        if n > o:
            return "increase"
        return "decrease"
    except (ValueError, TypeError):
        return "change"


def _config_diff(prev: dict, curr: dict) -> str:
    """Human-readable description: 'decrease weight decay (0.22→0.08)'."""
    best_diff = None
    best_rel = -1.0
    for hp in curr:
        if hp not in prev or prev[hp] != curr[hp]:
            name = HP_HUMAN.get(hp, hp.lower())
            if hp in prev:
                verb = _describe_change(prev[hp], curr[hp])
                diff_str = f"{verb} {name} ({_format_val(prev[hp])}→{_format_val(curr[hp])})"
                try:
                    old_v, new_v = float(prev[hp]), float(curr[hp])
                    denom = max(abs(old_v), 1e-12)
                    rel = abs(new_v - old_v) / denom
                except (ValueError, TypeError):
                    rel = float("inf")
            else:
                diff_str = f"set {name}={_format_val(curr[hp])}"
                rel = float("inf")
            if rel > best_rel:
                best_rel = rel
                best_diff = diff_str
    if best_diff is None:
        return "baseline"
    return best_diff


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

    # Annotate incumbents with adjustText for automatic decluttering
    texts = []
    prev_config = None
    for idx, (trial_i, val, config) in enumerate(incumbents):
        if prev_config is None:
            label = "baseline"
        else:
            label = _config_diff(prev_config, config)
        prev_config = config
        texts.append(ax.text(trial_i, val, label, fontsize=8, color="#1a7a3a", alpha=0.9))

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

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#1a7a3a", alpha=0.4, lw=0.5),
                    force_points=(0.5, 0.8), force_text=(0.5, 0.8), expand=(1.2, 1.4))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_progress(results_dir: Path, assets_dir: Path):
    """Generate per-backend Karpathy-style progress plots."""
    bench_dir = results_dir / "exp2_benchmark"

    backends_to_plot = [
        ("optuna", "TPE", "#2196F3"),
        ("llm_greedy", "Karpathy Agent (14 HPs) [0.8B]", "#FF9800"),
        ("llm_greedy_Qwen3_5_27B_nothink", "Karpathy Agent (14 HPs) [27B]", "#FF9800"),
        ("llambo_original", "LLAMBO (Paper) [0.8B]", "#00BCD4"),
        ("llambo_original_Qwen3_5_27B_nothink", "LLAMBO (Paper) [27B]", "#00BCD4"),
        ("llambo", "LLAMBO (Optuna) [0.8B]", "#9C27B0"),
        ("llambo_Qwen3_5_27B_nothink", "LLAMBO (Optuna) [27B]", "#9C27B0"),
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

    # Build labels for adjustText
    texts = []
    prev_config = None
    for idx, (trial_i, val, config) in enumerate(incumbents):
        if trial_i in desc_map:
            label = desc_map[trial_i]
        elif prev_config is None:
            label = "baseline"
        else:
            label = _config_diff(prev_config, config)
        prev_config = config
        texts.append(ax.text(trial_i, val, label, fontsize=5.5, color="#1a7a3a", alpha=0.9))

    n_fail = sum(1 for t in trials if not t.get("success", False))
    ax.set_title(f"{display_name}\n{len(trials)} trials, {len(incumbents)} impr., {n_fail} fail",
                 fontsize=9)
    ax.set_xlim(0, len(trials))
    ax.set_ylim(0.975, 1.012)
    ax.grid(True, alpha=0.2)

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#1a7a3a", alpha=0.4, lw=0.5),
                    force_points=(0.5, 0.8), force_text=(0.5, 0.8), expand=(1.2, 1.4))


def plot_progress_combined(results_dir: Path, output_path: Path):
    """Combined incumbent traces plot: 2x2 grid, 27B backends."""
    bench_dir = results_dir / "exp2_benchmark"

    # Load cached LLM descriptions if available
    desc_path = Path(__file__).parent.parent / "assets" / "incumbent_descriptions.json"
    all_descriptions = {}
    if desc_path.exists():
        all_descriptions = json.loads(desc_path.read_text())

    backends = [
        ("optuna", "TPE", "#2196F3"),
        ("cma_es", "CMA-ES", "#00796B"),
        ("centaur_Qwen3_5_27B", "Centaur [27B]", "#D32F2F"),
        ("llambo_original_Qwen3_5_27B_nothink", "LLAMBO (Paper) [27B]", "#00BCD4"),
        ("llm_greedy_Qwen3_5_27B_nothink", "Karpathy Agent (14 HPs) [27B]", "#FF9800"),
        ("karpathy_agent_Qwen3_5_27B", "Karpathy Agent (Code) [27B]", "#795548"),
        ("random", "Random", "#607D8B"),
    ]

    n_backends = len(backends)
    ncols = 4
    nrows = (n_backends + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
    axes_flat = axes.flatten()

    for idx, (backend, name, color) in enumerate(backends):
        descs = all_descriptions.get(backend)
        plot_progress_subplot(axes_flat[idx], bench_dir, backend, name, color, descriptions=descs)

    # Hide unused subplots
    for idx in range(n_backends, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
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
