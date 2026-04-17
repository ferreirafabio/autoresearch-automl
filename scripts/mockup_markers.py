"""Generate two mockup variants of Fig 1 for marker style comparison.

Option A (current): Each method has a unique marker shape, no group pattern.
Option B (grouped): Markers reflect category - classical all circles, hybrid
diamond, pure LLM stars/plus. Distinguishes groups visually.
"""

from pathlib import Path

from scripts.plot_convergence import plot_convergence_walltime


BACKENDS_OPTION_A = {
    "optuna": {"label": "TPE", "color": "#2196F3", "linestyle": "--", "marker": "o"},
    "cma_es": {"label": "CMA-ES", "color": "#2E7D32", "linestyle": "--", "marker": "s"},
    "smac": {"label": "SMAC", "color": "#F57C00", "linestyle": "--", "marker": "^"},
    "random": {"label": "Random", "color": "#607D8B", "linestyle": "--", "marker": "v"},
    "centaur_Qwen3_5_27B": {"label": "Centaur (CMA-ES+LLM)", "color": "#E91E63", "linestyle": "-.", "marker": "D"},
    "karpathy_agent_Qwen3_5_27B": {"label": "Karpathy Agent (Code)", "color": "#4A148C", "linestyle": "-", "marker": "P"},
    "karpathy_agent_hps_Qwen3_5_27B": {"label": "Karpathy Agent (14 HPs)", "color": "#FFC107", "linestyle": "-", "marker": "X"},
    "llambo_original_Qwen3_5_27B": {"label": "LLAMBO (Paper)", "color": "#00BCD4", "linestyle": "-", "marker": "*"},
    "llambo_Qwen3_5_27B": {"label": "LLAMBO (Optuna)", "color": "#9C27B0", "linestyle": "-", "marker": "H"},
}

# Option B: classical all circles, hybrid diamond, pure LLM all stars
BACKENDS_OPTION_B = {
    "optuna": {"label": "TPE", "color": "#2196F3", "linestyle": "--", "marker": "o"},
    "cma_es": {"label": "CMA-ES", "color": "#2E7D32", "linestyle": "--", "marker": "o"},
    "smac": {"label": "SMAC", "color": "#F57C00", "linestyle": "--", "marker": "o"},
    "random": {"label": "Random", "color": "#607D8B", "linestyle": "--", "marker": "o"},
    "centaur_Qwen3_5_27B": {"label": "Centaur (CMA-ES+LLM)", "color": "#E91E63", "linestyle": "-.", "marker": "D"},
    "karpathy_agent_Qwen3_5_27B": {"label": "Karpathy Agent (Code)", "color": "#4A148C", "linestyle": "-", "marker": "*"},
    "karpathy_agent_hps_Qwen3_5_27B": {"label": "Karpathy Agent (14 HPs)", "color": "#FFC107", "linestyle": "-", "marker": "*"},
    "llambo_original_Qwen3_5_27B": {"label": "LLAMBO (Paper)", "color": "#00BCD4", "linestyle": "-", "marker": "*"},
    "llambo_Qwen3_5_27B": {"label": "LLAMBO (Optuna)", "color": "#9C27B0", "linestyle": "-", "marker": "*"},
}


if __name__ == "__main__":
    results_dir = Path("results")
    plot_convergence_walltime(
        results_dir, BACKENDS_OPTION_A,
        Path("assets/mockup_markers_option_a.png"),
        title="Option A: unique markers per method",
        ylim=(0.974, 0.993),
    )
    plot_convergence_walltime(
        results_dir, BACKENDS_OPTION_B,
        Path("assets/mockup_markers_option_b.png"),
        title="Option B: marker = category (classical=circle, hybrid=diamond, LLM=star)",
        ylim=(0.974, 0.993),
    )
