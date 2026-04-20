"""Convergence plot comparing Opus 4.6 vs Opus 4.7 across Centaur / KA HPs / KA Code,
with TPE (best classical method) as reference.

Follows the same visual conventions as plot_convergence.py:
  - x-axis = cumulative training wall-time (hours), 24h budget
  - y-axis = best val_bpb so far (lower is better)
  - classical = dashed + circle, hybrid = dashdot + diamond, pure LLM = solid + star
  - 4.6 and 4.7 variants of the same method share hue/linestyle/marker but differ in shade
  - bands for top N methods
"""

from pathlib import Path

from plot_convergence import plot_convergence_walltime


RESULTS_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/results")
OPUS46_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/opus46_benchmark")
OPUS47_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/opus47_benchmark")


def build_backends() -> dict[str, dict]:
    return {
        # Classical reference
        "optuna": {
            "label": "TPE (classical)",
            "color": "#E69F00",
            "linestyle": "--",
            "marker": "o",
            "path": str(RESULTS_BASE / "optuna"),
        },
        # Centaur (hybrid): Opus 4.6 darker, Opus 4.7 lighter
        "centaur_claude_opus_4_6": {
            "label": "Centaur [Opus 4.6]",
            "color": "#8E0000",
            "linestyle": "-.",
            "marker": "D",
            "path": str(OPUS46_BASE / "centaur_claude_opus_4_6"),
        },
        "centaur_claude_opus_4_7": {
            "label": "Centaur [Opus 4.7]",
            "color": "#E57373",
            "linestyle": "-.",
            "marker": "D",
            "path": str(OPUS47_BASE / "centaur_claude_opus_4_7"),
        },
        # KA HPs (pure LLM, fixed search space)
        "karpathy_agent_hps_claude_opus_4_6": {
            "label": "KA HPs [Opus 4.6]",
            "color": "#00695C",
            "linestyle": "-",
            "marker": "*",
            "path": str(OPUS46_BASE / "karpathy_agent_hps_claude_opus_4_6"),
        },
        "karpathy_agent_hps_claude_opus_4_7": {
            "label": "KA HPs [Opus 4.7]",
            "color": "#4DB6AC",
            "linestyle": "-",
            "marker": "*",
            "path": str(OPUS47_BASE / "karpathy_agent_hps_claude_opus_4_7"),
        },
        # KA Code (pure LLM, free code editing)
        "karpathy_agent_claude_opus_4_6": {
            "label": "KA Code [Opus 4.6]",
            "color": "#1565C0",
            "linestyle": "-",
            "marker": "*",
            "path": str(OPUS46_BASE / "karpathy_agent_claude_opus_4_6"),
        },
        "karpathy_agent_claude_opus_4_7": {
            "label": "KA Code [Opus 4.7]",
            "color": "#64B5F6",
            "linestyle": "-",
            "marker": "*",
            "path": str(OPUS47_BASE / "karpathy_agent_claude_opus_4_7"),
        },
    }


def main():
    output_path = Path("/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/assets/exp3_opus46_vs_opus47.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_convergence_walltime(
        results_dir=RESULTS_BASE,
        backends=build_backends(),
        output_path=output_path,
        title="Opus 4.6 vs Opus 4.7 (Centaur, KA HPs, KA Code) + TPE reference",
        ylim=(0.973, 0.993),
        max_bands=7,
    )


if __name__ == "__main__":
    main()
