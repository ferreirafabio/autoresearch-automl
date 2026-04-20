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
    """Color encodes method family; linestyle encodes model version (4.6 solid, 4.7 dashed)."""
    return {
        # Classical reference — distinct color + dotted linestyle
        "optuna": {
            "label": "TPE (classical)",
            "color": "#E69F00",
            "linestyle": ":",
            "path": str(RESULTS_BASE / "optuna"),
        },
        # Centaur (hybrid) — red family
        "centaur_claude_opus_4_6": {
            "label": "Centaur [Opus 4.6]",
            "color": "#C62828",
            "linestyle": "-",
            "path": str(OPUS46_BASE / "centaur_claude_opus_4_6"),
        },
        "centaur_claude_opus_4_7": {
            "label": "Centaur [Opus 4.7]",
            "color": "#C62828",
            "linestyle": "--",
            "path": str(OPUS47_BASE / "centaur_claude_opus_4_7"),
        },
        # KA HPs (pure LLM, fixed search space) — teal family
        "karpathy_agent_hps_claude_opus_4_6": {
            "label": "KA HPs [Opus 4.6]",
            "color": "#00897B",
            "linestyle": "-",
            "path": str(OPUS46_BASE / "karpathy_agent_hps_claude_opus_4_6"),
        },
        "karpathy_agent_hps_claude_opus_4_7": {
            "label": "KA HPs [Opus 4.7]",
            "color": "#00897B",
            "linestyle": "--",
            "path": str(OPUS47_BASE / "karpathy_agent_hps_claude_opus_4_7"),
        },
        # KA Code (pure LLM, free code editing) — blue family
        "karpathy_agent_claude_opus_4_6": {
            "label": "KA Code [Opus 4.6]",
            "color": "#1565C0",
            "linestyle": "-",
            "path": str(OPUS46_BASE / "karpathy_agent_claude_opus_4_6"),
        },
        "karpathy_agent_claude_opus_4_7": {
            "label": "KA Code [Opus 4.7]",
            "color": "#1565C0",
            "linestyle": "--",
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
