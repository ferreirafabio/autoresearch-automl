"""Convergence plots, Pareto fronts, and HP importance visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Generate plots for experiment analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convergence_plot(
        self,
        df: pd.DataFrame,
        objective: str = "val_bpb",
        group_by: str = "backend",
        title: str = "Convergence Plot",
    ) -> Path:
        """Plot incumbent val_bpb over trials, grouped by backend."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for group_name, group_df in df.groupby(group_by):
            group_df = group_df.sort_values("trial_id")
            incumbent = group_df[objective].cummin()
            ax.plot(group_df["trial_id"], incumbent, label=group_name, marker=".", markersize=3)

        ax.set_xlabel("Trial")
        ax.set_ylabel(objective)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "convergence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved convergence plot to %s", path)
        return path

    def anytime_performance(
        self,
        df: pd.DataFrame,
        objective: str = "val_bpb",
        group_by: str = "backend",
    ) -> Path:
        """Plot incumbent performance over wall-clock time."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for group_name, group_df in df.groupby(group_by):
            group_df = group_df.sort_values("timestamp")
            cumtime = group_df["wall_time_seconds"].cumsum() / 3600  # hours
            incumbent = group_df[objective].cummin()
            ax.plot(cumtime, incumbent, label=group_name)

        ax.set_xlabel("Wall-clock time (hours)")
        ax.set_ylabel(objective)
        ax.set_title("Anytime Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "anytime_performance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def pareto_front(
        self,
        df: pd.DataFrame,
        obj1: str = "val_bpb",
        obj2: str = "peak_memory_gb",
    ) -> Path:
        """Plot Pareto front for multi-objective optimization."""
        fig, ax = plt.subplots(figsize=(8, 6))

        valid = df.dropna(subset=[obj1, obj2])
        ax.scatter(valid[obj1], valid[obj2], alpha=0.5, s=20)

        # Compute and highlight Pareto front
        pareto = self._compute_pareto(valid[[obj1, obj2]].values)
        if len(pareto) > 0:
            pareto_sorted = pareto[pareto[:, 0].argsort()]
            ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], "r-o", label="Pareto front", markersize=6)

        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.set_title(f"Pareto Front: {obj1} vs {obj2}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "pareto_front.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def hp_importance_plot(self, importance: dict[str, float]) -> Path:
        """Bar plot of hyperparameter importances."""
        fig, ax = plt.subplots(figsize=(10, 6))

        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]

        ax.barh(names, values)
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        ax.invert_yaxis()

        path = self.output_dir / "hp_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _compute_pareto(points):
        """Compute Pareto front (minimize both objectives)."""
        import numpy as np
        is_pareto = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            if is_pareto[i]:
                is_pareto[is_pareto] = ~(
                    (points[is_pareto] <= p).all(axis=1) & (points[is_pareto] < p).any(axis=1)
                )
                is_pareto[i] = True
        return points[is_pareto]
