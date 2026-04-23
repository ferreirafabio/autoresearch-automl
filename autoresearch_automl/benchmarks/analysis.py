"""Statistical comparison of HPO backends: bootstrap CI, rank tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between methods."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    p_value: float | None = None
    significant: bool = False


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    return (np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100))


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank test for paired samples."""
    from scipy.stats import wilcoxon
    _, p_value = wilcoxon(a, b, alternative="two-sided")
    return p_value


def compare_methods(
    df: pd.DataFrame,
    method_col: str = "backend",
    metric: str = "val_bpb",
    seed_col: str = "seed",
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Compare all pairs of methods across seeds.

    For each method, takes the best metric value per seed, then compares
    distributions across seeds using bootstrap CI and Wilcoxon test.
    """
    # Best metric per (method, seed)
    best_per_seed = (
        df[df["success"]]
        .groupby([method_col, seed_col])[metric]
        .min()
        .reset_index()
    )

    methods = best_per_seed[method_col].unique()
    results = []

    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            vals_a = best_per_seed[best_per_seed[method_col] == ma][metric].values
            vals_b = best_per_seed[best_per_seed[method_col] == mb][metric].values

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            ci_a = bootstrap_ci(vals_a, n_bootstrap=n_bootstrap)
            ci_b = bootstrap_ci(vals_b, n_bootstrap=n_bootstrap)

            p_value = None
            if len(vals_a) == len(vals_b) and len(vals_a) >= 5:
                try:
                    p_value = wilcoxon_test(vals_a, vals_b)
                except Exception:
                    pass

            results.append(ComparisonResult(
                method_a=ma,
                method_b=mb,
                metric=metric,
                mean_a=float(np.mean(vals_a)),
                mean_b=float(np.mean(vals_b)),
                ci_a=ci_a,
                ci_b=ci_b,
                p_value=p_value,
                significant=p_value is not None and p_value < alpha,
            ))

    return results


def rank_methods(
    df: pd.DataFrame,
    method_col: str = "backend",
    metric: str = "val_bpb",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Rank methods by best metric across seeds."""
    best_per_seed = (
        df[df["success"]]
        .groupby([method_col, seed_col])[metric]
        .min()
        .reset_index()
    )

    summary = (
        best_per_seed
        .groupby(method_col)[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean")
    )

    return summary
