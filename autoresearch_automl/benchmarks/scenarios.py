"""Predefined benchmark scenarios for comparing HPO backends."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkScenario:
    """A benchmark scenario definition."""
    name: str
    description: str
    n_trials: int
    budget_min: float
    budget_max: float
    objectives: list[str] = field(default_factory=lambda: ["val_bpb"])
    hybrid_mode: str | None = None
    structural_interval: int = 20
    include_hps: list[str] | None = None  # restrict to these HPs
    hp_overrides: dict = field(default_factory=dict)


SCENARIOS = {
    "numeric_only": BenchmarkScenario(
        name="numeric_only",
        description="Pure numeric HP tuning, fixed architecture",
        n_trials=100,
        budget_min=60.0,
        budget_max=300.0,
        include_hps=[
            "DEPTH", "WIDTH", "HEAD_DIM", "ASPECT_RATIO",
            "LEARNING_RATE", "MATRIX_LR", "WEIGHT_DECAY",
            "WARMUP_RATIO", "COOLDOWN_RATIO", "TOTAL_BATCH_SIZE",
        ],
    ),
    "mixed": BenchmarkScenario(
        name="mixed",
        description="Numeric + structural changes (Mode B)",
        n_trials=100,
        budget_min=60.0,
        budget_max=300.0,
        hybrid_mode="B",
        structural_interval=20,
    ),
    "budget_constrained": BenchmarkScenario(
        name="budget_constrained",
        description="Only 20 evaluations — tests sample efficiency",
        n_trials=20,
        budget_min=60.0,
        budget_max=300.0,
    ),
    "multi_objective": BenchmarkScenario(
        name="multi_objective",
        description="Optimize val_bpb AND peak memory simultaneously",
        n_trials=100,
        budget_min=60.0,
        budget_max=300.0,
        objectives=["val_bpb", "peak_memory_gb"],
    ),
    "transfer": BenchmarkScenario(
        name="transfer",
        description="Warm-start from LLM domain knowledge (Mode C)",
        n_trials=50,
        budget_min=60.0,
        budget_max=300.0,
        hybrid_mode="C",
    ),
    "small_3hp": BenchmarkScenario(
        name="small_3hp",
        description="Small 3-HP space for validation/smoke testing",
        n_trials=10,
        budget_min=60.0,
        budget_max=300.0,
        include_hps=["DEPTH", "LEARNING_RATE", "WEIGHT_DECAY"],
    ),
}


def get_scenario(name: str) -> BenchmarkScenario:
    """Get a benchmark scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
