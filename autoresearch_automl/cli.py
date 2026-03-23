"""CLI entry point for autoresearch-automl."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from autoresearch_automl.core.runner import RunConfig, Runner

logger = logging.getLogger(__name__)

BACKEND_REGISTRY = {
    "random": "autoresearch_automl.backends.random_backend:RandomBackend",
    "llm_greedy": "autoresearch_automl.backends.llm_greedy_backend:LLMGreedyBackend",
    "karpathy_agent": "autoresearch_automl.backends.karpathy_agent_backend:KarpathyAgentBackend",
    "optuna": "autoresearch_automl.backends.optuna_backend:OptunaBackend",
    "llambo": "autoresearch_automl.backends.llambo_backend:LLAMBOBackend",
    "llambo_original": "autoresearch_automl.backends.llambo_original_backend:LLAMBOOriginalBackend",
    "smac": "autoresearch_automl.backends.smac_backend:SMACBackend",
    "cma_es": "autoresearch_automl.backends.cma_es_backend:CmaESBackend",
    "centaur": "autoresearch_automl.backends.centaur_backend:CentaurBackend",
}


def _load_backend(name: str, **kwargs):
    """Dynamically load a backend class by name."""
    if name not in BACKEND_REGISTRY:
        raise click.BadParameter(f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")
    module_path, class_name = BACKEND_REGISTRY[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """autoresearch-automl: Integrating AutoML into Karpathy's Autoresearch."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--backend", "-b", default="optuna", help="HPO backend to use")
@click.option("--trials", "-n", default=100, help="Number of trials")
@click.option("--budget-min", default=60.0, help="Min budget in seconds")
@click.option("--budget-max", default=300.0, help="Max budget in seconds")
@click.option("--seed", "-s", default=0, help="Random seed")
@click.option("--train-py", default="autoresearch/train.py", help="Path to train.py")
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--hybrid-mode", type=click.Choice(["A", "B", "C"]), default=None, help="Hybrid collaboration mode")
@click.option("--config-file", type=click.Path(exists=True), default=None, help="YAML config file")
@click.option("--gpu", type=int, default=None, help="GPU device index")
@click.option("--storage", default=None, help="Optuna storage URL for parallel execution")
@click.option("--llm-model", default=None, help="LLM model name for LLM-based backends (e.g. Qwen/Qwen3.5-9B)")
@click.option("--resume/--no-resume", default=True, help="Auto-resume from existing trials.jsonl (default: on)")
@click.option("--time-budget", default=86400.0, help="Training-time budget in seconds (sum of wall_time_seconds). Default: 86400 (24h). 0 to disable.")
def run(
    backend: str,
    trials: int,
    budget_min: float,
    budget_max: float,
    seed: int,
    train_py: str,
    results_dir: str,
    hybrid_mode: str | None,
    config_file: str | None,
    gpu: int | None,
    storage: str | None,
    llm_model: str | None,
    resume: bool,
    time_budget: float,
):
    """Run HPO optimization loop."""
    # Load config file if provided
    if config_file:
        with open(config_file) as f:
            file_cfg = yaml.safe_load(f)
        backend = file_cfg.get("backend", backend)
        trials = file_cfg.get("trials", trials)
        budget_min = file_cfg.get("budget_min", budget_min)
        budget_max = file_cfg.get("budget_max", budget_max)
        seed = file_cfg.get("seed", seed)
        llm_model = file_cfg.get("llm_model", llm_model)

    # Build backend
    backend_kwargs = {}
    if backend == "optuna" and storage:
        backend_kwargs["storage"] = storage
    if llm_model and backend in ("llm_greedy", "karpathy_agent", "llambo", "llambo_original", "centaur"):
        backend_kwargs["model"] = llm_model
    if backend in ("llambo", "llambo_original", "llm_greedy", "karpathy_agent", "centaur"):
        backend_kwargs["log_dir"] = Path(results_dir)
    hpo_backend = _load_backend(backend, **backend_kwargs)

    train_py_path = Path(train_py)
    if not train_py_path.exists():
        raise click.BadParameter(f"train.py not found at {train_py_path}")

    run_config = RunConfig(
        train_py_path=train_py_path,
        backend=hpo_backend,
        n_trials=trials,
        budget_min=budget_min,
        budget_max=budget_max,
        seed=seed,
        results_dir=Path(results_dir),
        gpu_device=gpu,
        hybrid_mode=hybrid_mode,
        llm_model=llm_model,
        resume=resume,
        time_budget=time_budget,
    )

    runner = Runner(run_config)
    summary = runner.run()
    click.echo(f"\nRun complete! Best val_bpb: {summary['best_val_bpb']:.4f}")
    click.echo(f"Total time: {summary['total_time_hours']:.2f}h")
    click.echo(f"Training time: {summary['total_train_time_hours']:.2f}h")


@main.command()
@click.option("--method", required=True, help="Method/backend to benchmark")
@click.option("--scenario", required=True, help="Benchmark scenario name")
@click.option("--seed", "-s", default=0, help="Random seed")
@click.option("--train-py", default="autoresearch/train.py", help="Path to train.py")
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--llm-model", default=None, help="LLM model name for LLM-based backends (e.g. Qwen/Qwen3.5-9B)")
@click.option("--resume/--no-resume", default=True, help="Auto-resume from existing trials.jsonl (default: on)")
def benchmark(method: str, scenario: str, seed: int, train_py: str, results_dir: str, llm_model: str | None, resume: bool):
    """Run a benchmark scenario."""
    from autoresearch_automl.benchmarks.scenarios import get_scenario

    sc = get_scenario(scenario)
    backend_kwargs = {}
    if llm_model and method in ("llm_greedy", "karpathy_agent", "llambo", "llambo_original", "centaur"):
        backend_kwargs["model"] = llm_model
    if method in ("llambo", "llambo_original", "llm_greedy", "karpathy_agent", "centaur"):
        backend_kwargs["log_dir"] = Path(results_dir)
    hpo_backend = _load_backend(method, **backend_kwargs)

    train_py_path = Path(train_py)
    if not train_py_path.exists():
        raise click.BadParameter(f"train.py not found at {train_py_path}")

    run_config = RunConfig(
        train_py_path=train_py_path,
        backend=hpo_backend,
        n_trials=sc.n_trials,
        budget_min=sc.budget_min,
        budget_max=sc.budget_max,
        seed=seed,
        objectives=sc.objectives,
        results_dir=Path(results_dir) / sc.name / method / f"seed_{seed}",
        hybrid_mode=sc.hybrid_mode,
        structural_interval=sc.structural_interval,
        llm_model=llm_model,
        resume=resume,
    )

    runner = Runner(run_config)
    summary = runner.run()
    click.echo(f"\nBenchmark {sc.name}/{method}/seed={seed} complete!")
    click.echo(f"Best val_bpb: {summary['best_val_bpb']:.4f}")


@main.command()
@click.option("--results-dir", default="results", help="Results directory")
@click.option("--output-dir", default="plots", help="Plot output directory")
def analyze(results_dir: str, output_dir: str):
    """Analyze and visualize results across runs."""
    import pandas as pd

    from autoresearch_automl.benchmarks.analysis import compare_methods, rank_methods
    from autoresearch_automl.tracking.results_db import ResultsDB
    from autoresearch_automl.tracking.visualizer import Visualizer

    # Collect all results
    results_path = Path(results_dir)
    all_records = []
    for jsonl in results_path.rglob("trials.jsonl"):
        db = ResultsDB(jsonl)
        all_records.extend(db.load_all())

    if not all_records:
        click.echo("No results found.")
        return

    from dataclasses import asdict
    df = pd.DataFrame([asdict(r) for r in all_records])
    click.echo(f"Loaded {len(df)} trials from {df['backend'].nunique()} backends")

    # Rankings
    rankings = rank_methods(df)
    click.echo("\nMethod Rankings (by mean best val_bpb):")
    click.echo(rankings.to_string())

    # Statistical comparisons
    comparisons = compare_methods(df)
    for comp in comparisons:
        sig = "*" if comp.significant else ""
        click.echo(
            f"\n{comp.method_a} vs {comp.method_b}: "
            f"{comp.mean_a:.4f} vs {comp.mean_b:.4f} "
            f"(p={comp.p_value:.4f}){sig}" if comp.p_value else ""
        )

    # Plots
    viz = Visualizer(Path(output_dir))
    viz.convergence_plot(df)
    viz.anytime_performance(df)
    click.echo(f"\nPlots saved to {output_dir}/")


@main.command()
@click.option("--train-py", default="autoresearch/train.py", help="Path to train.py")
def extract_space(train_py: str):
    """Extract and display the hyperparameter search space from train.py."""
    from autoresearch_automl.core.search_space import SearchSpaceBuilder

    train_py_path = Path(train_py)
    if not train_py_path.exists():
        raise click.BadParameter(f"train.py not found at {train_py_path}")

    builder = SearchSpaceBuilder(train_py_path)
    hps = builder.extract_hps()

    click.echo(f"Found {len(hps)} hyperparameters in {train_py}:\n")
    for hp in hps:
        click.echo(f"  {hp.name} = {hp.value!r}  (line {hp.lineno})")

    cs = builder.build_configspace()
    click.echo(f"\nConfigurationSpace ({len(cs)} HPs):")
    click.echo(str(cs))


if __name__ == "__main__":
    main()
