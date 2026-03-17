"""Tests for training-time budget enforcement."""

import json
from unittest.mock import MagicMock, patch

import pytest
from ConfigSpace import ConfigurationSpace, Float, Integer

from autoresearch_automl.backends.random_backend import RandomBackend
from autoresearch_automl.core.runner import RunConfig, Runner
from autoresearch_automl.tracking.results_db import ResultsDB, TrialRecord


@pytest.fixture
def simple_space():
    cs = ConfigurationSpace(seed=42)
    cs.add(Integer("depth", bounds=(4, 32), default=12))
    cs.add(Float("lr", bounds=(1e-5, 1.0), default=3e-4, log=True))
    cs.add(Float("wd", bounds=(0.0, 0.5), default=0.1))
    return cs


# --- ResultsDB: total_train_time in resume_state ---


def test_resume_state_includes_total_train_time(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    for i in range(5):
        db.add(TrialRecord(
            trial_id=i, backend="random", generation=0,
            config={"depth": 12}, budget=300.0,
            val_bpb=2.0, success=True,
            wall_time_seconds=100.0,
        ))
    state = db.resume_state()
    assert state["total_train_time"] == pytest.approx(500.0)


def test_resume_state_total_train_time_handles_none(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    db.add(TrialRecord(
        trial_id=0, backend="random", generation=0,
        config={"depth": 12}, budget=300.0,
        val_bpb=2.0, success=True,
        wall_time_seconds=200.0,
    ))
    db.add(TrialRecord(
        trial_id=1, backend="random", generation=0,
        config={"depth": 12}, budget=300.0,
        val_bpb=None, success=False,
        wall_time_seconds=None,  # failed trial, no time
    ))
    state = db.resume_state()
    assert state["total_train_time"] == pytest.approx(200.0)


def test_resume_state_empty_total_train_time(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    state = db.resume_state()
    assert state["total_train_time"] == 0.0


# --- Runner: time budget enforcement ---


def _make_mock_outcome(wall_time_seconds=100.0, val_bpb=1.5, success=True):
    """Create a mock ExperimentOutcome."""
    outcome = MagicMock()
    outcome.result.wall_time_seconds = wall_time_seconds
    outcome.result.val_bpb = val_bpb
    outcome.result.success = success
    outcome.result.train_loss = 0.5
    outcome.result.peak_memory_gb = 10.0
    outcome.result.tokens_per_second = 1000.0
    outcome.result.error = None
    outcome.result.extra_metrics = {}
    outcome.objectives = {"val_bpb": val_bpb}
    return outcome


@patch("autoresearch_automl.core.runner.ExperimentRunner")
@patch("autoresearch_automl.core.runner.SearchSpaceBuilder")
def test_time_budget_stops_loop(mock_space_builder, mock_exp_runner_cls, tmp_path, simple_space):
    """Verify runner stops when cumulative training time exceeds budget."""
    mock_space_builder.return_value.build_configspace.return_value = simple_space

    # Each trial takes 100s, budget is 350s -> should run baseline + 2 trials then stop before 4th
    mock_runner = MagicMock()
    mock_runner.run.return_value = _make_mock_outcome(wall_time_seconds=100.0)
    mock_exp_runner_cls.return_value = mock_runner

    backend = RandomBackend()
    config = RunConfig(
        train_py_path=tmp_path / "fake_train.py",
        backend=backend,
        n_trials=9999,
        budget_max=300.0,
        seed=0,
        results_dir=tmp_path / "results",
        resume=False,
        time_budget=350.0,  # 350s budget
    )

    runner = Runner(config)
    summary = runner.run()

    # Baseline (100s) + trial 1 (100s) + trial 2 (100s) = 300s < 350s
    # Trial 3 would start (300 < 350), after trial 3: 400s >= 350s
    # Trial 4 check: 400s >= 350s -> stop
    # So we expect baseline + 3 trials = 4 calls total
    assert mock_runner.run.call_count == 4  # baseline + 3 main loop trials
    assert runner.cumulative_train_time == pytest.approx(400.0)


@patch("autoresearch_automl.core.runner.ExperimentRunner")
@patch("autoresearch_automl.core.runner.SearchSpaceBuilder")
def test_time_budget_zero_disables(mock_space_builder, mock_exp_runner_cls, tmp_path, simple_space):
    """Verify time_budget=0 disables budget enforcement."""
    mock_space_builder.return_value.build_configspace.return_value = simple_space

    mock_runner = MagicMock()
    mock_runner.run.return_value = _make_mock_outcome(wall_time_seconds=100.0)
    mock_exp_runner_cls.return_value = mock_runner

    backend = RandomBackend()
    config = RunConfig(
        train_py_path=tmp_path / "fake_train.py",
        backend=backend,
        n_trials=5,  # should run all 5 trials + baseline
        budget_max=300.0,
        seed=0,
        results_dir=tmp_path / "results",
        resume=False,
        time_budget=0,  # disabled
    )

    runner = Runner(config)
    summary = runner.run()

    # baseline (trial_id=0) + main loop (trial_id 1..4) = 5 calls
    assert mock_runner.run.call_count == 5
    assert runner.cumulative_train_time == pytest.approx(500.0)


@patch("autoresearch_automl.core.runner.ExperimentRunner")
@patch("autoresearch_automl.core.runner.SearchSpaceBuilder")
def test_time_budget_with_resume(mock_space_builder, mock_exp_runner_cls, tmp_path, simple_space):
    """Verify resumed runs account for prior training time."""
    mock_space_builder.return_value.build_configspace.return_value = simple_space

    # Pre-populate 3 trials with 100s each (300s total)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    db = ResultsDB(results_dir / "trials.jsonl")
    for i in range(3):
        db.add(TrialRecord(
            trial_id=i, backend="random", generation=0,
            config={"depth": 12, "lr": 0.001, "wd": 0.1}, budget=300.0,
            val_bpb=2.0 - i * 0.1, success=True,
            wall_time_seconds=100.0,
        ))

    # Budget is 350s, already used 300s -> should only run 1 more trial before stopping
    mock_runner = MagicMock()
    mock_runner.run.return_value = _make_mock_outcome(wall_time_seconds=100.0)
    mock_exp_runner_cls.return_value = mock_runner

    backend = RandomBackend()
    config = RunConfig(
        train_py_path=tmp_path / "fake_train.py",
        backend=backend,
        n_trials=9999,
        budget_max=300.0,
        seed=0,
        results_dir=results_dir,
        resume=True,
        time_budget=350.0,
    )

    runner = Runner(config)
    summary = runner.run()

    # Resumed with 300s, budget=350s
    # Trial 3: 300 < 350 -> runs, cumulative becomes 400s
    # Trial 4: 400 >= 350 -> stops
    assert mock_runner.run.call_count == 1
    assert runner.cumulative_train_time == pytest.approx(400.0)


@patch("autoresearch_automl.core.runner.ExperimentRunner")
@patch("autoresearch_automl.core.runner.SearchSpaceBuilder")
def test_time_budget_already_exhausted_on_resume(mock_space_builder, mock_exp_runner_cls, tmp_path, simple_space):
    """Verify runner stops immediately if budget already exhausted from prior run."""
    mock_space_builder.return_value.build_configspace.return_value = simple_space

    # Pre-populate trials that already exceed budget
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    db = ResultsDB(results_dir / "trials.jsonl")
    for i in range(5):
        db.add(TrialRecord(
            trial_id=i, backend="random", generation=0,
            config={"depth": 12, "lr": 0.001, "wd": 0.1}, budget=300.0,
            val_bpb=2.0, success=True,
            wall_time_seconds=100.0,
        ))

    mock_runner = MagicMock()
    mock_exp_runner_cls.return_value = mock_runner

    backend = RandomBackend()
    config = RunConfig(
        train_py_path=tmp_path / "fake_train.py",
        backend=backend,
        n_trials=9999,
        budget_max=300.0,
        seed=0,
        results_dir=results_dir,
        resume=True,
        time_budget=350.0,  # already at 500s > 350s
    )

    runner = Runner(config)
    summary = runner.run()

    # Should not run any new trials
    assert mock_runner.run.call_count == 0
    assert runner.cumulative_train_time == pytest.approx(500.0)
