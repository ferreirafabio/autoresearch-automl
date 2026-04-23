"""Tests for resume/checkpoint functionality."""

import pytest
from ConfigSpace import ConfigurationSpace, Float, Integer

from autoresearch_automl.backends.random_backend import RandomBackend
from autoresearch_automl.backends.optuna_backend import OptunaBackend
from autoresearch_automl.backends.bohb_backend import BOHBBackend
from autoresearch_automl.tracking.results_db import ResultsDB, TrialRecord


@pytest.fixture
def simple_space():
    cs = ConfigurationSpace(seed=42)
    cs.add(Integer("depth", bounds=(4, 32), default=12))
    cs.add(Float("lr", bounds=(1e-5, 1.0), default=3e-4, log=True))
    cs.add(Float("wd", bounds=(0.0, 0.5), default=0.1))
    return cs


def _make_history(n=5):
    """Create synthetic trial history."""
    history = []
    for i in range(n):
        config = {"depth": 12 + i, "lr": 0.001 * (i + 1), "wd": 0.1}
        budget = 300.0
        results = {"val_bpb": 2.0 - i * 0.1}
        history.append((config, budget, results))
    return history


# --- ResultsDB resume tests ---


def test_resume_state_empty(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    state = db.resume_state()
    assert state["n_completed"] == 0
    assert state["next_trial_id"] == 0
    assert state["best_val_bpb"] == float("inf")


def test_resume_state_with_trials(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    for i in range(5):
        db.add(TrialRecord(
            trial_id=i, backend="random", generation=0,
            config={"depth": 12 + i}, budget=300.0,
            val_bpb=2.0 - i * 0.1, success=True,
        ))
    state = db.resume_state()
    assert state["n_completed"] == 5
    assert state["next_trial_id"] == 5
    assert state["best_val_bpb"] == pytest.approx(1.6)
    assert state["max_generation"] == 0


def test_resume_state_with_warmstart(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    # Warm-start trials have negative IDs
    db.add(TrialRecord(
        trial_id=-1, backend="warmstart", generation=0,
        config={"depth": 12}, budget=300.0, val_bpb=1.9, success=True,
    ))
    db.add(TrialRecord(
        trial_id=0, backend="random", generation=0,
        config={"depth": 16}, budget=300.0, val_bpb=1.8, success=True,
    ))
    state = db.resume_state()
    assert state["n_completed"] == 2
    assert state["next_trial_id"] == 1
    assert state["best_val_bpb"] == pytest.approx(1.8)


def test_replay_history_ordering(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    db.add(TrialRecord(
        trial_id=0, backend="random", generation=0,
        config={"depth": 12}, budget=300.0, val_bpb=2.0, success=True,
    ))
    db.add(TrialRecord(
        trial_id=-1, backend="warmstart", generation=0,
        config={"depth": 10}, budget=300.0, val_bpb=1.9, success=True,
    ))
    history = db.replay_history()
    # Warmstart (negative ID) should come first
    assert len(history) == 2
    assert history[0][0]["depth"] == 10  # warmstart config
    assert history[1][0]["depth"] == 12  # regular config


def test_resume_state_with_generations(tmp_path):
    db = ResultsDB(tmp_path / "trials.jsonl")
    for i in range(3):
        db.add(TrialRecord(
            trial_id=i, backend="optuna", generation=0,
            config={"depth": 12}, budget=300.0, val_bpb=2.0 - i * 0.1, success=True,
        ))
    db.add(TrialRecord(
        trial_id=3, backend="optuna", generation=1,
        config={"depth": 16}, budget=300.0, val_bpb=1.5, success=True,
    ))
    state = db.resume_state()
    assert state["max_generation"] == 1
    assert state["next_trial_id"] == 4


# --- Backend replay tests ---


def test_random_replay(simple_space):
    backend = RandomBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    history = _make_history(5)
    backend.replay(history)

    assert len(backend._results) == 5
    incumbents = backend.incumbents()
    assert len(incumbents) == 1
    # Best should be last trial (val_bpb=1.6)
    assert incumbents[0]["depth"] == 16


def test_optuna_replay(simple_space):
    backend = OptunaBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    history = _make_history(5)
    backend.replay(history)

    assert len(backend.study.trials) == 5
    assert backend.study.best_trial.values[0] == pytest.approx(1.6)


def test_optuna_replay_with_failed_trials(simple_space):
    backend = OptunaBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    history = _make_history(3)
    # Add a failed trial (no val_bpb)
    history.append(({"depth": 20, "lr": 0.01, "wd": 0.1}, 300.0, {}))

    backend.replay(history)
    assert len(backend.study.trials) == 4
    # Best should still be from the 3 successful trials
    assert backend.study.best_trial.values[0] == pytest.approx(1.8)


def test_bohb_replay(simple_space):
    backend = BOHBBackend()
    backend.configure(simple_space, objectives=["val_bpb"], budget_range=(60, 300), seed=42)

    history = _make_history(5)
    backend.replay(history)

    assert len(backend._results) == 5
    assert backend._budget_idx == 5


def test_optuna_replay_then_suggest(simple_space):
    """Verify that suggest works correctly after replay."""
    backend = OptunaBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    history = _make_history(3)
    backend.replay(history)

    # Should be able to suggest new configs after replay
    config, budget = backend.suggest()
    assert isinstance(config, dict)
    assert "depth" in config
    assert budget == 300.0
