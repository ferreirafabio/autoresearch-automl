"""Tests for results database."""

import pytest

from autoresearch_automl.tracking.results_db import ResultsDB, TrialRecord


@pytest.fixture
def db(tmp_path):
    return ResultsDB(tmp_path / "trials.jsonl")


def test_add_and_load(db):
    record = TrialRecord(
        trial_id=0,
        backend="random",
        generation=0,
        config={"depth": 12, "lr": 0.001},
        budget=300.0,
        val_bpb=1.5,
        success=True,
    )
    db.add(record)

    records = db.load_all()
    assert len(records) == 1
    assert records[0].trial_id == 0
    assert records[0].val_bpb == 1.5


def test_best_trial(db):
    for i in range(5):
        db.add(TrialRecord(
            trial_id=i,
            backend="random",
            generation=0,
            config={"depth": 12 + i},
            budget=300.0,
            val_bpb=2.0 - i * 0.1,
            success=True,
        ))

    best = db.best_trial()
    assert best is not None
    assert best.trial_id == 4
    assert best.val_bpb == pytest.approx(1.6)


def test_to_dataframe(db):
    db.add(TrialRecord(
        trial_id=0, backend="optuna", generation=0,
        config={}, budget=300.0, val_bpb=1.5, success=True,
    ))
    db.add(TrialRecord(
        trial_id=1, backend="random", generation=0,
        config={}, budget=300.0, val_bpb=1.8, success=True,
    ))

    df = db.to_dataframe()
    assert len(df) == 2
    assert set(df["backend"]) == {"optuna", "random"}


def test_summary(db):
    db.add(TrialRecord(
        trial_id=0, backend="optuna", generation=0,
        config={}, budget=300.0, val_bpb=1.5, success=True,
    ))
    summary = db.summary()
    assert summary["total_trials"] == 1
    assert summary["successful_trials"] == 1
