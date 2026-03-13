"""Tests for HPO backends (suggest/tell interface)."""

import pytest
from ConfigSpace import ConfigurationSpace, Float, Integer

from autoresearch_automl.backends.random_backend import RandomBackend


@pytest.fixture
def simple_space():
    cs = ConfigurationSpace(seed=42)
    cs.add(Integer("depth", bounds=(4, 32), default=12))
    cs.add(Float("lr", bounds=(1e-5, 1.0), default=3e-4, log=True))
    cs.add(Float("wd", bounds=(0.0, 0.5), default=0.1))
    return cs


def test_random_backend_suggest(simple_space):
    backend = RandomBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    config, budget = backend.suggest()

    assert isinstance(config, dict)
    assert "depth" in config
    assert "lr" in config
    assert "wd" in config
    assert 4 <= config["depth"] <= 32
    assert 1e-5 <= config["lr"] <= 1.0
    assert budget == 300.0


def test_random_backend_tell(simple_space):
    backend = RandomBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    config, budget = backend.suggest()
    backend.tell(config, budget, {"val_bpb": 1.5})

    incumbents = backend.incumbents()
    assert len(incumbents) == 1
    assert incumbents[0] == config


def test_random_backend_multiple_trials(simple_space):
    backend = RandomBackend()
    backend.configure(simple_space, objectives=["val_bpb"], seed=42)

    for i in range(10):
        config, budget = backend.suggest()
        backend.tell(config, budget, {"val_bpb": 2.0 - i * 0.1})

    incumbents = backend.incumbents()
    assert len(incumbents) == 1


def test_random_backend_budget_range(simple_space):
    backend = RandomBackend()
    backend.configure(simple_space, objectives=["val_bpb"], budget_range=(60, 180), seed=42)

    _, budget = backend.suggest()
    assert budget == 180.0
