"""Integration tests against the real Karpathy autoresearch train.py."""

from pathlib import Path

import pytest

from autoresearch_automl.core.config_injector import ConfigInjector
from autoresearch_automl.core.search_space import SearchSpaceBuilder, configspace_to_optuna

TRAIN_PY = Path(__file__).parent.parent / "autoresearch" / "train.py"

pytestmark = pytest.mark.skipif(
    not TRAIN_PY.exists(),
    reason="autoresearch submodule not available",
)


def test_extract_real_hps():
    builder = SearchSpaceBuilder(TRAIN_PY)
    hps = builder.extract_hps()
    names = {hp.name for hp in hps}

    # Core HPs that must be found
    assert "DEPTH" in names
    assert "ASPECT_RATIO" in names
    assert "HEAD_DIM" in names
    assert "TOTAL_BATCH_SIZE" in names
    assert "MATRIX_LR" in names
    assert "EMBEDDING_LR" in names
    assert "WEIGHT_DECAY" in names
    assert "WARMUP_RATIO" in names
    assert "WARMDOWN_RATIO" in names
    assert "WINDOW_PATTERN" in names
    assert "DEVICE_BATCH_SIZE" in names

    # Constants from prepare.py must NOT appear
    assert "MAX_SEQ_LEN" not in names
    assert "TIME_BUDGET" not in names
    assert "EVAL_TOKENS" not in names


def test_build_real_configspace():
    builder = SearchSpaceBuilder(TRAIN_PY)
    cs = builder.build_configspace()

    assert len(cs) >= 8  # at least the core numeric/categorical HPs
    assert "DEPTH" in cs
    assert "MATRIX_LR" in cs
    assert "WEIGHT_DECAY" in cs


def test_optuna_mapping_from_real():
    builder = SearchSpaceBuilder(TRAIN_PY)
    cs = builder.build_configspace(include=["DEPTH", "MATRIX_LR", "HEAD_DIM", "WEIGHT_DECAY"])
    mapping = configspace_to_optuna(cs)

    assert mapping["DEPTH"]["method"] == "suggest_int"
    assert mapping["MATRIX_LR"]["method"] == "suggest_float"
    assert mapping["MATRIX_LR"]["kwargs"]["log"] is True
    assert mapping["HEAD_DIM"]["method"] == "suggest_categorical"
    assert mapping["WEIGHT_DECAY"]["method"] == "suggest_float"


def test_inject_real_train_py():
    injector = ConfigInjector(TRAIN_PY)
    result = injector.inject({"DEPTH": 12, "MATRIX_LR": 0.02, "WEIGHT_DECAY": 0.15})

    assert "DEPTH = 12" in result
    assert "0.02" in result
    assert "0.15" in result
    # Original values should be gone
    assert "DEPTH = 8" not in result
