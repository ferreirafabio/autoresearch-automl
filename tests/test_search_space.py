"""Tests for search space extraction and ConfigSpace building."""

from pathlib import Path
from textwrap import dedent

import pytest
from ConfigSpace import Float, Integer, Categorical

from autoresearch_automl.core.search_space import SearchSpaceBuilder, configspace_to_optuna


@pytest.fixture
def sample_train_py(tmp_path):
    """Create a minimal train.py mimicking Karpathy's autoresearch."""
    code = dedent("""\
        import torch

        from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer

        # Hyperparameters
        ASPECT_RATIO = 64
        HEAD_DIM = 128
        WINDOW_PATTERN = "SSSL"
        TOTAL_BATCH_SIZE = 2**19
        EMBEDDING_LR = 0.6
        MATRIX_LR = 0.04
        WEIGHT_DECAY = 0.2
        WARMUP_RATIO = 0.0
        WARMDOWN_RATIO = 0.5
        DEPTH = 8
        DEVICE_BATCH_SIZE = 128

        # Training loop...
        model = build_model(DEPTH, ASPECT_RATIO)
    """)
    path = tmp_path / "train.py"
    path.write_text(code)
    return path


def test_extract_hps(sample_train_py):
    builder = SearchSpaceBuilder(sample_train_py)
    hps = builder.extract_hps()

    names = {hp.name for hp in hps}
    # Should find these HPs
    assert "DEPTH" in names
    assert "ASPECT_RATIO" in names
    assert "MATRIX_LR" in names
    assert "WEIGHT_DECAY" in names
    assert "TOTAL_BATCH_SIZE" in names
    assert "WINDOW_PATTERN" in names

    # Should NOT find these (excluded — imported from prepare.py)
    assert "MAX_SEQ_LEN" not in names
    assert "TIME_BUDGET" not in names


def test_extract_power_expression(sample_train_py):
    builder = SearchSpaceBuilder(sample_train_py)
    hps = builder.extract_hps()
    batch_hp = next(hp for hp in hps if hp.name == "TOTAL_BATCH_SIZE")
    assert batch_hp.value == 2**19


def test_build_configspace(sample_train_py):
    builder = SearchSpaceBuilder(sample_train_py)
    cs = builder.build_configspace()

    assert len(cs) > 0
    assert "DEPTH" in cs
    assert "MATRIX_LR" in cs


def test_build_configspace_with_include(sample_train_py):
    builder = SearchSpaceBuilder(sample_train_py)
    cs = builder.build_configspace(include=["DEPTH", "MATRIX_LR"])

    assert len(cs) == 2
    assert "DEPTH" in cs
    assert "MATRIX_LR" in cs
    assert "ASPECT_RATIO" not in cs


def test_configspace_to_optuna(sample_train_py):
    builder = SearchSpaceBuilder(sample_train_py)
    cs = builder.build_configspace(include=["DEPTH", "MATRIX_LR", "HEAD_DIM"])

    mapping = configspace_to_optuna(cs)

    assert "DEPTH" in mapping
    assert mapping["DEPTH"]["method"] == "suggest_int"
    assert "MATRIX_LR" in mapping
    assert mapping["MATRIX_LR"]["method"] == "suggest_float"
    assert mapping["MATRIX_LR"]["kwargs"]["log"] is True
