"""Tests for AST-based config injection."""

from pathlib import Path
from textwrap import dedent

import pytest

from autoresearch_automl.core.config_injector import ConfigInjector


@pytest.fixture
def sample_train_py(tmp_path):
    code = dedent("""\
        import torch

        ASPECT_RATIO = 64
        DEPTH = 8
        MATRIX_LR = 0.04
        WEIGHT_DECAY = 0.2  # cautious weight decay for Muon
        TOTAL_BATCH_SIZE = 2**19

        model = build_model(DEPTH, ASPECT_RATIO)
    """)
    path = tmp_path / "train.py"
    path.write_text(code)
    return path


def test_inject_basic(sample_train_py):
    injector = ConfigInjector(sample_train_py)
    result = injector.inject({"DEPTH": 12, "MATRIX_LR": 0.02})

    assert "DEPTH = 12" in result
    assert "0.02" in result
    # ASPECT_RATIO should remain unchanged
    assert "ASPECT_RATIO = 64" in result


def test_inject_preserves_comments(sample_train_py):
    injector = ConfigInjector(sample_train_py)
    result = injector.inject({"WEIGHT_DECAY": 0.1})

    assert "0.1" in result
    assert "# cautious weight decay for Muon" in result


def test_inject_and_write(sample_train_py):
    injector = ConfigInjector(sample_train_py)
    injector.inject_and_write({"DEPTH": 12})

    content = sample_train_py.read_text()
    assert "DEPTH = 12" in content


def test_inject_unknown_hp(sample_train_py):
    injector = ConfigInjector(sample_train_py)
    result = injector.inject({"NONEXISTENT_HP": 42})

    # Should not crash, original content preserved
    assert "DEPTH = 8" in result


def test_inject_float_formatting(sample_train_py):
    injector = ConfigInjector(sample_train_py)
    result = injector.inject({"MATRIX_LR": 5.5e-5})

    # Should use scientific notation for small values
    assert "5.5" in result or "5.500000e-05" in result
