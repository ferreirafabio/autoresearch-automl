"""Tests for the Karpathy agent backend."""

import ast
from unittest.mock import MagicMock, patch

import pytest
from ConfigSpace import ConfigurationSpace

from autoresearch_automl.backends.karpathy_agent_backend import KarpathyAgentBackend


SAMPLE_SOURCE = """\
import os
import torch
from prepare import *

DEPTH = 8
ASPECT_RATIO = 64
HEAD_DIM = 128
DEVICE_BATCH_SIZE = 64
TOTAL_BATCH_SIZE = 524288
EMBEDDING_LR = 0.6
MATRIX_LR = 0.04
WINDOW_PATTERN = "SSSL"

# ... rest of training code ...
def main():
    pass
"""

MODIFIED_SOURCE = """\
import os
import torch
from prepare import *

DEPTH = 12
ASPECT_RATIO = 64
HEAD_DIM = 128
DEVICE_BATCH_SIZE = 64
TOTAL_BATCH_SIZE = 524288
EMBEDDING_LR = 0.8
MATRIX_LR = 0.04
WINDOW_PATTERN = "SSLL"

# ... rest of training code ...
def main():
    pass
"""


def _make_backend(source=SAMPLE_SOURCE):
    backend = KarpathyAgentBackend(llm_client=MagicMock(), model="test-model")
    space = ConfigurationSpace(seed=0)
    backend.configure(space=space, objectives=["val_bpb"], budget_range=(60, 300), seed=0)
    backend.set_source(source)
    return backend


class TestParseResponse:
    def test_parses_description_and_code(self):
        backend = _make_backend()
        text = f"DESCRIPTION: Increase depth to 12\n```python\n{MODIFIED_SOURCE}\n```"
        desc, source = backend._parse_response(text)
        assert desc == "Increase depth to 12"
        assert "DEPTH = 12" in source

    def test_fallback_no_code_block(self):
        backend = _make_backend()
        text = "DESCRIPTION: some change\nno code block here"
        desc, source = backend._parse_response(text)
        assert desc == "some change"
        # Falls back to current best source
        assert source == SAMPLE_SOURCE

    def test_handles_generic_code_block(self):
        backend = _make_backend()
        text = f"DESCRIPTION: test\n```\n{MODIFIED_SOURCE}\n```"
        desc, source = backend._parse_response(text)
        assert "DEPTH = 12" in source


class TestExtractConfig:
    def test_extracts_uppercase_constants(self):
        backend = _make_backend()
        config = backend._extract_config(SAMPLE_SOURCE)
        assert config["DEPTH"] == 8
        assert config["ASPECT_RATIO"] == 64
        assert config["HEAD_DIM"] == 128
        assert config["EMBEDDING_LR"] == 0.6
        assert config["WINDOW_PATTERN"] == "SSSL"

    def test_handles_invalid_python(self):
        backend = _make_backend()
        config = backend._extract_config("this is not python{{{")
        assert config == {}


class TestHillClimbing:
    def test_keeps_improvement(self):
        backend = _make_backend()
        # Seed baseline
        backend.seed_trial({"DEPTH": 8}, 300, {"val_bpb": 1.0})

        # Simulate suggest returning modified source
        backend._pending_source = MODIFIED_SOURCE
        backend._pending_description = "Increase depth"

        # Tell with improvement
        backend.tell({"DEPTH": 12}, 300, {"val_bpb": 0.95})
        assert backend._best_val_bpb == 0.95
        assert "DEPTH = 12" in backend._current_best_source

    def test_reverts_regression(self):
        backend = _make_backend()
        backend.seed_trial({"DEPTH": 8}, 300, {"val_bpb": 1.0})

        backend._pending_source = MODIFIED_SOURCE
        backend._pending_description = "Bad change"

        backend.tell({"DEPTH": 12}, 300, {"val_bpb": 1.1})
        assert backend._best_val_bpb == 1.0
        # Should still have original source
        assert "DEPTH = 8" in backend._current_best_source

    def test_reverts_crash(self):
        backend = _make_backend()
        backend.seed_trial({"DEPTH": 8}, 300, {"val_bpb": 1.0})

        backend._pending_source = MODIFIED_SOURCE
        backend._pending_description = "OOM change"

        backend.tell({"DEPTH": 12}, 300, {"_error": "GPU out-of-memory"})
        assert backend._best_val_bpb == 1.0
        assert "DEPTH = 8" in backend._current_best_source


class TestSuggestWithMockLLM:
    def test_suggest_calls_llm_and_returns_config(self):
        backend = _make_backend()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = (
            f"DESCRIPTION: Increase depth\n```python\n{MODIFIED_SOURCE}\n```"
        )
        llm_response.choices[0].message.model_extra = {}
        backend._llm_client.chat.completions.create.return_value = llm_response

        config, budget = backend.suggest()
        assert config["DEPTH"] == 12
        assert budget == 300.0
        assert backend.get_source_override() is not None
        assert "DEPTH = 12" in backend.get_source_override()

    def test_suggest_validates_python_syntax(self):
        backend = _make_backend()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = (
            "DESCRIPTION: Bad syntax\n```python\ndef broken(:\n```"
        )
        llm_response.choices[0].message.model_extra = {}
        backend._llm_client.chat.completions.create.return_value = llm_response

        config, budget = backend.suggest()
        # Should fall back to current best source
        source = backend.get_source_override()
        assert source == SAMPLE_SOURCE


class TestFormatHistory:
    def test_formats_kept_and_reverted(self):
        backend = _make_backend()
        backend._results = [
            {"description": "baseline", "val_bpb": 1.0, "success": True, "error": None, "kept": True},
            {"description": "bigger model", "val_bpb": 0.95, "success": True, "error": None, "kept": True},
            {"description": "too big", "val_bpb": None, "success": False, "error": "OOM", "kept": False},
        ]
        history = backend._format_history()
        assert "[KEPT] baseline" in history
        assert "[KEPT] bigger model" in history
        assert "[REVERTED] too big" in history
        assert "CRASHED (OOM)" in history
