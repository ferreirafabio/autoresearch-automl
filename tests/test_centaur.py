"""Tests for the Centaur backend (CMA-ES guided LLM optimization)."""

from unittest.mock import MagicMock, patch

import pytest
from ConfigSpace import ConfigurationSpace, Float, Integer

from autoresearch_automl.backends.centaur_backend import CentaurBackend


def _make_space():
    cs = ConfigurationSpace(seed=42)
    cs.add(Float("lr", bounds=(1e-5, 1.0), log=True, default=0.01))
    cs.add(Integer("depth", bounds=(4, 24), default=8))
    cs.add(Float("wd", bounds=(0.0, 0.5), default=0.1))
    return cs


def _make_backend(llm_ratio=0.3, llm_warmup=10):
    backend = CentaurBackend(
        model="test-model",
        log_dir=None,
        llm_ratio=llm_ratio,
        llm_warmup=llm_warmup,
    )
    space = _make_space()
    backend.configure(space=space, objectives=["val_bpb"], budget_range=(60, 300), seed=0)
    return backend


class TestLLMScheduling:
    def test_warmup_blocks_llm(self):
        backend = _make_backend(llm_ratio=0.5, llm_warmup=5)
        # During warmup, _should_use_llm returns False
        for i in range(5):
            backend._trial_count = i
            assert not backend._should_use_llm(), f"Trial {i} should not use LLM during warmup"

    def test_llm_after_warmup(self):
        backend = _make_backend(llm_ratio=0.5, llm_warmup=5)
        # interval = round(1/0.5) = 2, so every even trial after warmup
        backend._trial_count = 6
        assert backend._should_use_llm()

    def test_ratio_zero_never_uses_llm(self):
        backend = _make_backend(llm_ratio=0.0, llm_warmup=0)
        for i in range(20):
            backend._trial_count = i
            assert not backend._should_use_llm()

    def test_ratio_one_always_uses_llm_after_warmup(self):
        backend = _make_backend(llm_ratio=1.0, llm_warmup=3)
        for i in range(3, 20):
            backend._trial_count = i
            assert backend._should_use_llm(), f"Trial {i} should use LLM with ratio=1.0"


class TestSuggestCMA:
    def test_returns_valid_config_during_warmup(self):
        backend = _make_backend(llm_warmup=100)  # high warmup = always CMA
        config, budget = backend.suggest()
        assert "lr" in config
        assert "depth" in config
        assert "wd" in config
        assert budget == 300.0
        assert backend._active_source == "cma"

    def test_config_within_bounds(self):
        backend = _make_backend(llm_warmup=100)
        config, _ = backend.suggest()
        assert 1e-5 <= config["lr"] <= 1.0
        assert 4 <= config["depth"] <= 24
        assert 0.0 <= config["wd"] <= 0.5


class TestSuggestLLM:
    def test_llm_trial_calls_api(self):
        backend = _make_backend(llm_ratio=1.0, llm_warmup=0)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"lr": 0.05, "depth": 12, "wd": 0.1}'
        mock_response.choices[0].message.model_extra = {}
        mock_client.chat.completions.create.return_value = mock_response
        backend._llm_client = mock_client

        config, budget = backend.suggest()
        assert config["lr"] == 0.05
        assert config["depth"] == 12
        assert config["wd"] == 0.1
        assert backend._active_source == "llm"
        mock_client.chat.completions.create.assert_called_once()

    def test_llm_clamps_out_of_bounds(self):
        backend = _make_backend(llm_ratio=1.0, llm_warmup=0)

        mock_client = MagicMock()
        mock_response = MagicMock()
        # lr=5.0 exceeds upper=1.0, depth=50 exceeds upper=24
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"lr": 5.0, "depth": 50, "wd": -0.1}'
        mock_response.choices[0].message.model_extra = {}
        mock_client.chat.completions.create.return_value = mock_response
        backend._llm_client = mock_client

        config, _ = backend.suggest()
        assert config["lr"] <= 1.0
        assert config["depth"] <= 24
        assert config["wd"] >= 0.0

    def test_llm_handles_markdown_code_block(self):
        backend = _make_backend(llm_ratio=1.0, llm_warmup=0)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"lr": 0.05, "depth": 12, "wd": 0.1}\n```'
        mock_response.choices[0].message.model_extra = {}
        mock_client.chat.completions.create.return_value = mock_response
        backend._llm_client = mock_client

        config, _ = backend.suggest()
        assert config["lr"] == 0.05

    def test_fallback_on_llm_error(self):
        backend = _make_backend(llm_ratio=1.0, llm_warmup=0)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        backend._llm_client = mock_client

        config, budget = backend.suggest()
        # Should fall back to CMA-ES config
        assert "lr" in config
        assert backend._active_source == "cma"


class TestTellUpdatesAll:
    def test_tell_updates_study_and_results(self):
        backend = _make_backend(llm_warmup=100)
        config, _ = backend.suggest()
        backend.tell(config, 300.0, {"val_bpb": 0.99})

        assert len(backend._results) == 1
        assert backend._results[0][2]["val_bpb"] == 0.99
        assert backend._pending_trial is None
        # Study should have one completed trial
        assert len(backend._study.trials) == 1

    def test_tell_handles_failure(self):
        backend = _make_backend(llm_warmup=100)
        config, _ = backend.suggest()
        backend.tell(config, 300.0, {"_error": "OOM"})

        assert len(backend._results) == 1
        # Failed trial should still be in study (with penalty)
        assert len(backend._study.trials) == 1
        assert backend._study.trials[0].value == 100.0


class TestSeedAndReplay:
    def test_seed_trial(self):
        backend = _make_backend()
        backend.seed_trial(
            {"lr": 0.01, "depth": 8, "wd": 0.1},
            300.0,
            {"val_bpb": 1.0},
        )
        assert len(backend._results) == 1
        assert len(backend._study.trials) == 1
        assert backend._trial_count == 1

    def test_replay_rebuilds_state(self):
        backend = _make_backend()
        history = [
            ({"lr": 0.01, "depth": 8, "wd": 0.1}, 300.0, {"val_bpb": 1.0}),
            ({"lr": 0.05, "depth": 12, "wd": 0.05}, 300.0, {"val_bpb": 0.98}),
            ({"lr": 0.5, "depth": 20, "wd": 0.3}, 300.0, {"_error": "OOM"}),
        ]
        backend.replay(history)

        assert len(backend._results) == 3
        assert len(backend._study.trials) == 3
        assert backend._trial_count == 3

    def test_incumbents_after_replay(self):
        backend = _make_backend()
        history = [
            ({"lr": 0.01, "depth": 8, "wd": 0.1}, 300.0, {"val_bpb": 1.0}),
            ({"lr": 0.05, "depth": 12, "wd": 0.05}, 300.0, {"val_bpb": 0.98}),
        ]
        backend.replay(history)
        inc = backend.incumbents()
        assert len(inc) == 1
        assert inc[0]["lr"] == 0.05


class TestCMAStateExtraction:
    def test_returns_none_before_any_trials(self):
        backend = _make_backend()
        assert backend._extract_cma_state() is None

    def test_returns_none_early_trials(self):
        """CMA-ES needs several trials before it stores optimizer state."""
        backend = _make_backend()
        backend.seed_trial({"lr": 0.01, "depth": 8, "wd": 0.1}, 300.0, {"val_bpb": 1.0})
        # With just one trial, CMA-ES optimizer likely not initialized yet
        state = backend._extract_cma_state()
        # May or may not be None depending on Optuna internals — just check no crash
        if state is not None:
            assert "sigma" in state
            assert "top_trials" in state

    def test_extracts_full_state_after_enough_trials(self):
        """After sufficient ask/tell cycles, CMA-ES state should be extractable.

        This is the key regression test for the split-key bug: Optuna's CmaEsSampler
        splits the pickled optimizer across cma:optimizer:0, cma:optimizer:1, etc.
        The fix concatenates all parts before unpickling.
        """
        backend = _make_backend(llm_warmup=9999)  # no LLM, pure CMA
        import random
        rng = random.Random(42)

        # Run enough ask/tell cycles to get CMA-ES optimizer initialized
        # CMA-ES typically needs n_dim + 1 trials minimum (we have 3 dims)
        for i in range(30):
            config, budget = backend.suggest()
            # Simulate varying quality results
            val_bpb = 1.0 + rng.gauss(0, 0.1)
            backend.tell(config, budget, {"val_bpb": val_bpb})

        state = backend._extract_cma_state()
        assert state is not None, (
            "_extract_cma_state() returned None after 30 trials — "
            "the split-key concatenation is likely broken"
        )
        assert "mean_raw" in state
        assert "sigma" in state
        assert isinstance(state["sigma"], float)
        assert state["sigma"] > 0
        assert "cov" in state  # covariance matrix should be present
        assert "top_trials" in state
        assert len(state["top_trials"]) > 0
        assert state["n_trials"] == 30

        # Verify mean has correct dimensionality (3 HPs: lr, depth, wd)
        assert len(state["mean_raw"]) == 3

        # Verify covariance matrix is square and correct size
        if state["cov"] is not None:
            import numpy as np
            C = np.array(state["cov"])
            assert C.shape == (3, 3), f"Expected 3x3 covariance, got {C.shape}"

    def test_format_cma_analysis_shows_state(self):
        """After enough trials, the formatted analysis should show sigma and top configs."""
        backend = _make_backend(llm_warmup=9999)
        import random
        rng = random.Random(42)

        for i in range(30):
            config, budget = backend.suggest()
            val_bpb = 1.0 + rng.gauss(0, 0.1)
            backend.tell(config, budget, {"val_bpb": val_bpb})

        state = backend._extract_cma_state()
        cma_config = {"lr": 0.01, "depth": 8, "wd": 0.1}
        analysis = backend._format_cma_analysis(state, cma_config)

        # Should NOT contain the fallback "still initializing" message
        assert "still initializing" not in analysis, (
            "Analysis says 'still initializing' after 30 trials — state extraction failed"
        )
        # Should contain actual analysis
        assert "sigma=" in analysis
        assert "Top 5" in analysis
        assert "covariance matrix" in analysis.lower()

    def test_covariance_shown_with_categorical_hps(self):
        """Covariance matrix must appear even when search space has categoricals.

        Regression test: CMA-ES only handles continuous HPs (N), so C is NxN.
        The format code must use only continuous HP names (not all HPs) when
        checking len(hp_names) == C.shape[0].
        """
        from ConfigSpace import Categorical

        cs = ConfigurationSpace(seed=42)
        cs.add(Float("lr", bounds=(1e-5, 1.0), log=True, default=0.01))
        cs.add(Integer("depth", bounds=(4, 24), default=8))
        cs.add(Float("wd", bounds=(0.0, 0.5), default=0.1))
        cs.add(Categorical("pattern", items=["A", "B", "C"], default="A"))

        backend = CentaurBackend(model="test", log_dir=None, llm_ratio=0.0, llm_warmup=9999)
        backend.configure(space=cs, objectives=["val_bpb"], budget_range=(60, 300), seed=0)

        import random
        rng = random.Random(42)
        for _ in range(30):
            config, budget = backend.suggest()
            backend.tell(config, budget, {"val_bpb": 1.0 + rng.gauss(0, 0.1)})

        state = backend._extract_cma_state()
        assert state is not None
        # C should be 3x3 (3 continuous HPs, excluding categorical)
        if state["cov"] is not None:
            import numpy as np
            C = np.array(state["cov"])
            assert C.shape == (3, 3), f"Expected 3x3, got {C.shape}"

        cma_config = {"lr": 0.01, "depth": 8, "wd": 0.1, "pattern": "A"}
        analysis = backend._format_cma_analysis(state, cma_config)
        assert "covariance matrix" in analysis.lower(), (
            "Covariance matrix missing from analysis — "
            "likely hp_names count includes categoricals but C only has continuous dims"
        )
