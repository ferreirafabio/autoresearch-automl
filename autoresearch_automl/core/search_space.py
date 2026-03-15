"""Extract hyperparameters from train.py and build ConfigurationSpace."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

logger = logging.getLogger(__name__)

# ALL_CAPS names that are NOT hyperparameters (control flow, constants from prepare.py, etc.)
EXCLUDE_NAMES = frozenset({
    # Constants imported from prepare.py (read-only, cannot modify)
    "MAX_SEQ_LEN", "TIME_BUDGET", "EVAL_TOKENS",
    # Internal variables, not tunable HPs
    "H100_BF16_PEAK_FLOPS",
})

# Known HP metadata for Karpathy's autoresearch train.py
# These serve as defaults; can be overridden by LLM or config
KNOWN_HP_METADATA: dict[str, dict] = {
    # Model architecture
    "DEPTH": {"type": "int", "low": 4, "high": 24},
    "ASPECT_RATIO": {"type": "int", "low": 32, "high": 128},
    "HEAD_DIM": {"type": "int", "low": 64, "high": 256, "log": True},
    "DEVICE_BATCH_SIZE": {"type": "int", "low": 32, "high": 256, "log": True},
    # Optimization
    "TOTAL_BATCH_SIZE": {"type": "int", "low": 65536, "high": 2097152, "log": True},
    "EMBEDDING_LR": {"type": "float", "low": 0.01, "high": 2.0, "log": True},
    "UNEMBEDDING_LR": {"type": "float", "low": 0.0005, "high": 0.05, "log": True},
    "MATRIX_LR": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
    "SCALAR_LR": {"type": "float", "low": 0.05, "high": 2.0, "log": True},
    "WEIGHT_DECAY": {"type": "float", "low": 0.0, "high": 0.5},
    "WARMUP_RATIO": {"type": "float", "low": 0.0, "high": 0.3},
    "WARMDOWN_RATIO": {"type": "float", "low": 0.1, "high": 0.8},
    "FINAL_LR_FRAC": {"type": "float", "low": 0.0, "high": 0.2},
    # Attention pattern (string HP)
    "WINDOW_PATTERN": {"type": "cat", "choices": ["SSSL", "SSLL", "SLSL", "LLLL", "SSSS", "LSSL"]},
}

# Karpathy's starting config (commit b11d6f28, train.py lines 364-382).
# This is the UNOPTIMIZED starting point — NOT his final optimized values.
# Used as the baseline (trial 0) for all HPO runs.
KARPATHY_STARTING_CONFIG: dict[str, object] = {
    "ASPECT_RATIO": 64,
    "HEAD_DIM": 128,
    "WINDOW_PATTERN": "SSSL",
    "TOTAL_BATCH_SIZE": 2**19,  # 524288
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.2,
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
    "DEPTH": 8,
    "DEVICE_BATCH_SIZE": 128,
}

# HPs that must be rounded to nearest power of 2 after sampling
# (GPU memory alignment, batch size divisibility constraints)
POWER_OF_2_HPS = frozenset({"HEAD_DIM", "DEVICE_BATCH_SIZE", "TOTAL_BATCH_SIZE"})


def snap_to_power_of_2(config: dict[str, object]) -> dict[str, object]:
    """Snap power-of-2 HPs to nearest power of 2, enforce batch constraints."""
    result = dict(config)
    for name in POWER_OF_2_HPS:
        if name in result and isinstance(result[name], (int, float)):
            v = max(1, int(result[name]))
            low = 1 << (v.bit_length() - 1)
            high = low << 1
            result[name] = low if (v - low) <= (high - v) else high

    # Enforce TOTAL_BATCH_SIZE >= DEVICE_BATCH_SIZE * MAX_SEQ_LEN (2048)
    dbs = result.get("DEVICE_BATCH_SIZE")
    tbs = result.get("TOTAL_BATCH_SIZE")
    if isinstance(dbs, int) and isinstance(tbs, int):
        min_tbs = dbs * 2048
        if tbs < min_tbs:
            result["TOTAL_BATCH_SIZE"] = min_tbs

    return result


@dataclass
class ExtractedHP:
    """A hyperparameter extracted from train.py."""
    name: str
    value: object  # current value in train.py
    lineno: int
    col_offset: int


@dataclass
class SearchSpaceBuilder:
    """Extracts HPs from train.py and builds a ConfigurationSpace."""

    train_py_path: Path
    exclude_names: frozenset[str] = field(default_factory=lambda: EXCLUDE_NAMES)
    hp_metadata: dict[str, dict] = field(default_factory=lambda: dict(KNOWN_HP_METADATA))

    def get_default_config(self) -> dict[str, Any]:
        """Return the default config (train.py baseline values)."""
        cs = self.build_configspace()
        return dict(cs.get_default_configuration())

    def extract_hps(self) -> list[ExtractedHP]:
        """Parse train.py AST and find ALL_CAPS = literal assignments."""
        source = self.train_py_path.read_text()
        tree = ast.parse(source)
        hps = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id

            # Must be ALL_CAPS (with underscores)
            if not re.match(r"^[A-Z][A-Z0-9_]+$", name):
                continue
            if name in self.exclude_names:
                continue

            # Must be a literal value
            value = self._try_literal(node.value)
            if value is None:
                continue

            hps.append(ExtractedHP(
                name=name,
                value=value,
                lineno=node.lineno,
                col_offset=target.col_offset,
            ))
            logger.debug("Extracted HP: %s = %r (line %d)", name, value, node.lineno)

        return hps

    def build_configspace(
        self,
        include: list[str] | None = None,
        overrides: dict[str, dict] | None = None,
    ) -> ConfigurationSpace:
        """Build a ConfigurationSpace from extracted HPs.

        Args:
            include: If set, only include these HP names.
            overrides: Per-HP metadata overrides (type, low, high, etc.).
        """
        hps = self.extract_hps()
        overrides = overrides or {}

        cs = ConfigurationSpace(seed=42)

        for hp in hps:
            if include and hp.name not in include:
                continue

            meta = {**self.hp_metadata.get(hp.name, {}), **overrides.get(hp.name, {})}

            if not meta:
                # Infer from current value
                meta = self._infer_metadata(hp.name, hp.value)

            if not meta:
                logger.debug("Skipping %s — no metadata and cannot infer type", hp.name)
                continue

            param = self._make_param(hp.name, hp.value, meta)
            if param is not None:
                cs.add(param)
                logger.info("Added %s to ConfigSpace: %s", hp.name, param)

        return cs

    def _make_param(self, name: str, current_value: object, meta: dict):
        """Create a ConfigSpace hyperparameter from metadata."""
        hp_type = meta.get("type", "float")

        if hp_type == "cat":
            choices = meta.get("choices", [current_value])
            return Categorical(name, items=choices, default=current_value if current_value in choices else choices[0])

        if hp_type == "int":
            low = meta.get("low", max(1, int(current_value) // 4))
            high = meta.get("high", int(current_value) * 4)
            log = meta.get("log", False)
            default = max(low, min(high, int(current_value)))
            return Integer(name, bounds=(low, high), default=default, log=log)

        if hp_type == "float":
            low = meta.get("low", current_value / 10 if current_value > 0 else 0.0)
            high = meta.get("high", current_value * 10 if current_value > 0 else 1.0)
            log = meta.get("log", False)
            if log and low <= 0:
                low = 1e-8
            default = max(low, min(high, float(current_value)))
            return Float(name, bounds=(low, high), default=default, log=log)

        logger.warning("Unknown HP type %r for %s", hp_type, name)
        return None

    def _infer_metadata(self, name: str, value: object) -> dict:
        """Infer HP metadata from the current value."""
        if isinstance(value, bool):
            return {"type": "cat", "choices": [True, False]}
        if isinstance(value, str):
            return {"type": "cat", "choices": [value]}
        if isinstance(value, tuple):
            return {}  # tuples (e.g. ADAM_BETAS) need explicit metadata
        if isinstance(value, int):
            return {"type": "int", "low": max(1, value // 4), "high": value * 4}
        if isinstance(value, float):
            if 0 < value < 0.01:
                return {"type": "float", "low": value / 100, "high": value * 100, "log": True}
            if value > 0:
                return {"type": "float", "low": value / 10, "high": value * 10}
            return {"type": "float", "low": 0.0, "high": 1.0}
        return {}

    @staticmethod
    def _try_literal(node: ast.expr) -> int | float | str | bool | None:
        """Try to evaluate an AST node as a literal value."""
        try:
            value = ast.literal_eval(node)
            if isinstance(value, (int, float, str, bool, tuple)):
                return value
        except (ValueError, TypeError):
            pass

        # Handle simple expressions like 2**14
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = SearchSpaceBuilder._try_literal(node.left)
            exp = SearchSpaceBuilder._try_literal(node.right)
            if isinstance(base, (int, float)) and isinstance(exp, (int, float)):
                return int(base ** exp) if isinstance(base, int) and isinstance(exp, int) else base ** exp

        return None


def configspace_to_optuna(cs: ConfigurationSpace) -> dict[str, dict]:
    """Convert ConfigurationSpace to Optuna suggest_* kwargs.

    Returns a dict mapping HP name → dict with keys:
        'method': 'suggest_int' | 'suggest_float' | 'suggest_categorical'
        'kwargs': dict to pass to trial.suggest_*
    """
    mapping = {}
    for hp in cs.values():
        if isinstance(hp, UniformIntegerHyperparameter):
            mapping[hp.name] = {
                "method": "suggest_int",
                "kwargs": {"name": hp.name, "low": hp.lower, "high": hp.upper, "log": hp.log},
            }
        elif isinstance(hp, UniformFloatHyperparameter):
            mapping[hp.name] = {
                "method": "suggest_float",
                "kwargs": {"name": hp.name, "low": hp.lower, "high": hp.upper, "log": hp.log},
            }
        elif isinstance(hp, CategoricalHyperparameter):
            mapping[hp.name] = {
                "method": "suggest_categorical",
                "kwargs": {"name": hp.name, "choices": list(hp.choices)},
            }
    return mapping
