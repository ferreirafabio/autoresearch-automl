"""ObjectiveFunction wrapper: run experiment, parse metrics."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from running a single experiment."""
    val_bpb: float | None = None
    train_loss: float | None = None
    peak_memory_gb: float | None = None
    wall_time_seconds: float | None = None
    tokens_per_second: float | None = None
    success: bool = False
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    extra_metrics: dict[str, float] = field(default_factory=dict)


class ObjectiveFunction:
    """Run train.py and parse metrics to evaluate a configuration.

    This wraps the experiment execution: run the training script with a given
    time budget, capture stdout/stderr, parse val_bpb and other metrics.
    """

    def __init__(
        self,
        train_py_path: Path,
        run_command: list[str] | None = None,
        default_budget: float = 300.0,
        gpu_device: int | None = None,
        extra_env: dict[str, str] | None = None,
    ):
        self.train_py_path = train_py_path
        # Default: uv run train.py (as per Karpathy's setup)
        self.run_command = run_command or ["uv", "run", str(train_py_path)]
        self.default_budget = default_budget
        self.gpu_device = gpu_device
        self.extra_env = extra_env or {}

    def evaluate(self, budget: float | None = None) -> ExperimentResult:
        """Run train.py and return parsed metrics.

        Args:
            budget: Time budget in seconds (overrides default).

        Returns:
            ExperimentResult with parsed metrics.
        """
        budget = budget or self.default_budget
        result = ExperimentResult()

        env = dict(self.extra_env)
        if self.gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_device)

        start_time = time.time()
        try:
            # Timeout: budget + 300s grace for startup/compilation overhead
            # (train.py excludes first 10 steps from time budget for torch.compile)
            # program.md says kill after 10 minutes total
            timeout = max(budget + 300, 600)
            proc = subprocess.run(
                self.run_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**dict(__import__("os").environ), **env},
                cwd=self.train_py_path.parent,
            )
            result.wall_time_seconds = time.time() - start_time
            result.stdout = proc.stdout
            result.stderr = proc.stderr

            if proc.returncode != 0:
                result.error = f"train.py exited with code {proc.returncode}"
                logger.warning("train.py failed: %s", result.error)
                return result

            result.success = True
            self._parse_metrics(result)

        except subprocess.TimeoutExpired as e:
            result.wall_time_seconds = time.time() - start_time
            result.stdout = e.stdout or ""
            result.stderr = e.stderr or ""
            result.error = f"Timeout after {budget}s"
            # Still try to parse partial output
            self._parse_metrics(result)
            if result.val_bpb is not None:
                result.success = True

        except Exception as e:
            result.wall_time_seconds = time.time() - start_time
            result.error = str(e)
            logger.exception("Unexpected error running train.py")

        return result

    def _parse_metrics(self, result: ExperimentResult) -> None:
        """Parse metrics from train.py's final summary block.

        Karpathy's train.py prints a summary after '---':
            val_bpb:          0.997900
            training_seconds: 300.1
            total_seconds:    325.9
            peak_vram_mb:     45060.2
            mfu_percent:      39.80
            total_tokens_M:   499.6
            num_steps:        953
            num_params_M:     50.3
            depth:            8

        Training loop prints lines like:
            step 00100 (33.3%) | loss: 3.456789 | ... | tok/sec: 123,456 | ...
        """
        output = result.stdout + "\n" + result.stderr

        # Parse final summary block (key: value lines after '---')
        summary_metrics = {}
        for match in re.finditer(r"^(\w+):\s+([0-9.]+)", output, re.MULTILINE):
            summary_metrics[match.group(1)] = float(match.group(2))

        if "val_bpb" in summary_metrics:
            result.val_bpb = summary_metrics["val_bpb"]
        if "peak_vram_mb" in summary_metrics:
            result.peak_memory_gb = summary_metrics["peak_vram_mb"] / 1024
        if "training_seconds" in summary_metrics:
            result.wall_time_seconds = summary_metrics["training_seconds"]

        # Store all summary metrics as extras
        for key, val in summary_metrics.items():
            if key not in ("val_bpb", "peak_vram_mb", "training_seconds"):
                result.extra_metrics[key] = val

        # Parse smoothed training loss from training loop
        loss_matches = re.findall(r"loss:\s+([0-9.]+)", output)
        if loss_matches:
            result.train_loss = float(loss_matches[-1])

        # Parse tokens/sec from training loop
        tps_matches = re.findall(r"tok/sec:\s+([0-9,]+)", output)
        if tps_matches:
            result.tokens_per_second = float(tps_matches[-1].replace(",", ""))

        # Check for FAIL (train.py prints "FAIL" and exits with code 1 on NaN/exploding loss)
        if "FAIL" in output:
            result.success = False
            result.error = result.error or "Training failed (NaN or exploding loss)"

    def _parse_results_tsv(self, result: ExperimentResult, tsv_path: Path) -> None:
        """Parse Karpathy's results.tsv for the latest val_bpb."""
        try:
            lines = tsv_path.read_text().strip().splitlines()
            if len(lines) < 2:
                return
            header = lines[0].split("\t")
            last_row = lines[-1].split("\t")
            for col_name, value in zip(header, last_row):
                col_lower = col_name.strip().lower()
                if col_lower in ("val_bpb", "val bpb"):
                    result.val_bpb = float(value)
                elif col_lower not in ("step", "time", "date"):
                    try:
                        result.extra_metrics[col_lower] = float(value)
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug("Could not parse results.tsv: %s", e)
