"""Git-based experiment tracking — preserves Karpathy's ratchet pattern."""

from __future__ import annotations

import logging
from pathlib import Path

import git

logger = logging.getLogger(__name__)


class GitTracker:
    """Tracks experiments using git commits, preserving the ratchet pattern.

    Each trial that improves val_bpb gets committed. Structural mutations
    get tagged with generation numbers.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)

    def commit_improvement(
        self,
        val_bpb: float,
        config: dict,
        trial_id: int,
        backend_name: str,
        generation: int = 0,
    ) -> str:
        """Commit an improvement to the ratchet.

        Returns the commit hash.
        """
        # Stage train.py and results
        self.repo.index.add(["train.py"])
        results_tsv = self.repo_path / "results.tsv"
        if results_tsv.exists():
            self.repo.index.add(["results.tsv"])

        msg = (
            f"val_bpb={val_bpb:.4f} | trial={trial_id} | "
            f"backend={backend_name} | gen={generation}\n\n"
            f"Config: {config}"
        )

        commit = self.repo.index.commit(msg)
        logger.info("Committed improvement: %s (%.4f)", commit.hexsha[:8], val_bpb)
        return commit.hexsha

    def tag_generation(self, generation: int, description: str) -> None:
        """Tag the current commit as a structural generation milestone."""
        tag_name = f"gen-{generation}"
        self.repo.create_tag(tag_name, message=description)
        logger.info("Tagged generation %d: %s", generation, description)

    def get_best_val_bpb(self) -> float | None:
        """Parse the best val_bpb from recent commit messages."""
        best = None
        for commit in self.repo.iter_commits(max_count=100):
            msg = commit.message
            if "val_bpb=" in msg:
                try:
                    val_str = msg.split("val_bpb=")[1].split()[0].rstrip("|")
                    val = float(val_str)
                    if best is None or val < best:
                        best = val
                except (ValueError, IndexError):
                    continue
        return best

    def get_generation(self) -> int:
        """Get the current structural generation from tags."""
        gen = 0
        for tag in self.repo.tags:
            if tag.name.startswith("gen-"):
                try:
                    g = int(tag.name.split("-")[1])
                    gen = max(gen, g)
                except ValueError:
                    continue
        return gen
