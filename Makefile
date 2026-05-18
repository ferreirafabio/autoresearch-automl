# Project utility targets.  Run with `make <target>` from the repo root.

REPO_ROOT       := $(shell pwd)
MEMORY_SRC      := $(HOME)/.claude/projects/-work-dlclarge1-ferreira-autoresearch-automl-autoresearch-automl/memory
MEMORY_DST      := $(REPO_ROOT)/.claude/memory

.PHONY: help memory-sync memory-commit tracker dropbox-sync dropbox-sync-repo

help:
	@echo "Targets:"
	@echo "  memory-sync     One-way rsync ~/.claude project memory into .claude/memory/ (private repo)"
	@echo "  memory-commit   memory-sync then commit + push origin (private) if anything changed"
	@echo "  tracker         Rebuild the Live Benchmark tab (sections A/B/C/D) from current results"
	@echo "  dropbox-sync    One-way rclone copy /work/.../results -> dropbox:autoresearch-automl/results"
	@echo "  dropbox-sync-repo  One-way rclone copy this repo dir -> dropbox:autoresearch-automl/repo"

memory-sync:
	@mkdir -p $(MEMORY_DST)
	@rsync -a --delete $(MEMORY_SRC)/ $(MEMORY_DST)/
	@echo "synced memory: $(MEMORY_SRC) -> $(MEMORY_DST)"

memory-commit: memory-sync
	@cd $(REPO_ROOT) && git add .claude/memory && \
	  if git diff --cached --quiet -- .claude/memory; then \
	    echo "memory: no changes to commit"; \
	  else \
	    git commit -m "memory: sync from \$$HOME/.claude/projects (auto)" && \
	    git push origin main; \
	  fi

tracker:
	@PYTHONPATH=. python3 scripts/build_tracker_hero.py

# Dropbox: one-way sync /work/.../results -> dropbox:autoresearch-automl/results
# and the repo dir -> dropbox:autoresearch-automl/repo
# Credentials in .env (gitignored). Run by hand or via cron.
dropbox-sync:
	@rclone copy /work/dlclarge1/ferreira-autoresearch-automl/results \
		dropbox:autoresearch-automl/results \
		--transfers=16 --checkers=16 --fast-list --progress

dropbox-sync-repo:
	@rclone copy $(REPO_ROOT) dropbox:autoresearch-automl/repo \
		--exclude '.venv/**' --exclude '__pycache__/**' --exclude '*.pyc' \
		--transfers=16 --checkers=16 --fast-list --progress
