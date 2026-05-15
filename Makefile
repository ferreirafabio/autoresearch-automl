# Project utility targets.  Run with `make <target>` from the repo root.

REPO_ROOT       := $(shell pwd)
MEMORY_SRC      := $(HOME)/.claude/projects/-work-dlclarge1-ferreira-autoresearch-automl-autoresearch-automl/memory
MEMORY_DST      := $(REPO_ROOT)/.claude/memory

.PHONY: help memory-sync memory-commit tracker

help:
	@echo "Targets:"
	@echo "  memory-sync     One-way rsync ~/.claude project memory into .claude/memory/ (private repo)"
	@echo "  memory-commit   memory-sync then commit + push origin (private) if anything changed"
	@echo "  tracker         Rebuild the Live Benchmark tab (sections A/B/C/D) from current results"

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
