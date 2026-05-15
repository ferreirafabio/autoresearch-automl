---
name: Always fetch Overleaf before editing or pushing; never auto-resolve conflicts
description: Pre-edit and pre-push workflow for any repo with an `overleaf` remote — fetch first, diff, ask before resolving conflicts or overwriting
type: feedback
originSessionId: e0a34323-1589-4848-a6c1-dd6bdb721e95
---
For any local checkout with an `overleaf` remote:

**Before editing any tracked file:** `git fetch overleaf` and `git pull --ff-only overleaf/master` (or the active Overleaf branch). Confirm the working tree mirrors the live Overleaf state before making edits — the user routinely edits on Overleaf, so stale local content can clobber their online work.

**Before pushing back:** show `git diff overleaf/master` (full set of changes), ask for explicit go-ahead, and only then push. Never silently merge with `-X ours` or `-X theirs`.

**On any conflict during fetch/pull/merge:** stop, surface the diff, and ask the user how to resolve. Do not auto-resolve under any flag.

**Why:** Collaborators (and the user) edit on Overleaf in real time. Pushing from a stale local checkout — or auto-resolving a fetch-time conflict — silently overwrites their work and is hard to recover from.

**How to apply:** At the start of any paper-revision session, `git remote -v` to confirm the Overleaf URL is correct (match against the project URL the user references), then `git fetch overleaf && git pull --ff-only`. Before any push, `git diff overleaf/<branch>` and surface to user. The user explicitly reaffirmed this on 2026-04-28: "its good if you checkout again before taking local file and overwriting stuff online."
