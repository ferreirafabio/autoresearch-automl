---
name: Live Benchmark tab on the demo (2026-05-15 onwards)
description: Architecture of docs/index.html's second tab — what scripts populate sections A/B/C/D, where data is read from, and how to add a new Claude release
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## "NEW" badge on the tab card

The Live Benchmark tab card carries a small pulsing red `NEW` badge
(class `tc-new-badge`) so returning visitors notice the addition.
Keep it on while the tab is still novel; strip the `<span class="tc-new-badge">NEW</span>`
from the title when the live tracker is no longer the freshest piece
of the page. CSS uses a 2.2s opacity+scale pulse so it draws the eye
without being annoying.

## Two-tab layout

`docs/index.html` has top-level card tabs: "Interactive Demo" (default, the paper figures) and "Live Benchmark". The tab state persists across reloads via the URL hash (`#tab=tracker`). Switch handler is `initTopTabs` near the bottom of index.html — calls `Plotly.Plots.resize` three times (0/50/250ms) when the tracker tab becomes visible, because Plotly figures created in a `hidden` panel start at width 0.

## Sections inside the Live Benchmark tab

| ID | Title | Populated by |
|---|---|---|
| `tracker-hero-container` | A. Convergence across Claude generations | `build_traces` + `_plot_html` |
| `tracker-slope-container` | B. Per-method progression across Claude generations | `build_slopegraph_html` |
| `tracker-forest-container` | C. Wilcoxon &Delta; vs TPE | `build_forest_html` (uses scipy.stats.wilcoxon) |
| `tracker-cards-container` | D. Per-Claude-generation summary | `build_cards_html` |

All four are emitted by `scripts/build_tracker_hero.py`, which reads from:
- `/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/optuna/` (TPE = classical reference)
- `/work/dlclarge1/ferreira-autoresearch-automl/results/opus46_benchmark/...`
- `/work/dlclarge1/ferreira-autoresearch-automl/results/opus47_benchmark/...`

Seeds are counted "completed" at >= 95% of 24h training budget (`COMPLETION_THRESHOLD`).

## Visual conventions

- **Color = method family** (matches paper Fig 1): Centaur=#C62828, KA Code=#1565C0, KA HPs=#FFC107, TPE=#2196F3.
- **Dash = Claude generation**: index 0 of `GENERATIONS` list = solid, 1 = dash, 2 = dashdot, then longdash/longdashdot/dot in `DASH_BY_GEN_INDEX`.
- Filter buttons carry SVG line icons whose `stroke-dasharray` mirrors the Plotly dash, so the pill itself says "this is the dotted/dashed/solid style".

## Adding a new Claude release (e.g. Opus 4.8)

1. Drop new SLURM scripts under `slurm/exp2_{centaur,karpathy_agent,karpathy_agent_hps}_claude_code_opus48.sh` (copy from `_opus47.sh`, swap model id to `claude-opus-4-8`).
2. Run 5 seeds per method, results land in `results/opus48_benchmark/...`.
3. Append one entry to `GENERATIONS` in `scripts/build_tracker_hero.py`:
   ```python
   {"tag": "Opus 4.8", "tag_id": "opus_4_8", "base": RESULTS / "opus48_benchmark",
    "centaur": "centaur_claude_opus_4_8",
    "ka_code": "karpathy_agent_claude_opus_4_8",
    "ka_hps":  "karpathy_agent_hps_claude_opus_4_8"},
   ```
4. Add the new opus_4_8 tag to `classifyTraceTags` in index.html's tracker-hero branch. (The parser already extracts `\[opus\s*(\d+)\.(\d+)\]` from the trace name, so as long as the bracket is present it auto-tags. Filter buttons are auto-generated from `_present_generations`.)
5. `PYTHONPATH=. python3 scripts/build_tracker_hero.py` — regenerates all four sections.
6. `git push origin main && git push public main` (demo only deploys from public; see project_repo_split.md).

## Helper scripts in the tab pipeline

- `scripts/build_tracker_hero.py` — the main builder. Run after any new seed lands.
- `scripts/sort_fig1_legend.py` — re-sorts the paper-tab Fig 1 legend by best val_bpb (best at top). Idempotent. Hands off to `add_opus_demo_traces.py`'s output.
- `scripts/add_opus_demo_traces.py` — older helper that injects Opus traces into the paper-tab Fig 1. Still used for the Interactive Demo tab.
- `scripts/wilcoxon_tests.py` — standalone Wilcoxon report (txt + forest png). Section C reuses scipy stats directly inside `build_tracker_hero.py` rather than importing this.

**Why:** the tracker tab is the user's "live frontier vs classical" view. The whole point is that each Claude release becomes a 5-minute config change rather than a custom plot.
