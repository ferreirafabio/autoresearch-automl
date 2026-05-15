---
name: Paper writing style and plot preferences
description: User's academic writing style, plot layout preferences, and paper structure decisions
type: feedback
---

**Writing style** (from user's papers: arxiv 2409.14084, 2206.08476):
- Direct empirical statements paired with specific metrics
- "We observe that...", "This suggests...", "Notably..."
- Honest about limitations, cautious but confident
- Structured: observation -> pattern -> interpretation -> limitation
- No hyperbole, understate rather than overstate

**Plot preferences:**
- Hero plot: wall-time (primary comparison, fair compute budget)
- Secondary plots (0.8B vs 27B, incumbents): trial count only (sample efficiency)
- Same method = same color across 0.8B/27B, distinguished by solid (27B) vs dashed (0.8B)
- Incumbent traces: all panels share same x-axis (300 trials), auto y-zoom to incumbent range
- No duplicate plots showing same data in different views

**Paper structure:**
- Search diversity table in appendix, referenced from results discussion
- No LaTeX paper exists yet in repo (as of 2026-03-21)

**Why:** User prefers concise main text with heavy appendix for details.

**How to apply:** When writing results text, use specific numbers, acknowledge limitations, keep main text focused. Put detailed tables in appendix.
