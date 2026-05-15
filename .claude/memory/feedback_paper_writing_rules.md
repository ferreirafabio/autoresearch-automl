---
name: Paper writing rules
description: Comprehensive writing style rules for academic papers, accumulated from extensive feedback
type: feedback
---

## Frank's writing rules (mandatory)

1. **Work hierarchically**: Overall message -> sections -> subsections -> paragraphs -> paragraph content sentences -> fill in text.
2. **Paragraph content sentence**: The first sentence of every paragraph must summarize what the paragraph is about. A reader should be able to skim the paper by reading only first sentences.
3. **No bold-face paragraph headers** like `\paragraph{Task.}` for short paragraphs. Either use proper subsections if long enough, or lead with the content sentence directly.
4. **Parenthetical references are not nouns**: Wrong: "In [10], X showed." Right: "X [10] showed." or "X and Y~\citep{...} showed."
5. **Avoid passive voice**: Use "We performed experiments" not "Experiments were performed."
6. **Tense**: Past tense for experiments actually performed ("we trained", "outperformed"). Present tense for interpretations that still hold ("Figure 2 shows", "this indicates").
7. **No contractions**: No "it's", "there's", "we'll", "that's" in scientific writing.

## User's style preferences

8. **No em dashes** (`---`). Use commas, periods, "such as", "including", or "i.e." instead.
9. **No rhetorical questions** mid-paragraph. State hypotheses explicitly or use "In this paper, we..." framing.
10. **No informal language**: Avoid "plays it too safe", "heavy lifting", "recipe", "Perhaps most strikingly", "actually hurt".
11. **No excessive parentheses**: Instead of `(CMA-ES, TPE)` write "such as CMA-ES and TPE". Instead of `(mean, σ, covariance matrix)` write "including mean, σ, and covariance matrix".
12. **Don't repeat numbers**: "9 methods" or "~50M parameters" should appear at most twice (abstract + one other place).
13. **Contributions format**: "In summary, we make the following contributions:" with noun-phrase bullets. Don't start every bullet with "We".
14. **No "fair/fairness"**: Use "controlled", "under the same budgets and constraints" instead.
15. **Soften strong claims**: "uniquely" -> "particularly". "best overall result" -> "best result in our experiments". Add "on this benchmark" to generalization claims.
16. **Always compile-check locally** before pushing to paper repo.
17. **Use lib.bib** for standard references before adding to local.bib. Always check lib.bib first.
18. **No "GPT-2 scale"**: The model is ~50M params, not GPT-2 (124M). Use "small decoder-only transformer".
19. **Intro structure**: Funnel opening -> gap -> "In this paper, we..." -> contributions. Separate Related Work section with content-leading first sentences per paragraph (no bold headers).
20. **Related Work style**: Group by topic (like Quick-Tune paper). "Several methods have been proposed... ranging from X to Y." Then position explicitly: "In contrast to prior work, we..."

## Data integrity rules

21. **Hard 24h cap**: All results, plots, JSONs, and diversity analyses must exclude trials beyond 24h cumulative training time.
22. **Forward-fill interpolation**: Use `right=values[-1]` in np.interp to avoid "line going up" artifact when seeds end slightly before 24h.
23. **Budget filter**: 95% (MIN_BUDGET_FRAC=0.95). Seeds below this are excluded from plots.
24. **Refresh repo copies**: When updating plots, always copy latest from original results dir, crop at 24h, then regenerate.
25. **Verify VRAM**: All methods must use CUDA_MEM_FRACTION=0.543 (76GB cap). Baseline ~44GB, peak ~75GB.
