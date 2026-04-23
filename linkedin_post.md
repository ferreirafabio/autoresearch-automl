Classical methods still win at hyperparameter tuning. Qwen3.5-0.8B, Qwen3.5-27B, and Gemini 3.1 Pro Preview all lose to TPE. The exception: hybrid methods. A tiny 0.8B LLM that peeks at CMA-ES's internal state matches classical methods.

Interactive demo: https://ferreirafabio.github.io/autoresearch-automl
Paper: https://arxiv.org/abs/2603.24647

Five findings from the paper, in response to questions on my earlier post:

1. The gap is consistent and goes beyond final performance. CMA-ES, TPE, and SMAC outperform pure LLM methods (some even lose to random search) and get there ~3-4x faster than the best LLM code-editing agent. Performance is not highly determined by search diversity, but by avoiding failures.

2. Classical optimizers learn what to avoid; LLMs don't. CMA-ES and TPE stay at 10-16% OOM, while LLM optimizers stay closer to random search's OOM rate (~56%).

3. LLMs are least bad at HPO via code editing. Among pure LLM methods, only the ones that directly edit training source code get closer to classical methods, but classical still wins.

4. Frontier models don't close the gap. Gemini 3.1 Pro Preview is competitive with Qwen3.5-27B for both code editing and hybrid methods, but does not outperform the best classical methods.

5. A tiny 0.8B LLM paired with CMA-ES (Centaur) is the best method in our benchmark. Sharing CMA-ES's full internal state with the LLM cuts cross-seed variance ~4x, and scaling to 27B gives no further gain: a cheap LLM suffices when paired with a strong classical optimizer.

Thanks to my co-authors Lucca Wobbe, Arjun Krishnakumar, Frank Hutter, Arber Zela for the collaboration.
