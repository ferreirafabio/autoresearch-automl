# Centaur (CMA-ES+LLM): CMA-ES Guided LLM Optimization

## How it works

Centaur is a hybrid HPO algorithm that pairs CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with an LLM.

**CMA-ES** maintains a multivariate Gaussian over the search space: a mean vector (center of the promising region), a step-size sigma (search radius), and a covariance matrix (learned HP correlations). It runs on every trial, always updating its model of the landscape.

**The LLM** is called on a fraction of trials (controlled by `llm_ratio`, default 30%, after a `llm_warmup` of 10 pure CMA-ES trials). When called, the LLM sees:
- CMA-ES's current mean (where CMA-ES thinks is promising)
- CMA-ES's sigma (how focused/exploratory CMA-ES currently is)
- CMA-ES's full covariance matrix C (14×14, which HPs co-vary in good regions)
- CMA-ES's own next suggestion
- The top-5 configs found so far
- The last 20 trial results
- The full search space with bounds

The LLM uses this structured analysis plus its domain knowledge (e.g., transformer training dynamics, OOM-prone regions) to suggest a config. CMA-ES then **learns from the LLM's result** via its normal tell() cycle, so the covariance matrix adapts to include LLM-chosen points.

### Algorithm

```
1. Initialize CMA-ES with seed
2. For each trial t:
   a. CMA-ES proposes a config (always, maintains internal state)
   b. If t < warmup OR not an LLM turn: evaluate CMA-ES's config
   c. If LLM turn:
      - Extract CMA-ES state (mean, sigma, covariance matrix, top configs)
      - Build prompt with CMA-ES analysis + history + search space
      - LLM suggests config, clamp to bounds
      - Override CMA-ES's trial params with LLM config
   d. Evaluate the config
   e. Tell CMA-ES the result (always, it learns from ALL trials)
```

Key invariant: CMA-ES ask/tell runs on every trial. On LLM turns, CMA-ES proposes but the LLM overrides. CMA-ES still learns from the LLM's result, so its covariance adapts to the full trajectory.

## How CMA-ES + LLM contrasts with related work

### vs. LLAMBO (Liu et al., ICLR 2024)

LLAMBO replaces the **surrogate model** (normally a Gaussian Process) inside Bayesian Optimization with an LLM. The LLM predicts "config X will score ~0.98", and then a classical **acquisition function** (Expected Improvement) picks the best candidate from LLM-predicted scores.

- LLAMBO: LLM is a **component inside** BO (the surrogate). Classical math still makes the final decision.
- CMA-ES + LLM: LLM and CMA-ES are **two separate decision-makers** taking turns. The LLM makes the final decision on its turns, informed by CMA-ES's state.

Another key difference: LLAMBO's LLM sees the raw history and must implicitly learn the landscape. In CMA-ES + LLM, the LLM sees CMA-ES's **explicit landscape model** (mean, sigma, covariance, convergence state), a structured summary that's naturally expressible in language.

### vs. SLLMBO (Mahammadli & Ertekin, 2024)

SLLMBO is the closest prior work. It's a hybrid of LLM + TPE (Tree-structured Parzen Estimator). Both propose candidates, and the framework selects between them.

Differences:
- **Optimizer choice**: SLLMBO uses TPE (models each HP independently via kernel density estimation). CMA-ES + LLM uses CMA-ES (models joint HP distribution with full covariance). CMA-ES's state (mean vector, sigma, covariance) is far more interpretable to an LLM than TPE's density estimators.
- **Information flow**: In SLLMBO the LLM doesn't see the optimizer's internal state, it just sees trial history. In CMA-ES + LLM, the LLM receives CMA-ES's mean, sigma, covariance matrix, and top configs as structured guidance. This is the core insight: CMA-ES's Gaussian model translates naturally to language ("the center of the promising region is here, the search radius is X").
- **Learning**: CMA-ES always learns from LLM trials (the override-then-tell mechanism), so its covariance matrix incorporates LLM-chosen points. The two methods co-adapt.

### vs. Pure LLM HPO (Zhang et al., 2023)

Pure LLM approaches (like `karpathy_agent_hps` in this repo) prompt the LLM with history and ask for the next config. No traditional optimizer involved.

- Pure LLM has domain knowledge but no optimization state. It can't track covariance structure or convergence across trials.
- CMA-ES + LLM gives the LLM an optimization "advisor" that compensates for this: CMA-ES tracks the landscape, the LLM brings the domain knowledge.

### vs. LLaMA-ES (Kramer, ESANN 2024)

LLaMA-ES uses an LLM to tune CMA-ES's **own hyperparameters** (rankmu, rankone). It's meta-level: the LLM optimizes the optimizer itself, not the search space.

CMA-ES + LLM operates at the search-space level: the LLM directly suggests HP configs, guided by CMA-ES's landscape model.

### Why CMA-ES specifically (not TPE, GP-BO, etc.)

CMA-ES's internal state is uniquely interpretable for LLM communication:
- **Mean vector** → "CMA-ES thinks this region is promising" (a concrete config the LLM can read)
- **Sigma** → "CMA-ES is exploring widely / converging tightly" (a single scalar)
- **Covariance matrix C** → which HPs co-vary in good regions (14×14, with labeled rows/columns)

We pass all three to the LLM. The covariance matrix is annotated with HP name mappings so the LLM can reason about learned HP relationships (e.g., DEPTH and DEVICE_BATCH_SIZE trading off).

Compare with TPE (two separate density estimators, hard to summarize) or GP-BO (posterior mean + variance over the full space, high-dimensional, not concise). CMA-ES gives you a "center + search radius + learned correlations" story that fits naturally in a prompt.

## References

- LLAMBO: Liu et al., "Large Language Models to Enhance Bayesian Optimization", ICLR 2024. [arXiv:2402.03921](https://arxiv.org/abs/2402.03921)
- SLLMBO: Mahammadli & Ertekin, "Sequential Large Language Model-Based Hyperparameter Optimization", 2024. [arXiv:2410.20302](https://arxiv.org/abs/2410.20302)
- LLM for HPO: Zhang et al., "Using Large Language Models for Hyperparameter Optimization", 2023. [arXiv:2312.04528](https://arxiv.org/abs/2312.04528)
- LLaMA-ES: Kramer, "LLaMA Tunes CMA-ES", ESANN 2024. [PDF](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-136.pdf)
