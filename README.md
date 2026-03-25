<h2 align="center"><code>autoresearch-automl</code></h2>
<h3 align="center">Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on <i>autoresearch</i></h3>

> **Paper in progress.** All results are now final with 3 seeds for all 9 methods. Earlier versions shared on LinkedIn used 2 seeds for Karpathy Agent (Code) [27B]; the third seed performed notably worse, widening the gap between this method and classical HPO. The plots and tables below reflect the complete 3-seed results.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM agent edit training code through trial and error, with no fixed search space, just code diffs. [Shwartz Ziv showed](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) that a classical AutoML method (TPE + expert HPs) can beat it. This makes autoresearch an excellent in-the-wild testbed to assess classical AutoML/HPO methods against newer LLM-based (agent) methods. We extend Karpathy's and Shwartz Ziv's experiments with a more extensive classical HPO vs. LLM-based comparison. We compare classical HPO (TPE, CMA-ES, SMAC, Random Search), LLM-based HPO ([LLAMBO](https://arxiv.org/abs/2402.03921), Karpathy Agent), and Centaur, a hybrid method that augments CMA-ES with LLM suggestions informed by the optimizer's internal state. All under the same budgets and constraints.

![HPO Convergence](assets/exp2_27b_walltime.png)

## Table of Contents

- [Methods](#methods)
- [Setup](#setup)
- [Results](#results)
  - [All Methods](#all-methods)
  - [0.8B vs 27B LLM Optimizer](#08b-vs-27b-llm-optimizer)
  - [Incumbent Traces](#incumbent-traces)
  - [Centaur (CMA-ES+LLM)](#centaur-cma-esllm-cma-es-guided-llm-optimization)
  - [Search Diversity Analysis](#search-diversity-analysis)
- [Search Space](#search-space)
- [Usage](#usage)
- [Related work](#related-work)
- [Details](#details)
- [Acknowledgements](#acknowledgements)

## Methods

**Classical (fixed [14-HP search space](#search-space)):**
- **TPE:** Tree-structured Parzen Estimator ([Optuna](https://github.com/optuna/optuna)).
- **CMA-ES:** Covariance Matrix Adaptation Evolution Strategy ([Optuna CMA sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html)).
- **SMAC:** Sequential Model-based Algorithm Configuration with Random Forest surrogate ([SMAC3](https://github.com/automl/SMAC3)).
- **Random:** Uniform random sampling ([Optuna RandomSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html)).

**LLM-based (fixed [14-HP search space](#search-space)):**
- **LLAMBO (Optuna):** LLM as surrogate + candidate generator inside Bayesian optimization ([OptunaHub port](https://hub.optuna.org/samplers/llambo/)). The OptunaHub implementation has several issues: it uses binary surrogate labels (good/bad) instead of continuous values, delegates categorical HPs to random sampling, and hides failed trials from the surrogate (see [Details](#llambo-optuna-vs-llambo-paper)).
- **LLAMBO (Paper):** Our reimplementation faithful to the original paper, fixing the above issues: continuous surrogate labels, all HPs visible to the LLM, failed trials included ([Ye et al., 2024](https://arxiv.org/abs/2402.03921), [original code](https://github.com/tennisonliu/LLAMBO)).
- **Karpathy Agent (14 HPs):** LLM sees trial history and suggests the next config within the fixed search space. No surrogate model, pure LLM suggestion.

**LLM-based (open, no fixed search space):**
- **Karpathy Agent (Code):** LLM directly edits `train.py` source code each trial. Can change any constant, not just the 14 extracted HPs ([Karpathy's autoresearch](https://github.com/karpathy/autoresearch)).

**Hybrid (fixed [14-HP search space](#search-space)):**
- **Centaur (CMA-ES+LLM):** CMA-ES runs every trial, continuously learning the optimization landscape (covariance structure, convergence direction). On 30% of trials (after 10 warmup), the LLM receives CMA-ES's internal state (distribution mean, step-size sigma, covariance matrix, top configs) and suggests a config informed by both the learned landscape and transformer domain knowledge. CMA-ES always refits on all results, including LLM-suggested ones. See [centaur.md](centaur.md).

All LLM methods use self-hosted Qwen3.5 (0.8B and 27B) via vLLM on the same GPU as training.

## Setup

Single H200 GPU, 5 min/trial, minimize val_bpb. Search space: 14 HPs auto-extracted from `train.py` via [AST](https://docs.python.org/3/library/ast.html) parsing (every `ALL_CAPS = literal` assignment becomes a tunable HP, no manual curation). See [Search Space](#search-space) for the full table. 3 seeds per condition.

**Setup:** All methods get 24 hours of GPU training time (excluding LLM inference overhead), capped to ~80 GB VRAM (to match the H100 used in Karpathy's and Shwartz Ziv's experiments). Failed trials reported as `val_bpb=100.0` so samplers learn to avoid OOM regions. Results are trimmed to 300 trials, as no meaningful improvement occurs beyond that point.

## Results

### All Methods

The wall-time plot above shows convergence against cumulative training time, which is our primary comparison: all methods receive the same 24-hour training budget, and LLM inference overhead is not counted against it. The trial-number plot below shows sample efficiency — how many evaluations each method needs. These views can look quite different because LLM-based methods spend additional real time on inference between trials, compressing their wall-time curves even when they are competitive per trial.

![HPO Convergence (by trial)](assets/exp2_27b_convergence.png)

### Key Observations

We observe that classical HPO methods consistently outperform pure LLM-based approaches within a fixed search space. The top methods by mean best val_bpb are: Centaur [27B] (0.9763), Centaur [0.8B] (0.9766), TPE (0.9768), SMAC (0.9778), CMA-ES (0.9785), and Karpathy Agent (Code) [27B] (0.9814). Karpathy Agent (Code) operates outside the fixed search space by editing source code directly, making it the only LLM method competitive with classical approaches. The gap to the best pure LLM method within the fixed search space (LLAMBO (Paper) [27B] at 0.9862) is substantial. Perhaps most strikingly, several pure LLM methods — including Karpathy Agent (14 HPs) and LLAMBO (Optuna) [27B] — perform worse than random search within the fixed search space. This suggests that, at least for this task, LLMs used as standalone HP optimizers can actually hurt optimization compared to uniform random sampling.

It is worth noting that all our LLM methods use open-weight models (Qwen3.5 0.8B and 27B). With stronger frontier models, code-editing methods like Karpathy Agent (Code) and LLM-based surrogate methods may improve significantly and potentially outperform some classical methods, especially on tasks where domain knowledge and code-level modifications matter more.

Model size plays a nuanced role. For free-form code editing, a larger LLM clearly helps: Karpathy Agent (Code) [27B] significantly outperforms its 0.8B counterpart (0.9814 vs 0.9910), as the bigger model produces more coherent and architecturally sound code modifications. However, when the search space is restricted to 14 fixed hyperparameters, scaling up the LLM from 0.8B to 27B provides no measurable benefit — Karpathy Agent (14 HPs) [27B] performs comparably to its 0.8B version. This indicates that the bottleneck in fixed-HP optimization is not the LLM's reasoning capacity but rather the optimization strategy itself.

LLM-based methods generally converge more slowly than classical methods in wall-clock time. The most promising result comes from the hybrid approach: Centaur combines CMA-ES with an LLM that receives the optimizer's internal state and occasionally suggests informed perturbations. By using the LLM on only 30% of trials, Centaur preserves the fast convergence of CMA-ES while benefiting from occasional LLM-informed suggestions. Notably, Centaur [0.8B] outperforms both CMA-ES alone and Centaur [27B], demonstrating that a small, inexpensive LLM is sufficient when paired with a strong classical optimizer.

### 0.8B vs 27B LLM Optimizer

Does LLM size matter for HPO? Solid lines = 27B, dashed = 0.8B.

![Model size comparison (wall-time)](assets/exp2_all_walltime.png)

Trial-number view (sample efficiency):

![Model size comparison (by trial)](assets/exp2_all_convergence.png)

### Incumbent Traces

Grey dots are all trials, colored dots are new bests, staircase is the incumbent (best-so-far). Each panel shows the best seed for that method.

**Classical + Hybrid:**

![Incumbent Traces — Classical + Hybrid](assets/exp2_incumbents_classical.png)

**LLM-based:**

![Incumbent Traces — LLM-based](assets/exp2_incumbents_llm.png)

### Centaur (CMA-ES+LLM): CMA-ES Guided LLM Optimization

We introduce **Centaur (CMA-ES+LLM)**, a hybrid backend where CMA-ES is the primary optimizer that occasionally consults an LLM. CMA-ES runs every trial, learning the optimization landscape (covariance structure, convergence direction). On a fraction of trials (30%, after 10 warmup trials), the LLM receives CMA-ES's internal state (distribution mean, step-size sigma, covariance matrix, top configs) and uses it alongside transformer domain knowledge to suggest configs. CMA-ES learns from all results, including LLM-suggested ones. See [centaur.md](centaur.md) for the full algorithm and related work comparison.

### Search Diversity Analysis

To understand *why* some methods outperform others, we measure how each backend explores the 14-dimensional HP space (WINDOW_PATTERN is categorical and excluded from distance metrics, leaving 13 continuous dimensions). All values are normalized to [0,1] within their bounds. Only successful (non-OOM) trials are included.

**Metrics:**
- **Spread:** mean per-HP standard deviation (higher = more diverse sampling across each dimension)
- **Pairwise:** mean L2 distance between all config pairs (higher = configs are more different from each other)
- **Dist→Default:** mean L2 distance from Karpathy's default config (higher = exploring further from the starting point)
- **Step:** mean L2 distance between consecutive trials (higher = larger jumps between suggestions)
- **Cells:** unique cells when discretizing each HP into 5 bins (higher = more coverage of the search space)

| Method | Seeds | Avg Best | OOM% | Spread | Pairwise | Dist→Default | Step | Cells |
|--------|-------|----------|------|--------|----------|-------------|------|-------|
| Centaur [27B] | 3 | **0.9765** | 14% | 0.113 | 0.542 | 0.502 | 0.324 | 293 |
| Centaur [0.8B] | 3 | **0.9766** | 13% | 0.131 | 0.642 | 0.543 | 0.368 | 317 |
| TPE | 3 | **0.9768** | 11% | 0.197 | 0.999 | 0.938 | 0.413 | 494 |
| Karpathy Agent (Code) [27B] | 3 | **0.9814** | 11% | - | - | - | - | - |
| CMA-ES | 3 | **0.9791** | 15% | 0.156 | 0.785 | 0.925 | 0.581 | 717 |
| SMAC | 3 | **0.9803** | 32% | 0.239 | 1.198 | 0.935 | 0.373 | 195 |
| LLAMBO (Paper) [27B] | 3 | 0.9862 | 43% | 0.242 | 1.201 | 1.016 | 1.153 | 327 |
| Random | 3 | 0.9890 | 56% | 0.274 | 1.386 | 1.250 | 1.392 | 289 |
| LLAMBO (Optuna) [27B] | 3 | 0.9896 | 65% | 0.205 | 0.992 | 1.027 | 0.805 | 268 |
| Karpathy Agent (14 HPs) [27B] | 3 | 0.9904 | 1% | 0.035 | 0.225 | 0.228 | 0.057 | 29 |

**Observations:**

- **Karpathy Agent (14 HPs) has the lowest diversity by all metrics.** Spread 0.028 (10x less than random), only 24 unique grid cells, dist→default 0.240. The LLM makes minimal changes between trials (step 0.057), suggesting it converges to a narrow region early and fails to explore.
- **LLAMBO (Optuna) has a 60% OOM rate**, largely due to delegating the categorical HP (WINDOW_PATTERN) to random sampling, which causes the LLM surrogate to miss structure in the search space.
- **LLAMBO (Paper) is one of the most diverse methods** (spread 0.240, 234 unique cells), yet still underperforms all classical methods. High diversity alone does not guarantee good performance.
- **The top 6 methods (Centaur, TPE, Karpathy Agent Code, CMA-ES, SMAC) all have OOM rates at or below 15%**, except SMAC at 32%.
- **SMAC improved significantly after the facade fix** (RF instead of GP surrogate). OOM rate dropped from ~60% to 32%, and avg best improved from 1.0015 to 0.9803.
- **Performance correlates more with OOM avoidance than with diversity.** Methods that learn to stay within feasible VRAM regions consistently outperform those that explore broadly but waste trials on OOM configurations.

## Search Space

Classical, non-LLM-based methods work with search spaces. The quality of the search space greatly affects the results produced by these methods. To reduce human priors in the search space, we automatically extract hyperparameters rather than curating them manually. To do this, we automatically extract hyperparameters from `train.py` using [AST](https://docs.python.org/3/library/ast.html) (Abstract Syntax Tree) parsing: the source code is parsed into a syntax tree, and every top-level `ALL_CAPS = literal` assignment is identified as a tunable HP. We extract the following 14 hyperparameters to optimize (13 continuous/integer + 1 categorical):

| HP | Type | Range | Log | Default |
|----|------|-------|-----|---------|
| DEPTH | int | 4 – 24 | | 8 |
| ASPECT_RATIO | int | 32 – 128 | | 64 |
| HEAD_DIM | int | 64 – 256 | yes | 128 |
| DEVICE_BATCH_SIZE | int | 32 – 256 | yes | 128 |
| TOTAL_BATCH_SIZE | int | 65 536 – 2 097 152 | yes | 524 288 |
| EMBEDDING_LR | float | 0.01 – 2.0 | yes | 0.6 |
| UNEMBEDDING_LR | float | 0.0005 – 0.05 | yes | 0.004 |
| MATRIX_LR | float | 0.005 – 0.2 | yes | 0.04 |
| SCALAR_LR | float | 0.05 – 2.0 | yes | 0.5 |
| WEIGHT_DECAY | float | 0.0 – 0.5 | | 0.2 |
| WARMUP_RATIO | float | 0.0 – 0.3 | | 0.0 |
| WARMDOWN_RATIO | float | 0.1 – 0.8 | | 0.5 |
| FINAL_LR_FRAC | float | 0.0 – 0.2 | | 0.0 |
| WINDOW_PATTERN | categorical | SSSL, SSLL, SLSL, LLLL, SSSS, LSSL | | SSSL |

Defaults are Karpathy's starting config (commit `b11d6f28`), not his final optimized values.

## Usage

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,all]"

# TPE
python -m autoresearch_automl.cli run --backend optuna --trials 100 --seed 0

# Random Search
python -m autoresearch_automl.cli run --backend random --trials 100 --seed 0

# SMAC3
python -m autoresearch_automl.cli run --backend smac --trials 100 --seed 0

# CMA-ES
python -m autoresearch_automl.cli run --backend cma_es --trials 100 --seed 0

# LLAMBO (Optuna) (requires vLLM running)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
python -m autoresearch_automl.cli run --backend llambo --trials 100 --llm-model Qwen3.5-0.8B

# LLAMBO (Paper)
python -m autoresearch_automl.cli run --backend llambo_original --trials 100 --llm-model Qwen3.5-0.8B

# Karpathy Agent (14 HPs) — LLM suggests configs within fixed search space
python -m autoresearch_automl.cli run --backend karpathy_agent_hps --trials 100 --llm-model Qwen3.5-0.8B

# Karpathy Agent (Code) — edits train.py directly
python -m autoresearch_automl.cli run --backend karpathy_agent --trials 100 --llm-model Qwen3.5-0.8B

# Centaur (CMA-ES+LLM)
python -m autoresearch_automl.cli run --backend centaur --trials 100 --llm-model Qwen3.5-0.8B
```

## Related work

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the training task and the idea of LLM-driven experimentation
- [Ravid Shwartz Ziv](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) for showing that expert HP selection beats blind LLM search
- [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) for using LLMs as surrogate models in Bayesian optimization

## Details

### H200 vs H100 baseline

Our baseline (Karpathy's default config) achieves val_bpb ≈ 0.991 on H200 at full clock speed (~1750K tokens/s), comparable to Karpathy's ~0.998 on H100. Early runs showed a higher baseline of ~1.008 due to GPU power throttling: under sustained load, some H200 nodes clocked down to ~1600 MHz (81% of max 1980 MHz), yielding fewer training steps per 5-minute trial. All results reported here use non-throttled H200s. The remaining gap to Karpathy's H100 result reflects minor hardware differences, not a systematic bias.

### LLAMBO (Optuna) vs LLAMBO (Paper)

While integrating LLAMBO, we discovered that the [OptunaHub LLAMBO sampler](https://hub.optuna.org/samplers/llambo/) (LLAMBO (Optuna)) differs from the [original paper code](https://github.com/tennisonliu/LLAMBO) (LLAMBO (Paper)) in several ways that materially affect optimization quality. OptunaHub does great work making research accessible; these notes are meant to help users who need paper-faithful behavior.

**Key differences:**

| Aspect | Original paper | OptunaHub port |
|--------|---------------|----------------|
| **Surrogate labels** | Actual metric values (`## 0.970 ##`), LLM sees performance gradients | Binary 0/1 (top 20% threshold), LLM only sees "good" vs "bad" |
| **Categorical HPs** | All HPs included in LLM prompts | Categoricals delegated to random sampling, invisible to LLM |
| **Failed trials** | Visible to surrogate (can learn infeasible regions) | Marked as `TrialState.FAIL`, invisible to surrogate |

**Impact on our experiments:** The categorical delegation was the most painful. Our `WINDOW_PATTERN` hyperparameter (attention pattern per layer) strongly affects VRAM usage and model quality, but the OptunaHub port samples it randomly, so the LLM never sees or reasons about it. The binary labeling also loses information: the LLM can't distinguish a config scoring 0.99 from one scoring 1.50, they're both "good" or both "bad" depending on the threshold.

We implemented a [faithful adaptation](autoresearch_automl/backends/llambo_original/) of the paper's code (LLAMBO (Paper), `--backend llambo_original`) alongside the OptunaHub version (LLAMBO (Optuna), `--backend llambo`) to quantify these differences. Both are included in our benchmark.

## Citation

```bibtex
@misc{ferreira2026autoresearchautoml,
    title={Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch},
    author={Fabio Ferreira and Lucca Wobbe and Arjun Krishnakumar and Arber Zela},
    year={2026},
    howpublished={\url{https://github.com/ferreirafabio/autoresearch-automl}},
}
```

