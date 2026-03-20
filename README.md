# autoresearch-automl: When AutoML Meets autoresearch - Classical HPO, LLM Agents, and Hybrid Methods for Language Model Training

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM agent edit training code through trial and error, with no fixed search space, just code diffs. [Shwartz-Ziv showed](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) that a classical AutoML method (TPE + expert HPs) can beat it. This makes autoresearch an excellent in-the-wild testbed to assess classical AutoML/HPO methods against newer LLM-based (agent) methods. We extend Karpathy's and Shwartz-Ziv's experiments with a more extensive classical HPO vs. LLM-based comparison. We compare classical HPO (TPE, CMA-ES, SMAC, Random Search) and LLM-based HPO ([LLAMBO](https://arxiv.org/abs/2402.03921), Karpathy Agent), all under fair conditions.

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

**Fairness:** All methods get 24 hours of GPU training time (excluding LLM inference overhead), capped to ~80 GB VRAM (to match the H100 used in Karpathy's and Shwartz-Ziv's experiments). Failed trials reported as `val_bpb=100.0` so samplers learn to avoid OOM regions. Results are trimmed to 300 trials, as no meaningful improvement occurs beyond that point.

## Results

### All Methods

Convergence curves (mean ± std across available seeds) by cumulative training wall-time. All LLM methods use Qwen3.5-27B. Includes classical methods (TPE, CMA-ES, Random, SMAC), LLM-based (LLAMBO, Karpathy Agent), and hybrid (Centaur).

![HPO Convergence (wall-time)](assets/exp2_27b_walltime.png)

Same data by trial number (sample efficiency view):

![HPO Convergence (by trial)](assets/exp2_27b_convergence.png)

### 0.8B vs 27B LLM Optimizer

Does LLM size matter for HPO? CMA-ES (best classical) shown as reference. Solid lines = 27B, dashed = 0.8B.

![Model size comparison (wall-time)](assets/exp2_all_walltime.png)

Same data by trial number:

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
| CMA-ES | 3 | **0.9833** | 16% | 0.138 | 0.697 | 0.889 | 0.561 | 220 |
| Centaur (CMA-ES+LLM) [27B] | 3 | **0.9821** | 19% | 0.126 | 0.611 | 1.064 | 0.541 | 88 |
| TPE | 3 | **0.9840** | 10% | 0.196 | 0.963 | 1.288 | 0.569 | 169 |
| LLAMBO (Paper) [27B] | 3 | 0.9880 | 48% | 0.255 | 1.272 | 1.127 | 1.210 | 357 |
| Random | 3 | 0.9898 | 57% | 0.274 | 1.388 | 1.243 | 1.391 | 169 |
| LLAMBO (Optuna) [27B] | 3 | 0.9903 | 79% | 0.164 | 0.843 | 0.968 | 0.696 | 78 |
| Karpathy Agent (14 HPs) [27B] | 3 | 0.9930 | 1% | 0.020 | 0.101 | 0.249 | 0.059 | 14 |
| Karpathy Agent (Code) [27B] | 2 | 0.9936 | 11% | - | - | - | - | - |
| SMAC | 3 | 1.0015 | 48% | 0.241 | 1.199 | 1.115 | 0.450 | 36 |

**Observations:**

- **Karpathy Agent (14 HPs) has the lowest diversity by all metrics.** Spread 0.020 (14x less than random), only 14 unique grid cells, dist→default 0.249. It makes minimal changes between trials (step 0.059).
- **LLAMBO (Optuna) has 84% OOM rate** (up to 93% for seed 2), due to random categorical sampling of DEPTH.
- **LLAMBO (Paper) is the most diverse method with 0% OOM** (spread 0.255, 357 unique cells), yet still underperforms CMA-ES and TPE. Notably, LLAMBO (Paper) achieves higher coverage than Random Search (357 vs 169 unique cells) despite being model-based.
- **The top 3 methods (CMA-ES, TPE, Centaur) all have 0% OOM and moderate diversity** (spread 0.12–0.20).
- **SMAC has high spread (0.241) but only 36 unique cells.** Its GP surrogate + Expected Improvement acquisition keeps exploring OOM regions despite penalty costs, unlike TPE which directly models feasibility. Even after fixing a status bug (MEMORYOUT instead of SUCCESS), OOM rate remains ~60%.
- **Performance correlates more with OOM rate than with diversity.** All 0%-OOM methods outperform all high-OOM methods, suggesting that on this task, learning to avoid infeasible regions may matter more than LLM domain knowledge or search diversity.

## Search Space

Classical, non-LLM-based methods work with search spaces. The quality of the search space greatly affects the results produced by these methods. To make the comparison with Karpathy's autoresearch fair, we need to eliminate human priors that we would otherwise encode into the search space. To do this, we automatically extract hyperparameters from `train.py` using [AST](https://docs.python.org/3/library/ast.html) (Abstract Syntax Tree) parsing: the source code is parsed into a syntax tree, and every top-level `ALL_CAPS = literal` assignment is identified as a tunable HP. We extract the following 14 hyperparameters to optimize (13 continuous/integer + 1 categorical):

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

# Karpathy Agent (14 HPs)
python -m autoresearch_automl.cli run --backend llm_greedy --trials 100 --llm-model Qwen3.5-0.8B
```

## Related work

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the training task and the idea of LLM-driven experimentation
- [Ravid Shwartz-Ziv](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) for showing that expert HP selection beats blind LLM search
- [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) for using LLMs as surrogate models in Bayesian optimization

## Details

### H200 vs H100 baseline offset

Our baseline (Karpathy's default config) achieves val_bpb=1.008 on H200, while Karpathy reports ~0.998 on H100. Same code, same config, same `torch.compile` + FA3. The gap is entirely due to **GPU power throttling**: H200's HBM3e memory (4.8 TB/s) draws more power than H100's HBM3 (3.35 TB/s), leaving less of the shared 700W TDP for the SMs. Under sustained load, our H200 clocks down to ~1600 MHz (81% of max 1980 MHz), yielding 18% fewer training steps per 5-minute trial. Adjusting for actual clock speed, compute efficiency is identical (40.3% vs 39.8% MFU). This baseline offset does not affect HPO convergence.

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

## Acknowledgements

Thanks to Arjun Krishnakumar and Arber Zela for his feedback.
