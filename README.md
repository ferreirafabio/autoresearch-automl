<h2 align="center"><code>autoresearch-automl</code></h2>
<h3 align="center">Can LLMs Beat Classical Hyperparameter Optimization Algorithms?<br>A Study on <i>autoresearch</i></h3>

> Under anonymous review at NeurIPS 2026.

![Classical vs LLM-based HPO](assets/exp2_27b_walltime.png)

## Introduction

[autoresearch](https://github.com/karpathy/autoresearch) enables an LLM agent to optimize hyperparameters by editing training code directly. We use it as a testbed to compare classical HPO algorithms against LLM-based methods on tuning the hyperparameters of a small language model under a fixed compute budget. When defining a fixed search space, classical methods such as CMA-ES and TPE consistently outperform LLM-based agents, where avoiding out-of-memory failures matters more than search diversity. Allowing the LLM to directly edit source code narrows the gap to the classical methods but does not close it, even with frontier models available at the time of writing such as Claude Opus 4.6 and Gemini 3.1 Pro Preview. We observe that LLMs struggle to track optimization state across trials. In contrast, classical methods lack the domain knowledge of LLMs. To combine the strengths of both, we introduce Centaur, a hybrid that shares CMA-ES's interpretable internal state with the LLM. Centaur achieves the best result in our experiments, and a 0.8B LLM already suffices to outperform all classical and pure LLM methods. Unconstrained code editing requires larger models to be competitive with classical methods. All in all, our results suggest that LLMs are most effective as a complement to classical optimizers, not as a replacement. We benchmark 9 methods, 4 classical, 4 LLM-based, and 1 hybrid, all under the same 24-hour GPU training budget with 3 seeds.

## Methods

**Classical (fixed [14-HP search space](#search-space)):**
- **TPE:** Tree-structured Parzen Estimator ([Optuna](https://github.com/optuna/optuna)).
- **CMA-ES:** Covariance Matrix Adaptation Evolution Strategy ([Optuna CMA sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html)).
- **SMAC:** Sequential Model-based Algorithm Configuration with Random Forest surrogate ([SMAC3](https://github.com/automl/SMAC3)).
- **Random:** Uniform random sampling ([Optuna RandomSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html)).

**LLM-based (fixed [14-HP search space](#search-space)):**
- **LLAMBO (Optuna):** LLM as surrogate + candidate generator inside Bayesian optimization ([OptunaHub port](https://hub.optuna.org/samplers/llambo/)). Uses binary surrogate labels, delegates categorical HPs to random sampling, and hides failed trials from the surrogate (see [Details](#llambo-optuna-vs-llambo-paper)).
- **LLAMBO (Paper):** Our reimplementation faithful to the original paper: continuous surrogate labels, all HPs visible to the LLM, failed trials included ([Liu et al., 2024](https://arxiv.org/abs/2402.03921)).
- **Karpathy Agent (14 HPs):** LLM sees trial history and suggests the next config within the fixed search space.

**LLM-based (unconstrained search space):**
- **Karpathy Agent (Code):** LLM directly edits `train.py` source code each trial ([Karpathy's autoresearch](https://github.com/karpathy/autoresearch)).

**Hybrid (fixed [14-HP search space](#search-space)):**
- **Centaur (CMA-ES+LLM):** CMA-ES runs every trial; on 30% of trials, the LLM receives CMA-ES's internal state and suggests a config. CMA-ES updates from all results, including LLM-suggested ones. See [centaur.md](centaur.md).

Our main experiments use self-hosted Qwen3.5 (0.8B and 27B) as the LLM optimizer via vLLM on the same GPU that trains the optimizee (~50M parameter language model). We additionally run frontier model experiments with Gemini 3.1 Pro Preview via the Gemini API and Claude Opus 4.6 via the Claude API.

## Experimental Setup

Single H200 GPU, 5 min/trial, minimize val_bpb. Search space: 14 HPs auto-extracted from `train.py` via [AST](https://docs.python.org/3/library/ast.html) parsing (every `ALL_CAPS = literal` assignment becomes a tunable HP). All methods get 24 hours of GPU training time (excluding LLM inference overhead), capped to ~76 GB VRAM (to match the H100 used in Karpathy's and Shwartz Ziv's experiments). Failed trials reported as `val_bpb=100.0` so optimizers learn to avoid OOM regions. 3 seeds per condition.

## Results

### Classical methods outperform LLMs in fixed search spaces

Within the fixed search space, classical HPO methods consistently outperform LLM-based agents. The gap to the best fixed-space LLM method (LLAMBO Paper at 0.9862) is substantial, and several pure LLM methods perform worse than random search, indicating that restricting LLMs to a fixed HP search space does not leverage their strengths. OOM avoidance matters more than search diversity: the top methods all keep OOM rates below 16%, while the bottom four exceed 36%. Notably, LLAMBO observes full trial history yet produces OOM rates (48-61%) comparable to random search (56%), suggesting that small/mid-sized LLMs fail to learn which regions cause memory failures.

Best in each category is highlighted: <span style="background-color:#0072B2;color:white;padding:1px 6px;border-radius:3px">hybrid</span> <span style="background-color:#E69F00;color:black;padding:1px 6px;border-radius:3px">classical</span> <span style="background-color:#009E73;color:white;padding:1px 6px;border-radius:3px">pure LLM</span> (Okabe-Ito colorblind-safe).

| Method | Seeds | Best val_bpb | OOM% |
|--------|-------|-------------|------|
| <span style="background-color:#0072B2;color:white">**Centaur [Opus 4.6]**</span> | 3 | **0.9739 ± 0.0012** | 17% |
| Centaur [Qwen 27B] | 3 | 0.9763 ± 0.0005 | 15% |
| Centaur [Qwen 0.8B] | 3 | 0.9766 ± 0.0008 | 13% |
| Centaur [Gemini 3.1 Pro] | 3 | 0.9767 ± 0.0013 | 20% |
| <span style="background-color:#E69F00;color:black">**TPE**</span> | 3 | **0.9768 ± 0.0019** | 11% |
| <span style="background-color:#009E73;color:white">**Karpathy Agent (Code) [Opus 4.6]**</span> | 3 | **0.9770 ± 0.0027** | 5% |
| SMAC | 3 | 0.9778 ± 0.0020 | 36% |
| CMA-ES | 3 | 0.9785 ± 0.0036 | 16% |
| Karpathy Agent (Code) [Qwen 27B] | 3 | 0.9814 ± 0.0046 | 12% |
| Karpathy Agent (Code) [Gemini 3.1 Pro] | 3 | 0.9826 ± 0.0004 | 3% |
| LLAMBO (Paper) [Qwen 27B] | 3 | 0.9862 ± 0.0041 | 48% |
| Random | 3 | 0.9873 ± 0.0021 | 56% |
| LLAMBO (Optuna) [Qwen 27B] | 3 | 0.9882 ± 0.0012 | 61% |
| Karpathy Agent (14 HPs) [Qwen 27B] | 3 | 0.9904 ± 0.0002 | 1% |
| Karpathy Agent (Code) [Qwen 0.8B] | 3 | 0.9910 ± 0.0001 | 19% |

### Unconstrained code editing is viable but requires model scale

Karpathy Agent (Code), which directly edits training source code, is the only pure LLM method competitive with classical approaches. Experiments with frontier models Gemini 3.1 Pro Preview and Claude Opus 4.6 narrow the gap further (Opus 4.6: 0.9770 ± 0.0027, competitive with CMA-ES at 0.9785) but do not close it to the best classical methods like TPE (0.9768). Frontier model capability also shows up in the OOM rate: Karpathy Agent (Code)'s failure rate drops sharply with LLM capability (19% for Qwen 0.8B, 12% for Qwen 27B, 3% for Gemini 3.1 Pro, 5% for Opus 4.6), whereas Centaur's stays in a narrow range (13-20%) across model choices, indicating that CMA-ES dominates OOM avoidance in the hybrid.

### Frontier model comparison

Frontier models (Claude Opus 4.6, Gemini 3.1 Pro Preview) vs Qwen3.5-27B for Karpathy Agent (Code) and Centaur. Centaur with Opus 4.6 achieves the best result (0.9739). TPE shown as classical reference.

![Gemini 3.1 Pro Preview vs Qwen3.5-27B](assets/exp2_gemini_frontier.png)

### 0.8B vs 27B LLM optimizer

Scaling the LLM from 0.8B to 27B is essential for unconstrained code editing (0.9910 vs 0.9814) but provides no advantage for fixed-HP optimization. Solid lines = 27B, dashed = 0.8B.

![0.8B vs 27B comparison](assets/exp2_all_walltime.png)

### Hybrid optimization: best of both worlds

Centaur outperformed all methods including CMA-ES alone by using the LLM on only 30% of trials. The LLM receives CMA-ES's full internal state (mean vector, step-size, covariance matrix), the top-5 configurations, and the last 20 trials. A 0.8B LLM already suffices to outperform all classical and pure LLM methods. Scaling from 0.8B (0.9766) to 27B (0.9763) to Gemini Pro (0.9767) yields no improvement, suggesting a capability plateau. However, Centaur with Claude Opus 4.6 breaks through this plateau to 0.9739, with the improvement coming from higher-quality suggestions rather than better OOM avoidance (17% vs 15% for Qwen 27B).

We ablate the LLM ratio: higher ratios degrade performance, confirming that CMA-ES should retain majority control. See [centaur.md](centaur.md) for the full algorithm.

![Centaur LLM Ratio Ablation](assets/centaur_ratio_ablation.png)

### Incumbent Traces

Grey dots are all trials, colored dots are new bests, staircase is the incumbent (best-so-far). Each panel shows the best seed for that method.

**Classical + Hybrid:**

![Incumbent Traces — Classical + Hybrid](assets/exp2_incumbents_classical.png)

**LLM-based:**

![Incumbent Traces — LLM-based](assets/exp2_incumbents_llm.png)

## Search Space

14 hyperparameters auto-extracted via AST parsing (every `ALL_CAPS = literal` assignment in `train.py`):

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
- [LLAMBO (Liu et al., 2024)](https://arxiv.org/abs/2402.03921) for using LLMs as surrogate models in Bayesian optimization

## Details

### H200 vs H100 baseline

Our baseline (Karpathy's default config) achieves val_bpb ≈ 0.991 on H200 at full clock speed (~1750K tokens/s), comparable to Karpathy's ~0.998 on H100. Early runs showed a higher baseline of ~1.008 due to GPU power throttling. All results reported here use non-throttled H200s.

### LLM system prompts

The system prompts used by each LLM-based method include the optimization goal (minimize val_bpb), the model class and training stack (GPT-2 scale transformer, Muon+AdamW optimizer), the dataset (FineWeb), the hardware constraints and OOM warning, the search space with bounds, and the trial history. Full prompt templates are reproduced verbatim in the paper appendix (§LLM Prompts and Problem Context) and live in source here:

- Karpathy Agent (Code): [`karpathy_agent_backend.py`](autoresearch_automl/backends/karpathy_agent_backend.py) — `AGENT_PROMPT`
- Karpathy Agent (14 HPs): [`karpathy_agent_hps_backend.py`](autoresearch_automl/backends/karpathy_agent_hps_backend.py) — `SUGGEST_PROMPT`
- Centaur: [`centaur_backend.py`](autoresearch_automl/backends/centaur_backend.py) — `SUGGEST_PROMPT`
- LLAMBO (Paper): [`llambo_original_backend.py`](autoresearch_automl/backends/llambo_original_backend.py) — `DEFAULT_TASK_DESCRIPTION` + [`llambo_original/acquisition_function.py`](autoresearch_automl/backends/llambo_original/acquisition_function.py)
- LLAMBO (Optuna): [`llambo_backend.py`](autoresearch_automl/backends/llambo_backend.py) — `DEFAULT_TASK_DESCRIPTION`

### LLAMBO (Optuna) vs LLAMBO (Paper)

The [OptunaHub LLAMBO sampler](https://hub.optuna.org/samplers/llambo/) ([Ozaki et al., 2025](https://arxiv.org/abs/2510.02798)) differs from the [original paper code](https://github.com/tennisonliu/LLAMBO) in several ways that materially affect optimization quality:

| Aspect | Original paper | OptunaHub port |
|--------|---------------|----------------|
| **Surrogate labels** | Actual metric values, LLM sees performance gradients | Binary 0/1 (top 20% threshold) |
| **Categorical HPs** | All HPs included in LLM prompts | Categoricals delegated to random sampling |
| **Failed trials** | Visible to surrogate (can learn infeasible regions) | Hidden from surrogate |

We implemented a [faithful adaptation](autoresearch_automl/backends/llambo_original/) of the paper's code alongside the OptunaHub version to quantify these differences.

## Citation

```bibtex
@article{ANON2026autoresearchautoml,
    title={Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch},
    author={XXXX-1 and XXXX-2 and XXXX-3 and XXXX-4 and XXXX-5},
    year={2026},
    note={Under anonymous review at NeurIPS 2026},
}
```
