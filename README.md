# autoresearch-automl

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM agent tweak training code through trial and error. Ravid Shwartz-Ziv [showed](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) that model-based optimization (Optuna TPE + expert-picked hyperparameter search space) already beats it. We fill the gap by integrating [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) into autoresearch, an approach that puts the LLM inside model-based optimization, using it as both surrogate model and candidate generator.

As an AutoML enthusiast, it felt natural to fill this void — this repo applies LLAMBO to Karpathy's autoresearch problem and benchmarks it against TPE.

![LLAMBO candidate sampling quality](assets/llambo_fig6.png)
*Figure 6 from [Ye et al. (2024)](https://arxiv.org/abs/2402.03921): LLAMBO outperforms TPE in candidate sampling quality, especially with few observed points.*

## Setup

We benchmark 7 HPO backends on Karpathy's autoresearch training task: single H200 GPU, 5-minute budget per trial, minimize val_bpb. The search space (14 hyperparameters) is extracted automatically from train.py by parsing the source code — no manual HP curation.

**Backends:**
- **Classical:** Optuna TPE, Random Search, SMAC3, CMA-ES
- **LLM-based:** LLAMBO (OptunaHub), LLAMBO Original (paper-faithful), LLM Greedy

LLM-based backends use self-hosted Qwen3.5 (0.8B and 27B) via vLLM, running on the same GPU as training. No API keys, no proprietary models, fully reproducible. Each condition runs 3 seeds.

**Note on failure handling:** Infeasible configs (OOM, batch size assertion errors) are reported to the sampler as `val_bpb=100.0` instead of being silently dropped. Both TPE and LLAMBO otherwise ignore failed trials (`TrialState.FAIL`), which means they never learn to avoid bad regions. The penalty value is hardcoded for this task (real val_bpb ranges 0.99–2.4) — for other tasks, this would need adjustment.

## Results

### 0.8B: TPE vs LLAMBO vs LLM Greedy

TPE dominates with 0.8B — reaches ~0.978 by trial 40. LLAMBO (OptunaHub) and LLM Greedy plateau around 1.00 with high variance. LLAMBO 0.8B has a 70% failure rate due to random categorical sampling producing OOM configs.

![0.8B convergence](assets/exp2_0.8b_convergence.png)

### 27B: TPE vs LLAMBO vs LLM Greedy

Scaling to 27B improves all LLM-based backends. LLAMBO Original 27B reaches ~0.989 — the best LLM method, only 1.3% behind TPE. LLM Greedy 27B is competitive at ~0.995 with a 99% success rate.

![27B convergence](assets/exp2_27b_convergence.png)

### LLM-Based vs Classical HPO

Dashed lines are 0.8B, solid lines are 27B.

![LLM-Based vs Classical HPO](assets/exp2_all_convergence.png)

### Incumbent Traces (seed 0)

Grey dots are all trials, colored dots are new bests, staircase is the incumbent (best-so-far).

![Incumbent Traces](assets/exp2_pareto_fronts.png)

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

# LLAMBO (requires vLLM running)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
python -m autoresearch_automl.cli run --backend llambo --trials 100 --llm-model Qwen3.5-0.8B

# LLAMBO Original (paper-faithful)
python -m autoresearch_automl.cli run --backend llambo_original --trials 100 --llm-model Qwen3.5-0.8B

# LLM Greedy
python -m autoresearch_automl.cli run --backend llm_greedy --trials 100 --llm-model Qwen3.5-0.8B
```

## Notes

### H200 vs H100 baseline offset

Our baseline (Karpathy's default config) achieves val_bpb=1.008 on H200, while Karpathy reports ~0.998 on H100. Same code, same config, same `torch.compile` + FA3. The gap is entirely due to **GPU power throttling**: H200's HBM3e memory (4.8 TB/s) draws more power than H100's HBM3 (3.35 TB/s), leaving less of the shared 700W TDP for the SMs. Under sustained load, our H200 clocks down to ~1600 MHz (81% of max 1980 MHz), yielding 18% fewer training steps per 5-minute trial. Adjusting for actual clock speed, compute efficiency is identical (40.3% vs 39.8% MFU). This baseline offset does not affect HPO convergence.

### OptunaHub LLAMBO vs Original Paper

While integrating LLAMBO, we discovered that the [OptunaHub LLAMBO sampler](https://hub.optuna.org/samplers/llambo/) differs from the [original paper code](https://github.com/tennisonliu/LLAMBO) in several ways that materially affect optimization quality. OptunaHub does great work making research accessible — these notes are meant to help users who need paper-faithful behavior.

**Key differences:**

| Aspect | Original paper | OptunaHub port |
|--------|---------------|----------------|
| **Surrogate labels** | Actual metric values (`## 0.970 ##`) — LLM sees performance gradients | Binary 0/1 (top 20% threshold) — LLM only sees "good" vs "bad" |
| **Categorical HPs** | All HPs included in LLM prompts | Categoricals delegated to random sampling, invisible to LLM |
| **Failed trials** | Visible to surrogate (can learn infeasible regions) | Marked as `TrialState.FAIL`, invisible to surrogate |

**Impact on our experiments:** The categorical delegation was the most painful. Our `WINDOW_PATTERN` hyperparameter (attention pattern per layer) strongly affects VRAM usage and model quality, but the OptunaHub port samples it randomly — the LLM never sees or reasons about it. The binary labeling also loses information: the LLM can't distinguish a config scoring 0.99 from one scoring 1.50, they're both "good" or both "bad" depending on the threshold.

We implemented a [faithful adaptation](autoresearch_automl/backends/llambo_original/) of the paper's code (`--backend llambo_original`) alongside the OptunaHub version (`--backend llambo`) to quantify these differences. Both are included in our benchmark.

## Related work

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the training task and the idea of LLM-driven experimentation
- [Ravid Shwartz-Ziv](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) for showing that expert HP selection beats blind LLM search
- [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) for using LLMs as surrogate models in Bayesian optimization
