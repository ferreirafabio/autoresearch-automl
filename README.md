# autoresearch-automl

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM agent tweak training code through trial and error. Ravid Shwartz-Ziv [showed](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) that model-based optimization (Optuna TPE + expert-picked hyperparameter search space) already beats it. We fill the gap by integrating [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) into autoresearch, an approach that puts the LLM inside model-based optimization, using it as both surrogate model and candidate generator.

As an AutoML enthusiast, it felt natural to fill this void — this repo applies LLAMBO to Karpathy's autoresearch problem and benchmarks it against TPE.

![LLAMBO candidate sampling quality](assets/llambo_fig6.png)
*Figure 6 from [Ye et al. (2024)](https://arxiv.org/abs/2402.03921): LLAMBO outperforms TPE in candidate sampling quality, especially with few observed points.*

## Setup

We benchmark 7 HPO backends on Karpathy's autoresearch training task: single H200 GPU, 5-minute budget per trial, 24-hour training-time budget, minimize val_bpb.

**Zero-curation search space:** The search space (14 hyperparameters) is extracted fully automatically from `train.py` by parsing the AST — no manual HP selection or range tuning. The extractor finds all module-level uppercase constants with numeric or string values and infers types and ranges. This means any change to `train.py` (adding a new HP, renaming one) is picked up automatically. The search space is a property of the code, not an expert decision. Karpathy's `train.py` contains ~15 additional tunable values hardcoded inside functions (e.g., softcap, rotary embedding base, Muon momentum schedule) that our extractor intentionally does not expose — the unconstrained Karpathy agent baseline covers those.

**Backends:**
- **Classical:** Optuna TPE, Random Search, SMAC3, CMA-ES
- **LLM-based:** LLAMBO (OptunaHub), LLAMBO Original (paper-faithful), LLM Greedy
- **Hybrid:** Centaur (CMA-ES guided LLM optimization)

LLM-based backends use self-hosted Qwen3.5 (0.8B and 27B) via vLLM, running on the same GPU as training. No API keys, no proprietary models, fully reproducible. Each condition runs 3 seeds.

**Fair GPU memory allocation:** LLM backends share a single H200 (140 GB) between the vLLM inference server and the training process. With the 27B model, vLLM reserves ~45% of GPU memory, leaving ~76 GB for training. To ensure a fair comparison, we cap GPU memory for *all* backends — including classical ones — to the same 76 GB via `torch.cuda.set_per_process_memory_fraction()`. Without this cap, classical methods would silently exploit the full 140 GB, fitting deeper and wider models that LLM backends could never reach.

**Training-time budget:** We track cumulative *training time* (sum of `wall_time_seconds` across trials) rather than wall-clock time. LLM backends spend significant overhead on inference (e.g., LLAMBO ~50%), so a 24h wall-clock limit would give them far less actual training time. Every backend gets exactly 24 hours of GPU training, regardless of sampling overhead.

**Note on failure handling:** Infeasible configs (OOM, batch size assertion errors) are reported to the sampler as `val_bpb=100.0` instead of being silently dropped. Both TPE and LLAMBO otherwise ignore failed trials (`TrialState.FAIL`), which means they never learn to avoid bad regions. The penalty value is hardcoded for this task (real val_bpb ranges 0.99–2.4) — for other tasks, this would need adjustment.

## Results

### All Methods

Convergence curves (mean ± std across available seeds). Includes classical methods (TPE, CMA-ES, Random, SMAC), LLM-based (LLAMBO, LLM Greedy, Karpathy Agent), and hybrid (Centaur).

![All methods convergence](assets/exp2_all_convergence.png)

### 27B + Classical

27B LLM backends compared against classical methods. CMA-ES and TPE lead; Centaur is the best LLM-involving method.

![27B + Classical convergence](assets/exp2_27b_convergence.png)

### Incumbent Traces (seed 0)

Grey dots are all trials, colored dots are new bests, staircase is the incumbent (best-so-far).

![Incumbent Traces](assets/exp2_pareto_fronts.png)

### Centaur: CMA-ES Guided LLM Optimization

We introduce **Centaur**, a hybrid backend where CMA-ES acts as *critic* and an LLM acts as *actor*. CMA-ES runs every trial, learning the optimization landscape (covariance structure, convergence direction). On a fraction of trials (30%, after 10 warmup trials), the LLM receives CMA-ES's internal state — distribution mean, step-size sigma, top configs — and uses it alongside transformer domain knowledge to suggest configs. CMA-ES learns from all results, including LLM-suggested ones. See [centaur.md](centaur.md) for the full algorithm and related work comparison.

### Search Diversity Analysis

To understand *why* some methods outperform others, we measure how each backend explores the 13-dimensional continuous HP space. All values are normalized to [0,1] within their bounds. Only successful (non-OOM) trials are included.

**Metrics:**
- **Spread** — mean per-HP standard deviation (higher = more diverse sampling across each dimension)
- **Pairwise** — mean L2 distance between all config pairs (higher = configs are more different from each other)
- **Dist→Default** — mean L2 distance from Karpathy's default config (higher = exploring further from the starting point)
- **Step** — mean L2 distance between consecutive trials (higher = larger jumps between suggestions)
- **Cells** — unique cells when discretizing each HP into 5 bins (higher = more coverage of the search space)

| Method | Seeds | Avg Best | OOM% | Spread | Pairwise | Dist→Default | Step | Cells |
|--------|-------|----------|------|--------|----------|-------------|------|-------|
| cma_es | 2 | **0.9795** | 0% | 0.138 | 0.697 | 0.889 | 0.561 | 220 |
| optuna (TPE) | 2 | **0.9821** | 0% | 0.196 | 0.963 | 1.288 | 0.569 | 169 |
| centaur (27B) | 1 | **0.9848** | 0% | 0.126 | 0.611 | 1.064 | 0.541 | 88 |
| llambo_original (27B) | 3 | 0.9880 | 0% | 0.255 | 1.272 | 1.127 | 1.210 | 357 |
| random | 2 | 0.9893 | 56% | 0.274 | 1.388 | 1.243 | 1.391 | 169 |
| llambo (27B) | 3 | 0.9905 | 84% | 0.164 | 0.843 | 0.968 | 0.696 | 78 |
| llm_greedy (27B) | 3 | 0.9930 | 1% | 0.020 | 0.101 | 0.249 | 0.059 | 14 |
| smac | 2 | 1.0045 | 44% | 0.241 | 1.199 | 1.115 | 0.450 | 36 |

**Observations:**

- **LLM greedy has the lowest diversity by all metrics.** Spread 0.020 (14x less than random), only 14 unique grid cells, dist→default 0.249. It makes minimal changes between trials (step 0.059).
- **LLAMBO (OptunaHub) has 84% OOM rate** (up to 93% for seed 2), due to random categorical sampling of DEPTH.
- **llambo_original is the most diverse method with 0% OOM** — spread 0.255, 357 unique cells — yet still underperforms CMA-ES and TPE.
- **The top 3 methods (CMA-ES, TPE, Centaur) all have 0% OOM and moderate diversity** (spread 0.12–0.20).
- **SMAC has high spread (0.241) but only 36 unique cells** — it revisits similar configs while also producing 44% OOM.
- **Performance correlates more with OOM rate than with diversity.** All 0%-OOM methods outperform all high-OOM methods, suggesting that on this task, learning to avoid infeasible regions may matter more than LLM domain knowledge or search diversity.

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
