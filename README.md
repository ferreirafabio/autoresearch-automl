# autoresearch-automl

When I saw Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) post, my immediate thought was: how would classical HPO/AutoML do on the same problem? Ravid Shwartz-Ziv [had the same idea](https://www.linkedin.com/feed/update/urn:li:activity:7437556522240536576/) and showed that Optuna TPE with 8 expert-picked hyperparameters beats the LLM agent. Domain knowledge in HP selection matters more than LLM reasoning.

But that raises a more interesting question. What if we put the LLM _inside_ the optimizer, where its knowledge about ML training dynamics actually helps?

That is what [LLAMBO](https://arxiv.org/abs/2402.09359) does. Instead of prompting an LLM to suggest configs (Karpathy's approach), LLAMBO uses the LLM as the surrogate model in Bayesian optimization. The LLM knows from pretraining that high learning rates with large batch sizes are unstable, or that very deep transformers on short budgets undertrain. It brings that knowledge into the BO loop directly.

So the interesting comparison becomes TPE vs LLAMBO. TPE builds a density model from trial history but has no idea what a learning rate means. LLAMBO uses the same BO framework but with a surrogate that understands transformer training dynamics. If LLAMBO matches TPE with fewer trials, that is evidence that LLMs can substitute for human domain expertise in HPO.

I am most excited about what this means for new problem domains (code generation, RL, multimodal) where we do not have decades of tuning intuition yet. At the same time, the open question is whether the compute overhead of running an LLM inside the optimizer is worth it, or whether you are better off just running more TPE trials in the same wall-clock time.

## What we benchmark

7 HPO backends on Karpathy's autoresearch training task (single GPU, 5 min budget per trial, minimize val_bpb):

| Backend | Family | Uses LLM? |
|---------|--------|-----------|
| Random | Baseline | No |
| Optuna TPE | Bayesian (density estimation) | No |
| SMAC | Bayesian (random forest surrogate) | No |
| DEHB | Evolutionary + multi-fidelity | No |
| BOHB | Bayesian + multi-fidelity | No |
| LLM Greedy | LLM reasoning (Karpathy's approach) | Yes |
| LLAMBO | LLM as BO surrogate model | Yes |

All LLM-based methods use the same self-hosted open source model (Qwen3.5-0.8B via vLLM) so results are reproducible without API access or cost.

## Experiments

### Experiment 1: Does LLM size matter for HP suggestions?

We compare Qwen3.5-0.8B vs Qwen3.5-9B as the LLM backend for LLM Greedy. Same search space, same trial budget. Early results: the 0.8B model performs comparably or better than 9B (1.06 vs 1.13 val_bpb after 10 trials). A smaller model leaves more GPU memory for training and costs less to serve.

### Experiment 2: Classical AutoML vs LLM-assisted HPO

All 7 backends, 100 trials each, 3 seeds. The central question: does an LLM surrogate (LLAMBO) outperform purely statistical methods (TPE, SMAC) that have no understanding of what the hyperparameters mean?

## Key design decisions

**Automated search space.** The 14 hyperparameters and their ranges are extracted automatically from train.py via AST parsing. No manual HP curation. Ravid Shwartz-Ziv showed that expert-picked HPs matter. We deliberately avoid expert curation to test whether LLAMBO can compensate through pretrained knowledge.

**Self-hosted LLM.** Everything runs on a single H200. vLLM serves Qwen3.5-0.8B in the background (10% GPU memory), training uses the rest. No API keys, no proprietary models, fully reproducible.

**Failure feedback.** When a trial OOMs or crashes, that is signal. We feed crash information back to all backends so they learn which regions of the search space are infeasible.

## Architecture

```
CLI (click)
 |
 v
Runner (ask/tell loop)
 |
 +-- Backend.suggest()  -->  ConfigInjector (AST-based)  -->  train.py
 |                                                              |
 +-- Backend.tell()    <--  ObjectiveFunction (parse metrics) <-+
 |
 +-- ResultsDB (JSONL, resume/checkpoint)
```

All backends implement the same `HPOBackend` interface (`configure`, `suggest`, `tell`). Adding a new optimizer is just implementing those three methods.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,all]"
```

For LLM-based backends, download a model and start vLLM:
```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
vllm serve models/Qwen3.5-0.8B --host 127.0.0.1 --port 8000 --dtype bfloat16
```

## Usage

```bash
# Run HPO with a specific backend
python -m autoresearch_automl.cli run --backend optuna --trials 100 --seed 0

# LLM-based backend (requires vLLM running)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
python -m autoresearch_automl.cli run --backend llambo --trials 100 --llm-model Qwen3.5-0.8B

# Analyze results
python -m autoresearch_automl.cli analyze --results-dir results/ --output-dir plots/
```

## Related work

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the training task and the idea of LLM-driven experimentation
- [Ravid Shwartz-Ziv](https://www.linkedin.com/feed/update/urn:li:activity:7437556522240536576/) for showing that expert HP selection beats blind LLM search
- [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.09359) for using LLMs as surrogate models in Bayesian optimization
