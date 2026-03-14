# Comparison: Our Work vs. Shwartz-Ziv (2025)

## Reference

Ravid Shwartz-Ziv, ["Do LLM Coding agents fool us?"](https://www.linkedin.com/feed/update/urn:li:activity:7437556522240536576/), LinkedIn, 2025.

## Shwartz-Ziv's Experiment

Shwartz-Ziv compared Karpathy's LLM agent approach against Optuna TPE on nanochat:

- **Methods**: LLM agent vs. Optuna TPE (2 methods)
- **Search space**: 8 hand-picked HPs (expert-selected) and 23 HPs (Claude-selected)
- **Trials**: ~80
- **Codebase**: nanochat (multi-GPU)
- **Result**: TPE with 8 expert-selected HPs outperformed the LLM agent
- **Finding**: Expert domain knowledge in HP selection matters — 8 informed choices beat 23 blind ones

## Our Experiment

We extend this line of inquiry with a systematic multi-backend benchmark:

- **Methods**: 7 backends — random, Optuna TPE, SMAC, DEHB, BOHB, LLM greedy, LLAMBO
- **LLM model size study**: Qwen3.5-0.8B vs. Qwen3.5-9B as HP suggestion engines
- **Search space**: 14 HPs, auto-extracted from train.py via AST parsing
- **Trials**: 30 per backend (Experiment 1), 100 planned (Experiment 2)
- **Codebase**: autoresearch (single-GPU, 1×H200)
- **LLM serving**: Self-hosted via vLLM (no API dependency)
- **Failure feedback**: OOM/crash information fed back to all samplers

## Key Differences

| Dimension | Shwartz-Ziv | Ours |
|-----------|-------------|------|
| Backends compared | 2 (agent, TPE) | 7 (random, TPE, SMAC, DEHB, BOHB, LLM greedy, LLAMBO) |
| LLM as variable | No | Yes — model size comparison (0.8B vs 9B) |
| LLM provider | Proprietary API (likely Claude/GPT) | Self-hosted open-source (Qwen3.5 via vLLM) |
| Search space definition | Manual expert selection | Automated extraction from source code |
| Failure handling | Not discussed | Explicit — OOM/crash results fed to samplers |
| Hardware | nanochat, multi-GPU | autoresearch, single H200 |
| Reproducibility | No code/repo shared | Open-source framework with ask/tell API |

## What We Add

### 1. Systematic Multi-Backend Benchmark

Shwartz-Ziv's comparison is binary: LLM agent vs. TPE. We benchmark across the major families of HPO methods (random, model-based BO, evolutionary, multi-fidelity, LLM-based), enabling a more complete picture of where LLM-based HPO sits relative to established methods.

### 2. LLM Model Size as Experimental Variable

Does a larger LLM produce better HP suggestions? We test this directly by comparing Qwen3.5-0.8B against Qwen3.5-9B as the suggestion engine, holding everything else constant. Early results suggest the smaller model performs comparably or better — a relevant finding for the cost-effectiveness of LLM-assisted HPO.

### 3. Open-Source Reproducibility

By using self-hosted open-source LLMs (Qwen3.5) rather than proprietary APIs, our experiments are fully reproducible without API access or cost constraints. The framework is designed for others to plug in additional backends or LLMs.

### 4. Automated Search Space Construction

Shwartz-Ziv's key advantage was expert HP selection (8 HPs chosen by a domain expert). Our search space is extracted automatically from the training script — testing whether the framework can work without manual HP curation. This matters for new domains (e.g., code generation, RL) where practitioners may lack deep tuning expertise.

### 5. Failure-Aware Optimization

Trials that crash (OOM, NaN) provide signal about which regions of the search space are infeasible. We explicitly feed crash information back to all backends, enabling samplers to learn the feasible region rather than repeatedly suggesting configs that will OOM.

## Central Hypothesis

Can an LLM provide "informed choices" without a human expert? LLAMBO is the most direct test: it uses the LLM as a surrogate model inside Bayesian optimization, replacing TPE's statistical density estimator with an LLM that brings pretrained knowledge about transformer training dynamics. If LLAMBO matches or beats expert-curated TPE with fewer trials, that is evidence that LLM-based surrogate models can substitute for human domain expertise in HPO — which matters most for new problem domains where that expertise does not yet exist.

## Open Questions

- **Trial budget**: Shwartz-Ziv ran ~80 trials; we start with 30. At 5 min/trial, 100+ trials are feasible within a 12h window. More trials would reveal whether LLM-based methods catch up or diverge further from classical methods.
- **Compute cost**: LLM-based suggestions add inference overhead (vLLM serving cost + latency). Is the improvement worth the extra compute vs. running more TPE trials in the same wall-clock time?
- **Generalization**: Both experiments test on transformer pretraining. The real promise of LLM-assisted HPO is on problems where domain knowledge is scarce. Testing on diverse tasks would strengthen or weaken the case for LLM-based approaches.
