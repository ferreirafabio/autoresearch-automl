# autoresearch-automl

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) is a neat idea: let an LLM agent tweak training code, run short experiments, keep what works. But at its core, the agent is doing hyperparameter search without a defined search space. Ravid Shwartz-Ziv [tested this](https://www.linkedin.com/feed/update/urn:li:activity:7437556522240536576/) and found that Optuna TPE with just 8 expert-picked hyperparameters already beats the LLM agent on nanochat. So the LLM reasoning is not the bottleneck. Picking the right hyperparameters is.

As an AutoML person, the naturally arising next step is to put the LLM where it can actually help: inside the optimizer. That is exactly what [LLAMBO](https://arxiv.org/abs/2402.09359) does. Instead of prompting an LLM to suggest configs (Karpathy's approach), LLAMBO uses the LLM as the surrogate model in Bayesian optimization. The LLM knows from pretraining that high learning rates with large batch sizes are unstable, or that very deep transformers on short budgets undertrain. It brings that knowledge into the BO loop directly.

So the interesting comparison becomes TPE vs LLAMBO. TPE builds a density model from trial history but has no idea what a learning rate means. LLAMBO uses the same BO framework but with a surrogate that understands transformer training dynamics. If LLAMBO matches TPE with fewer trials, that is evidence that LLMs can substitute for human domain expertise in HPO.

## Setup

I run TPE and LLAMBO on Karpathy's autoresearch training task: single GPU, 5 min budget per trial, minimize val_bpb. The search space (14 hyperparameters) is extracted automatically from train.py via AST parsing. No manual HP curation. Ravid Shwartz-Ziv showed that expert-picked HPs matter. I deliberately avoid expert curation to test whether LLAMBO can compensate through pretrained knowledge.

LLAMBO uses self-hosted open source LLMs (Qwen3.5 via vLLM), running on the same GPU as training. I also compare different LLM sizes to check whether a larger model actually produces better suggestions. No API keys, no proprietary models, fully reproducible.

## Usage

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,all]"

# TPE
python -m autoresearch_automl.cli run --backend optuna --trials 100 --seed 0

# LLAMBO (requires vLLM running)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
python -m autoresearch_automl.cli run --backend llambo --trials 100 --llm-model Qwen3.5-0.8B
```

## Related work

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the training task and the idea of LLM-driven experimentation
- [Ravid Shwartz-Ziv](https://www.linkedin.com/feed/update/urn:li:activity:7437556522240536576/) for showing that expert HP selection beats blind LLM search
- [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.09359) for using LLMs as surrogate models in Bayesian optimization
