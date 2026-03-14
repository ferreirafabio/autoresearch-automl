# autoresearch-automl

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM agent tweak training code, run short experiments, and keep what works. At its core, the agent is doing hyperparameter search without a defined search space. Ravid Shwartz-Ziv [showed](https://www.linkedin.com/feed/update/urn:li:activity:7437556522240536576/) that Optuna TPE with 8 expert-picked hyperparameters already beats the LLM agent. Picking the right HPs matters more than LLM reasoning.

As an AutoML person, the natural next step is to put the LLM _inside_ the optimizer. [LLAMBO](https://arxiv.org/abs/2402.09359) does this: it replaces TPE's statistical surrogate with an LLM that has pretrained knowledge about training dynamics (learning rate schedules, batch size stability, depth vs compute tradeoffs). Same BO framework, but the surrogate actually understands what the hyperparameters mean. If LLAMBO matches TPE with fewer trials, LLMs can substitute for human domain expertise in HPO.

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
