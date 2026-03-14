# autoresearch-automl

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM agent tweak training code, run short experiments, and keep what works. At its core, the agent is doing hyperparameter search without a defined search space. Ravid Shwartz-Ziv [showed](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) that Optuna TPE with 8 expert-picked hyperparameters already beats the LLM agent. Picking the right HPs matters more than LLM reasoning.

We know from [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) that using an LLM as a surrogate in Bayesian optimization can outperform GP-based BO in low-trial regimes (see figure below), because the LLM brings pretrained knowledge about training dynamics (learning rate schedules, batch size stability, depth vs compute tradeoffs).

![LLAMBO candidate sampling quality](assets/llambo_fig6.png)
*Figure 6 from [Ye et al. (2024)](https://arxiv.org/abs/2402.03921): LLAMBO outperforms TPE in candidate sampling quality, especially with few observed points.*

As an AutoML enthusiast, it felt natural to fill this void and apply LLAMBO to Karpathy's autoresearch problem.

## Setup

I run [Optuna](https://github.com/optuna/optuna) TPE and [LLAMBO via OptunaHub](https://hub.optuna.org/samplers/llambo/) on Karpathy's autoresearch training task: single GPU, 5 min budget per trial, minimize val_bpb. The search space (14 hyperparameters) is extracted automatically from train.py by parsing the source code for ALL_CAPS variable assignments. No manual HP curation. Ravid Shwartz-Ziv showed that expert-picked HPs matter. I deliberately avoid expert curation to test whether LLAMBO can compensate through pretrained knowledge.

LLAMBO uses self-hosted open source LLMs (Qwen3.5 via vLLM), running on the same GPU as training. I also compare different LLM sizes to check whether a larger model actually produces better suggestions. No API keys, no proprietary models, fully reproducible.

## Results (in progress)

### TPE vs LLAMBO

LLAMBO converges faster in the first ~20 trials. TPE catches up given enough budget. Experiments still running (24h Slurm jobs, 3 seeds each).

![TPE vs LLAMBO convergence](assets/exp2_tpe_vs_llambo.png)

### LLM model size

0.8B finds good configs from the start. 9B starts worse but converges to the same level after ~15 trials. Bigger is not better for HP suggestions.

![LLM model size comparison](assets/exp1_model_size.png)

### Progress (Karpathy-style)

Same format as [Karpathy's progress plot](https://github.com/karpathy/autoresearch/blob/master/progress.png). Grey dots are discarded trials, colored dots are new bests, staircase is the running best.

![Progress plot](assets/exp2_progress.png)

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
- [Ravid Shwartz-Ziv](https://www.linkedin.com/posts/ravid-shwartz-ziv-8bb18761_do-llm-coding-agents-fool-us-karpathys-activity-7437556522240536576-ygrQ) for showing that expert HP selection beats blind LLM search
- [LLAMBO (Ye et al., 2024)](https://arxiv.org/abs/2402.03921) for using LLMs as surrogate models in Bayesian optimization
