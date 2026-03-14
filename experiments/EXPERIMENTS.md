# Experiment Log

## Baseline

| Metric | Value |
|--------|-------|
| val_bpb | 1.006408 |
| training_seconds | 300.0 |
| total_seconds | 355.8 |
| peak_vram_mb | 45060.2 |
| mfu_percent | 33.48 |
| total_tokens_M | 421.0 |
| num_steps | 803 |
| num_params_M | 50.3 |
| depth | 8 |
| GPU | NVIDIA H200 |
| node | dlc2gpu18 |

Default hyperparameters from train.py (DEPTH=8, ASPECT_RATIO=64, MATRIX_LR=0.04, WEIGHT_DECAY=0.2, etc.)

---

## Experiment 1: LLM Model Size Comparison

**Goal**: Compare HPO performance using three Qwen3.5 model tiers as the LLM backend for `llm_greedy` suggestions.

| Run | LLM Model | Params | VRAM (bf16) | Backend | Trials | Status |
|-----|-----------|--------|-------------|---------|--------|--------|
| 1a | Qwen3.5-0.8B | 0.8B dense | ~2GB | llm_greedy | 30 | running |
| 1b | Qwen3.5-9B | 9B dense | ~18GB | llm_greedy | 30 | running |
| ~~1c~~ | ~~Qwen3.5-35B-A3B~~ | ~~35B MoE~~ | ~~~70GB~~ | ~~llm_greedy~~ | ~~30~~ | dropped (too large for single GPU with training) |

### Results

#### Run 1a: Qwen3.5-0.8B (small)

- Best val_bpb: _pending_
- Trials completed: _pending_

#### Run 1b: Qwen3.5-9B (medium)

- Best val_bpb: _pending_
- Trials completed: _pending_

#### Run 1c: Qwen3.5-35B-A3B (large)

- Best val_bpb: _pending_
- Trials completed: _pending_

### Visualizations

#### Convergence Plot (val_bpb vs trial number)
![Convergence](plots/exp1_convergence.png)

#### Anytime Performance (val_bpb vs wall-clock time)
![Anytime](plots/exp1_anytime.png)

#### Pareto Front (val_bpb vs peak memory)
![Pareto](plots/exp1_pareto.png)

#### HP Importance
![HP Importance](plots/exp1_hp_importance.png)

---

## Experiment 2: AutoML Backends Comparison

**Goal**: Compare classical HPO backends against LLM-based approaches.

_Planned after Experiment 1 validates the pipeline._

| Run | Backend | LLM Model | Trials | Status |
|-----|---------|-----------|--------|--------|
| 2a | random | - | 100 | pending |
| 2b | optuna (TPE) | - | 100 | pending |
| 2c | llm_greedy | Qwen3.5-9B | 100 | pending |
| 2d | llambo | Qwen3.5-9B | 30 | pending |
| 2e | smac | - | 100 | pending |
| 2f | dehb | - | 100 | pending |
| 2g | bohb | - | 100 | pending |

---

## Hardware

- **Training GPU**: NVIDIA H200 (141GB HBM3e) via `alldlc2_gpu-h200`
- **LLM Serving**: vLLM on H200, bf16
- **Data prep**: L40S via `alldlc2_gpu-l40s`
- **Dataset**: karpathy/climbmix-400b-shuffle (10 train shards + 1 val shard)
- **Time budget**: 300s (5 min) per trial
