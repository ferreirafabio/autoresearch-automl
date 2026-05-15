---
name: Kimi K2.6 self-hosting setup on KISLURM3
description: Complete working recipe + root-cause analysis for Kimi K2.6 on H200 cluster; 3 required patches, plus alternative paths we evaluated
type: project
originSessionId: be4ceefa-b43b-4d87-b775-f942ca0db5de
---
## Working setup (current)

**Cluster**: KISLURM3, `alldlc2_gpu-h200` partition. 4 nodes × 8 H200s (141 GB each).
**Driver**: 570.211.01 reports CUDA 12.8 support (but — see PTX note below).
**Configuration**: single 8×H200 node, TP=8, no Ray, no multi-node.
**Venv**: `/work/dlclarge1/ferreira-autoresearch-automl/.venv-kimi/` (separate from main `.venv`).
**Stack**: vLLM 0.19.1 + torch 2.10.0+cu128.

## Model details

- **1 trillion total params**, 32 B activated (MoE, 384 experts, 8 selected per token)
- **4-bit WNA16 compressed-tensors quantization** (no bf16 release exists — 2 TB would be needed)
- 595 GB on disk, 64 safetensors shards, each GPU holds ~72 GB of 4-bit weights
- Multimodal (text + images), but text-only usable with `vision_chunk=0`
- Modality name is **`vision_chunk`** (unique to Kimi; NOT `image` / `video`)
- Model dir: `/work/dlclarge1/ferreira-autoresearch-automl/models/Kimi-K2.6/` (cold download ~1 h via `huggingface-cli` with `HF_HUB_ENABLE_HF_TRANSFER=1`)

## Three critical patches needed

All are env-var-gated so they toggle cleanly:

1. **`VLLM_DISABLE_MARLIN_MOE=1`** + local patch to `compressed_tensors_moe.py`
   - Adds an env-var-gated override to force the Triton WNA16 path instead of Marlin
   - Needed because the cluster driver can't parse Marlin's prebuilt PTX
   - Cost: Triton fallback is ~2-3× slower than Marlin (same math, just slower kernel)

2. **`VLLM_DISABLE_FLASHINFER_ROPE=1`** + local patch to `deepseek_scaling_rope.py`
   - Skips flashinfer's `apply_rope_with_cos_sin_cache_inplace` (uses `tvm_ffi` CUDA kernels)
   - Needed because flashinfer's `tvm_ffi` bundled kernels also need a newer driver (same `cudaErrorInsufficientDriver`)
   - Falls back to vLLM's native Torch RoPE (negligible perf diff)

3. **`VLLM_DISABLE_MOE_WNA16_CUDA=1`** + local patch to `fused_moe.py`
   - Forces the Triton WNA16 path instead of vLLM's `moe_wna16_gemm` CUDA kernel
   - Needed because the CUDA kernel crashes with `BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]` at runtime for Kimi K2.6's `group_size=32` on small batches (HPO batch=1)
   - Falls back to Triton MoE kernel (same accuracy, slightly slower for small batches)
   - The check sits in `should_moe_wna16_use_cuda()` — it auto-picks CUDA for `num_valid_tokens/num_experts <= 6` which is always true for our HPO inference

4. **Explicit `LD_LIBRARY_PATH`** before `python`
   - Must include `.venv-kimi/lib/python3.12/site-packages/{torch/lib,nvidia/*/lib}`
   - Without it, multiproc TP spawn-children don't find the bundled CUDA libs and return `cudaErrorInsufficientDriver`
   - See the loop in `slurm/kimi_k26_server.sh`

Patch targets:
- `.venv-kimi/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` (`_force_no_marlin` gate)
- `.venv-kimi/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` (`_disable_fi_rope` gate)
- After patching, `rm -f` the `__pycache__/*.pyc` so Python recompiles.

## Partition strategy (avoid preemption)

`alldlc2_gpu-h200` has `PreemptMode=REQUEUE, PriorityTier=0` — any higher-priority user kicks you. Lost our Kimi server after 1h10m once because of this. `mldlc2_gpu-h200` has `PreemptMode=OFF, PriorityTier=1000` but caps at 9 GPU per user (QoS `MaxTRESPU: cpu=411,gres/gpu=9`).

**User preference (2026-04-21)**: use `alldlc2_gpu-h200` for BOTH server and HPO going forward. Accept preemption risk — the alternative (mldlc2 with 9-GPU cap) can't run both an 8-GPU server and 9 parallel HPO jobs simultaneously.

**Previous emergency-only split** (kept in this memory in case we need it again):
- Server on `mldlc2_gpu-h200` (PreemptMode=OFF, PriorityTier=1000)
- HPO jobs on `alldlc2_gpu-h200` (PreemptMode=REQUEUE; `--requeue` + fail-hard patch handle losses)
Was used once when preemption cost us 1h10m of server load time; user wants alldlc2 for both next time.

## Critical HPO-script env var gotcha

**`openai` Python SDK ≥ 1.x ignores `OPENAI_API_BASE`** (legacy name from openai 0.x). The modern env var is **`OPENAI_BASE_URL`**. Set **both** in HPO scripts for safety:

```bash
export OPENAI_BASE_URL="http://${KIMI_ENDPOINT}/v1"
export OPENAI_API_BASE="http://${KIMI_ENDPOINT}/v1"
```

If only `OPENAI_API_BASE` is set, the client silently sends requests to `api.openai.com` with the dummy key → 401 → fall-hard-on-connection patch doesn't trigger (401 is a status error, not a connection error) → backend falls back to random / CMA / last-best, all trials get tagged with the LLM model but are actually non-LLM. Polluted run, scientifically unusable.

The Qwen reference script `slurm/exp2_centaur.sh` correctly uses `OPENAI_BASE_URL` — past Qwen/Opus/Gemini runs are unaffected. This was a bug in my initial Kimi HPO scripts only, fixed in commit `26354f2`.

**How to verify the env var is right**: SLURM log should show HTTP requests to `http://dlc2gpu20:8100/v1/chat/completions`, NOT `https://api.openai.com/v1/...`. Grep for `POST http` in the log.

## Required vLLM flags

- `--tensor-parallel-size 8`
- `--dtype auto` (respects compressed-tensors)
- `--trust-remote-code` (Kimi custom modeling code)
- `--enforce-eager` (no CUDA graphs — reduces surface area for PTX issues)
- `--limit-mm-per-prompt '{"vision_chunk":0,"image":0,"video":0}'` (skip vision profiling; `vision_chunk` is essential — that's Kimi's modality name)
- `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2` (per Moonshot docs)
- `--gpu-memory-utilization 0.90`
- `--max-model-len 65536` — KA Code's prompt includes the full train.py source (~10k tokens) + `max_tokens=16384` output, so total can reach ~26k. 16384 was too small and caused 400 Bad Request for KA Code (Centaur + KA HPs worked fine). 65536 is safe margin; Kimi's native default is 256K.

## Files

- Server sbatch: `slurm/kimi_k26_server.sh`
- HPO sbatches: `slurm/exp2_{centaur,karpathy_agent_hps,karpathy_agent}_kimi.sh`
- HPO scripts wait on `/work/dlclarge1/ferreira-autoresearch-automl/kimi_k26_endpoint.txt` (shared file, server writes on startup + clears on exit)

## Load time

- Warm FS cache: ~4 min (100% shards)
- Cold FS cache: ~100 min (network FS bandwidth limited, reads 595 GB)
- Engine init + profile_run adds another ~5-10 min after 100% loaded
- Uvicorn "Application startup complete" ≈ 10-15 min total when warm

## Root cause of the PTX error

**Symptom**: `cudaErrorUnsupportedPtxVersion` during Marlin kernel init, `cudaErrorInsufficientDriver` during flashinfer kernel launches.

**Root cause**: vLLM's pip-distributed `_C.abi3.so` ships precompiled CUDA kernels where the embedded PTX targets ISA 8.9+ (CUDA 12.9 toolchain). Our driver 570.211.01 reports CUDA 12.8 support — it **cannot parse PTX ISA 8.9+**. Same problem for flashinfer's bundled `tvm_ffi` kernels.

**Why the driver version is ambiguous**: `nvidia-smi` says "CUDA Version: 12.8", but 12.8 driver doesn't always mean it can parse 12.8 PTX. Some CUDA driver features (e.g., Blackwell sm_100a instructions, certain Hopper async copy features) only appear in post-12.8 driver branches. If vLLM's wheel was compiled using a newer CUDA toolkit, its PTX might contain instructions our driver doesn't know about.

**Why both Marlin and flashinfer hit this**: they're the two heaviest kernels in vLLM for MoE inference, both distributed as precompiled `.so`s with PTX for forward-compat. vLLM's handwritten CUDA kernels (in `_C.abi3.so`) and flashinfer's `tvm_ffi` kernels are independent — both happen to use toolchain features our driver lacks.

## Things that DIDN'T work

All tested:
- **vLLM 0.17.1** (original `.venv`): same PTX error
- **vLLM 0.19.1** (Moonshot-verified stable): same PTX error
- **vLLM nightly** (`wheels.vllm.ai/nightly`, 0.19.2rc1 at time of testing): same PTX error — so Moonshot's own recipe doesn't work out-of-the-box on this cluster
- `VLLM_ENABLE_CUDA_COMPATIBILITY=1` env var (Docker-only per docs; no effect outside Docker)
- `--enforce-eager` alone (doesn't touch Marlin/flashinfer PTX path)
- Multi-node PP=2 TP=4 via Ray (2 nodes × 4 GPUs): Ray worker env issues on dlc2gpu21 caused `cudaErrorInsufficientDriver` on the remote worker. Sourcing the venv in the srun worker command didn't help.
- TP=7 single-node: fails because 7 doesn't divide 64 attention heads.
- TP=4 single-node with 4 GPUs remaining for training: Kimi's weights don't fit (need ≥5 GPUs for the 4-bit model).

## Alternative paths we considered but didn't take

### Build vLLM from source with cluster-local nvcc

The cluster has CUDA toolkits 11.7–12.6 preinstalled in `/usr/local/cuda-X.Y/`, switchable via `source /etc/cuda_env; cuda12.6` (see the KISLURM ticket-system wiki entry). Using `cuda12.6`'s nvcc to build vLLM from source would emit PTX 8.6, which our driver can parse → Marlin works.

Steps (if ever needed):
```bash
source /etc/cuda_env
cuda12.6  # or cuda12.4 for broader compat
export CUDA_HOME=/usr/local/cuda-12.6
# Match torch to toolkit:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Build vLLM from source (~30-60 min compile):
pip install --no-binary=vllm vllm==0.19.1
```

Complications expected:
- vLLM 0.19.1's build may require torch ≥ 2.7; older torch may not satisfy deps.
- 32+ GB RAM needed — build on a compute node interactive session, not login node.
- Resulting wheel is ABI-pinned to cluster's CUDA 12.6.

Cost vs benefit: ~1 evening of debugging. Benefit: Marlin = 2-3× faster inference, but only matters if we run Kimi-scale models many times.

### Ask cluster admin to update NVIDIA driver

Driver ≥ 570.86 (or whatever supports CUDA 12.9) would fix both Marlin and flashinfer kernels without any patches. External dependency, slow. Worth asking if the cluster admin is responsive.

### Use Moonshot API (platform.moonshot.ai)

Hosted K2.6, OpenAI-compatible. Costs money. User explicitly declined (local self-hosting preferred).

## Why Kimi K2.6 is quantized (paper-relevance)

Moonshot only publishes K2.6 as 4-bit compressed-tensors. The bf16 version would be ~2 TB and doesn't fit on any reasonable cluster. **Every** K2.6 deployment uses this same quantized checkpoint; it's not our choice.

In our paper's LLM comparison:
- Kimi K2.6: 4-bit (Moonshot's only release)
- Qwen 3.5-27B: bf16 (self-hosted)
- Gemini 3.1 Pro, Claude Opus 4.6/4.7: vendor-served, precision unknown (probably quantized internally by the provider)
- 50M optimizee: bf16, unchanged

Worth a methods-section footnote: "Kimi K2.6 served via vLLM's Triton WNA16 MoE backend (equivalent output to Marlin, slightly slower inference)".

## HPO job robustness

Backends (`centaur_backend.py`, `karpathy_agent_backend.py`, `karpathy_agent_hps_backend.py`) were patched 2026-04-21 to **raise on `APIConnectionError` / `APITimeoutError`** instead of silently falling back to random / last-best / pure-CMA. Content errors (bad JSON from LLM) still fall back as before.

Reason: if server dies mid-run, a silent fallback produces "LLM" trials that are actually random/CMA. `trials.jsonl` would still be tagged `backend=centaur_kimi_k2_6`, making the run scientifically unusable. Better to fail hard and requeue.

Committed as `dab9416` in private repo.

## How to verify the setup is running correctly

When the server is up, monitor log should show these key lines in order:
1. `Wrote endpoint: <node>:8100` — endpoint file published
2. `Using CompressedTensorsWNA16MoEMethod` — **non-Marlin path** (the patch is working). Must NOT see `CompressedTensorsWNA16MarlinMoEMethod`.
3. `Loading safetensors checkpoint shards: 100% Completed | 64/64` — weights loaded
4. `Model loading took XX GiB memory and YYY seconds` — post-load processing done
5. `flashinfer.jit: [Autotuner]: Autotuning process starts/ends` — flashinfer works for non-RoPE paths
6. `Application startup complete` — server ready, can submit HPO jobs

Bad signs: `cudaErrorUnsupportedPtxVersion`, `cudaErrorInsufficientDriver`, `Error executing method 'determine_available_memory'`, any `Traceback` in the log.

## Proxy gotcha (cluster squid intercept)

The cluster has `http_proxy`/`https_proxy` set by default on compute nodes for external web access (pip installs etc.). If not unset, **httpx routes requests to the local Kimi server through the proxy**, which returns 503 with "Unable to determine IP address" squid error page. The `no_proxy` env var is sometimes not respected by httpx when IP-based.

**Fix applied in all 3 Kimi HPO scripts**: unset ALL proxy vars before setting no_proxy:
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost,dlc2gpu18,dlc2gpu19,dlc2gpu20,dlc2gpu21,10.5.166.0/24,.dlc2gpu"
export NO_PROXY="127.0.0.1,localhost,..."
```

The server also now writes IP (not hostname) to the endpoint file via `hostname -I | awk '{print $1}'` — avoids DNS lookup entirely.
