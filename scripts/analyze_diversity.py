"""Compute search diversity metrics for exp2_benchmark results."""

import json
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "exp2_benchmark"

# Search space bounds (for normalization to [0,1])
BOUNDS = {
    "DEPTH": (4, 24),
    "ASPECT_RATIO": (32, 128),
    "HEAD_DIM": (64, 256),
    "DEVICE_BATCH_SIZE": (32, 256),
    "TOTAL_BATCH_SIZE": (65536, 2097152),
    "EMBEDDING_LR": (0.01, 2.0),
    "UNEMBEDDING_LR": (0.0005, 0.05),
    "MATRIX_LR": (0.005, 0.2),
    "SCALAR_LR": (0.05, 2.0),
    "WEIGHT_DECAY": (0.0, 0.5),
    "WARMUP_RATIO": (0.0, 0.3),
    "WARMDOWN_RATIO": (0.1, 0.8),
    "FINAL_LR_FRAC": (0.0, 0.2),
}

# Karpathy's starting config (commit b11d6f28)
DEFAULT = {
    "DEPTH": 8, "ASPECT_RATIO": 64, "HEAD_DIM": 128, "DEVICE_BATCH_SIZE": 128,
    "TOTAL_BATCH_SIZE": 524288, "EMBEDDING_LR": 0.6, "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04, "SCALAR_LR": 0.5, "WEIGHT_DECAY": 0.2,
    "WARMUP_RATIO": 0.0, "WARMDOWN_RATIO": 0.5, "FINAL_LR_FRAC": 0.0,
}

HP_ORDER = list(BOUNDS.keys())

BACKENDS = {
    "centaur_Qwen3_5_0_8B": "Centaur [0.8B]",
    "centaur_Qwen3_5_27B": "Centaur [27B]",
    "optuna": "TPE",
    "cma_es": "CMA-ES",
    "smac": "SMAC",
    "karpathy_agent_Qwen3_5_27B": "Karpathy Agent (Code) [27B]",
    "karpathy_agent_Qwen3_5_0_8B": "Karpathy Agent (Code) [0.8B]",
    "random": "Random",
    "llambo_original_Qwen3_5_27B_nothink": "LLAMBO (Paper) [27B]",
    "llambo_original": "LLAMBO (Paper) [0.8B]",
    "llm_greedy_Qwen3_5_27B_nothink": "Karpathy Agent (14 HPs) [27B]",
    "llm_greedy": "Karpathy Agent (14 HPs) [0.8B]",
    "llambo_Qwen3_5_27B_nothink": "LLAMBO (Optuna) [27B]",
    "llambo": "LLAMBO (Optuna) [0.8B]",
}

MIN_BUDGET_FRAC = 0.70
BUDGET_SECONDS = 86400
MAX_TRIALS = 300


def normalize(config):
    vec = []
    for hp in HP_ORDER:
        lo, hi = BOUNDS[hp]
        v = float(config.get(hp, DEFAULT[hp]))
        vec.append((v - lo) / (hi - lo) if hi > lo else 0.0)
    return np.array(vec)


def load_trials(jsonl_path):
    trials = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trials


def main():
    default_vec = normalize(DEFAULT)
    results = []

    for backend_dir, display_name in BACKENDS.items():
        all_vecs = []
        all_bests = []
        n_total = 0
        n_oom = 0
        n_seeds = 0
        is_code = "Code" in display_name

        for seed in range(10):
            jsonl = RESULTS / backend_dir / f"seed_{seed}" / "trials.jsonl"
            if not jsonl.exists():
                continue
            trials = load_trials(jsonl)
            if len(trials) < 100:
                continue

            total_train_s = sum(t.get("wall_time_seconds") or 0.0 for t in trials)
            if total_train_s < BUDGET_SECONDS * MIN_BUDGET_FRAC:
                continue

            n_seeds += 1
            trimmed = trials[:MAX_TRIALS]

            seed_best = float("inf")
            for t in trimmed:
                n_total += 1
                if not t.get("success", False):
                    n_oom += 1
                elif t.get("val_bpb") is not None:
                    seed_best = min(seed_best, t["val_bpb"])
                    if not is_code:
                        all_vecs.append(normalize(t["config"]))

            if seed_best < float("inf"):
                all_bests.append(seed_best)

        if not all_bests:
            continue

        avg_best = np.mean(all_bests)
        oom_pct = n_oom / n_total * 100 if n_total > 0 else 0

        if all_vecs and not is_code:
            vecs = np.array(all_vecs)
            spread = np.mean(np.std(vecs, axis=0))

            # Pairwise L2 (subsample if too many configs)
            if len(vecs) > 2000:
                idx = np.random.default_rng(42).choice(len(vecs), 2000, replace=False)
                sub = vecs[idx]
            else:
                sub = vecs
            dists = np.linalg.norm(sub[:, None] - sub[None, :], axis=2)
            pairwise = np.mean(dists[np.triu_indices(len(sub), k=1)])

            # Distance to default config
            dist_default = np.mean(np.linalg.norm(vecs - default_vec, axis=1))

            # Step size (mean L2 between consecutive trials)
            steps = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
            step = np.mean(steps) if len(steps) > 0 else 0

            # Unique cells (5 bins per HP)
            binned = np.clip((vecs * 5).astype(int), 0, 4)
            cells = len(set(tuple(row) for row in binned))

            results.append((avg_best, display_name, n_seeds, oom_pct,
                            spread, pairwise, dist_default, step, cells))
        else:
            results.append((avg_best, display_name, n_seeds, oom_pct,
                            None, None, None, None, None))

    results.sort(key=lambda r: r[0])

    print("| Method | Seeds | Avg Best | OOM% | Spread | Pairwise "
          "| Dist→Default | Step | Cells |")
    print("|--------|-------|----------|------|--------|----------"
          "|-------------|------|-------|")
    for (avg_best, name, n_seeds, oom_pct,
         spread, pairwise, dist_default, step, cells) in results:
        best_str = f"**{avg_best:.4f}**" if avg_best < 0.984 else f"{avg_best:.4f}"
        if spread is not None:
            print(f"| {name} | {n_seeds} | {best_str} | {oom_pct:.0f}% "
                  f"| {spread:.3f} | {pairwise:.3f} | {dist_default:.3f} "
                  f"| {step:.3f} | {cells} |")
        else:
            print(f"| {name} | {n_seeds} | {best_str} | {oom_pct:.0f}% "
                  f"| - | - | - | - | - |")


if __name__ == "__main__":
    main()
