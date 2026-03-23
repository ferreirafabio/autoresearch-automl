"""Generate LLM-summarized descriptions for incumbent HP changes.

Reads trials.jsonl, finds incumbent points, diffs configs, and asks an LLM
to produce concise Karpathy-style descriptions (e.g. "halve batch size,
increase warmdown to 0.79"). Results cached in assets/incumbent_descriptions.json.

Usage:
    python scripts/generate_incumbent_descriptions.py [--api-base URL]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI


HP_DESCRIPTIONS = {
    "ASPECT_RATIO": "aspect ratio (width/depth ratio)",
    "DEPTH": "number of transformer layers",
    "DEVICE_BATCH_SIZE": "per-GPU batch size (tokens)",
    "EMBEDDING_LR": "embedding learning rate",
    "FINAL_LR_FRAC": "final LR fraction (cosine schedule end)",
    "HEAD_DIM": "attention head dimension",
    "MATRIX_LR": "matrix (Muon) learning rate",
    "SCALAR_LR": "scalar learning rate",
    "TOTAL_BATCH_SIZE": "total batch size (tokens)",
    "UNEMBEDDING_LR": "unembedding learning rate",
    "WARMDOWN_RATIO": "warmdown ratio (fraction of training for LR decay)",
    "WARMUP_RATIO": "warmup ratio (fraction of training for LR ramp)",
    "WEIGHT_DECAY": "weight decay",
    "WINDOW_PATTERN": "sliding window attention pattern (S=short, L=long per layer)",
}


def load_trials(jsonl_path: Path) -> list[dict]:
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


def find_incumbents(trials: list[dict]) -> list[tuple[int, float, dict]]:
    best = float("inf")
    incumbents = []
    for i, t in enumerate(trials):
        if t.get("success") and t.get("val_bpb") is not None and t["val_bpb"] < best:
            best = t["val_bpb"]
            incumbents.append((i, t["val_bpb"], t["config"]))
    return incumbents


def format_val(v) -> str:
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 0.01:
            return f"{v:.4f}"
        if abs(v) < 1:
            return f"{v:.3f}"
        return f"{v:g}"
    if isinstance(v, int) and v >= 10000:
        return f"{v // 1000}K"
    return str(v)


def config_diff_detailed(prev: dict, curr: dict) -> str:
    """Produce a detailed diff for the LLM prompt."""
    lines = []
    for hp in sorted(curr.keys()):
        if hp not in prev or prev[hp] != curr[hp]:
            desc = HP_DESCRIPTIONS.get(hp, hp)
            old = format_val(prev[hp]) if hp in prev else "N/A"
            new = format_val(curr[hp])
            lines.append(f"- {desc} ({hp}): {old} → {new}")
    return "\n".join(lines) if lines else "(no changes)"


def summarize_with_llm(client: OpenAI, model: str, diff_text: str, val_bpb: float, prev_bpb: float) -> str:
    prompt = f"""You are labeling a hyperparameter optimization progress plot.
Given the HP changes below between two consecutive best trials, write a very concise label
(max 30 chars) describing the most important 1-2 changes in plain English.

Style examples from Karpathy:
- "halve batch size"
- "warmdown 0.5→0.79"
- "short window 1/4 ctx"
- "deeper model (6→8 layers)"
- "2x embedding LR"
- "reduce weight decay"

HP changes (previous best → new best, val_bpb {prev_bpb:.4f} → {val_bpb:.4f}):
{diff_text}

Write ONLY the label, nothing else. Be specific with numbers when meaningful."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent.parent / "results"))
    parser.add_argument("--output", default="assets/incumbent_descriptions.json")
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="dummy")
    results_dir = Path(args.results_dir)
    bench_dir = results_dir 

    backends = [
        "optuna",
        "llambo_Qwen3_5_27B_nothink",
        "llambo_original_Qwen3_5_27B_nothink",
        "llm_greedy_Qwen3_5_27B_nothink",
    ]

    all_descriptions = {}

    for backend in backends:
        jsonl = bench_dir / backend / "seed_0" / "trials.jsonl"
        if not jsonl.exists():
            print(f"Skipping {backend}: no data")
            continue

        trials = load_trials(jsonl)
        incumbents = find_incumbents(trials)
        descriptions = []

        prev_config = None
        prev_bpb = None
        for trial_i, val_bpb, config in incumbents:
            if prev_config is None:
                desc = "baseline (default config)"
                descriptions.append({"trial": trial_i, "val_bpb": val_bpb, "description": desc})
            else:
                diff_text = config_diff_detailed(prev_config, config)
                desc = summarize_with_llm(client, args.model, diff_text, val_bpb, prev_bpb)
                descriptions.append({"trial": trial_i, "val_bpb": val_bpb, "description": desc})
                print(f"  {backend} trial {trial_i}: {desc}")

            prev_config = config
            prev_bpb = val_bpb

        all_descriptions[backend] = descriptions
        print(f"{backend}: {len(descriptions)} incumbents labeled")

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(all_descriptions, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
