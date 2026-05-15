"""Paired Wilcoxon signed-rank tests across seeds for headline paper claims.

Writes assets/wilcoxon_tests.txt with one table of every pair and the
significance markers used in the paper, plus two figures:

  - wilcoxon_forest.png        forest plot of Delta (mean A - mean B) per pair
                               with one-sided p annotated
  - wilcoxon_slopegraph_<pair>.png    per-seed paired traces for the headline
                                      Centaur [Opus 4.6] vs Centaur [Qwen 27B] pair

Run standalone: python scripts/wilcoxon_tests.py
Or via the plot_convergence __main__ block which calls run_all().
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


BUDGET_S = 86400.0
THRESHOLD_PCT = 99.0


def _budget_and_best(jsonl: Path) -> tuple[float, float | None]:
    if not jsonl.is_file():
        return 0.0, None
    total = 0.0
    best = float("inf")
    with open(jsonl) as f:
        for line in f:
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += float(t.get("wall_time_seconds") or 0.0)
            v = t.get("val_bpb") or t.get("best_val_bpb")
            if v is not None and v < best:
                best = v
    pct = min(100.0, 100.0 * total / BUDGET_S)
    return pct, (best if best < float("inf") else None)


def completed_seeds(method_dir: Path, threshold: float = THRESHOLD_PCT) -> dict[int, float]:
    out: dict[int, float] = {}
    if not method_dir.is_dir():
        return out
    for sdir in sorted(method_dir.glob("seed_*")):
        tag = sdir.name.replace("seed_", "")
        if not tag.isdigit():
            continue
        pct, best = _budget_and_best(sdir / "trials.jsonl")
        if pct >= threshold and best is not None:
            out[int(tag)] = best
    return out


EXP2_BENCH = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark")
OPUS46_BENCH = Path("/work/dlclarge1/ferreira-autoresearch-automl/results/opus46_benchmark")


def _method_paths(results_root: Path) -> dict[str, Path]:
    exp2 = EXP2_BENCH
    opus46 = OPUS46_BENCH
    return {
        "centaur_0.8B":       exp2 / "centaur_Qwen3_5_0_8B",
        "centaur_27B":        exp2 / "centaur_Qwen3_5_27B",
        "tpe":                exp2 / "optuna",
        "smac":               exp2 / "smac",
        "cma_es":             exp2 / "cma_es",
        "random":             exp2 / "random",
        "karpathy_hps_0.8B":  exp2 / "karpathy_agent_hps",
        "karpathy_hps_27Bnt": exp2 / "karpathy_agent_hps_Qwen3_5_27B_nothink",
        "llambo_0.8B":        exp2 / "llambo",
        "llambo_27Bnt":       exp2 / "llambo_Qwen3_5_27B_nothink",
        "llambo_orig_0.8B":   exp2 / "llambo_original",
        "centaur_opus46":     opus46 / "centaur_claude_opus_4_6",
        "karp_code_opus46":   opus46 / "karpathy_agent_claude_opus_4_6",
    }


# Pairs the paper makes claims about: (A, B) with H1 = A < B (lower bpb wins).
HEADLINE_PAIRS = [
    ("centaur_opus46",     "centaur_27B"),
    ("centaur_opus46",     "centaur_0.8B"),
    ("centaur_opus46",     "tpe"),
    ("centaur_opus46",     "smac"),
    ("centaur_opus46",     "cma_es"),
    ("centaur_27B",        "tpe"),
    ("centaur_0.8B",       "tpe"),
    ("centaur_27B",        "centaur_0.8B"),
    ("tpe",                "karpathy_hps_0.8B"),
    ("tpe",                "karpathy_hps_27Bnt"),
    ("smac",               "tpe"),
    ("cma_es",             "tpe"),
    ("random",             "tpe"),
]


def _sig_marker(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _wilcoxon_row(a_name: str, b_name: str, done: dict[str, dict[int, float]]) -> dict | None:
    da, db = done.get(a_name, {}), done.get(b_name, {})
    common = sorted(set(da) & set(db))
    if len(common) < 3:
        return {"a": a_name, "b": b_name, "n": len(common), "common": common, "skipped": True}
    a = np.array([da[s] for s in common])
    b = np.array([db[s] for s in common])
    diffs = a - b
    _, p_two = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    _, p_one = stats.wilcoxon(a, b, zero_method="wilcox", alternative="less")
    return {
        "a": a_name, "b": b_name, "n": len(common), "common": common,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
        "signs_neg": int((diffs < 0).sum()),
        "p_two": float(p_two), "p_one": float(p_one),
        "skipped": False,
    }


def write_table(rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("Paired Wilcoxon signed-rank tests across seeds")
    lines.append(f"Completion threshold: >= {THRESHOLD_PCT:g}% of 24h training budget")
    lines.append("H1 (one-sided): A < B (lower val_bpb is better)")
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    lines.append("")
    lines.append(f"{'A':<20} {'B':<20} {'n':>3} {'mean A':>8} {'mean B':>8} "
                 f"{'Delta':>8} {'A<B':>6} {'p two':>10} {'p one':>10}")
    lines.append("-" * 110)
    for r in rows:
        if r["skipped"]:
            lines.append(f"{r['a']:<20} {r['b']:<20} {r['n']:>3}  (skip; n<3, common={r['common']})")
            continue
        lines.append(
            f"{r['a']:<20} {r['b']:<20} {r['n']:>3} "
            f"{r['mean_a']:>8.4f} {r['mean_b']:>8.4f} {r['delta']:>+8.4f} "
            f"{r['signs_neg']:>2}/{r['n']:<3} "
            f"{r['p_two']:>7.4f} {_sig_marker(r['p_two']):<3} "
            f"{r['p_one']:>7.4f} {_sig_marker(r['p_one']):<3}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def forest_plot(rows: list[dict], out_path: Path) -> None:
    usable = [r for r in rows if not r["skipped"]]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(usable) + 1.5)))
    y = np.arange(len(usable))
    deltas = np.array([r["delta"] for r in usable])
    ns = np.array([r["n"] for r in usable])
    p_ones = np.array([r["p_one"] for r in usable])

    colors = ["#2E7D32" if d < 0 else "#C62828" for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.45, height=0.55)
    ax.scatter(deltas, y, color=colors, s=40, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)

    labels = [f"{r['a']} vs {r['b']}" for r in usable]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ = mean(A) - mean(B)   (negative: A better)", fontsize=10)
    ax.set_title("Paired Wilcoxon: Δ across seeds with one-sided p (H1: A<B)", fontsize=11)

    xmax = max(abs(deltas).max() * 1.5, 0.005)
    ax.set_xlim(-xmax, xmax)

    for yi, r, p in zip(y, usable, p_ones):
        marker = _sig_marker(p)
        txt = f"n={r['n']}  p={p:.3f}{(' ' + marker) if marker else ''}"
        xpos = r["delta"] + (xmax * 0.04 if r["delta"] >= 0 else -xmax * 0.04)
        ha = "left" if r["delta"] >= 0 else "right"
        ax.text(xpos, yi, txt, va="center", ha=ha, fontsize=8,
                color="black" if not marker else ("#1565C0" if p < 0.05 else "#444"))

    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def slopegraph_pair(a_name: str, b_name: str, done: dict[str, dict[int, float]],
                    out_path: Path) -> None:
    da, db = done.get(a_name, {}), done.get(b_name, {})
    common = sorted(set(da) & set(db))
    if len(common) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(common):
        col = cmap(i / max(len(common) - 1, 1))
        ax.plot([0, 1], [da[s], db[s]], "-o", color=col, linewidth=1.5, markersize=6,
                label=f"seed {s}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a_name, b_name])
    ax.set_ylabel("Best val_bpb (lower is better)")
    res = _wilcoxon_row(a_name, b_name, done)
    if res and not res["skipped"]:
        title = (f"{a_name} vs {b_name}  (paired n={res['n']})\n"
                 f"Δ={res['delta']:+.4f}  signs={res['signs_neg']}/{res['n']}  "
                 f"two-sided p={res['p_two']:.3f}  one-sided p={res['p_one']:.3f}")
    else:
        title = f"{a_name} vs {b_name}"
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_all(results_root: Path, assets_dir: Path) -> list[dict]:
    paths = _method_paths(results_root)
    done = {name: completed_seeds(p) for name, p in paths.items()}
    rows = [_wilcoxon_row(a, b, done) for a, b in HEADLINE_PAIRS]
    assets_dir.mkdir(exist_ok=True)
    write_table(rows, assets_dir / "wilcoxon_tests.txt")
    forest_plot(rows, assets_dir / "wilcoxon_forest.png")
    slopegraph_pair("centaur_opus46", "centaur_27B", done,
                    assets_dir / "wilcoxon_slopegraph_opus_vs_centaur27b.png")
    return rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "results"
    assets = Path(__file__).resolve().parent.parent / "assets"
    rows = run_all(root, assets)
    print((assets / "wilcoxon_tests.txt").read_text())
