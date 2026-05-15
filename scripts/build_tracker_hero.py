"""Build the Live-Benchmark hero plot (section A of the tracker tab) and
inject it into docs/index.html.

Methods shown: TPE (classical reference, always visible) plus three method
families across every Claude generation present on disk: Centaur,
Karpathy Agent (Code), Karpathy Agent (14 HPs).

Default-visible: TPE + the two most recent Claude generations (today: 4.6
and 4.7). Older generations are loaded as legendonly traces so the user
can toggle them on.

Run: python scripts/build_tracker_hero.py
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import numpy as np

from scripts.plot_convergence import load_trials, cumulative_walltime

REPO = Path(__file__).resolve().parent.parent
HTML_PATH = REPO / "docs" / "index.html"
RESULTS = Path("/work/dlclarge1/ferreira-autoresearch-automl/results")
EXP2_BENCH = RESULTS / "exp2_benchmark"

# Method-family colors match the canonical palette from plot_convergence.py
# / Fig 1, so a trace looks identical across the two tabs.
METHOD_COLOR = {
    "centaur": "#C62828",  # red, matches Centaur [Opus 4.6] in Fig 1
    "ka_code": "#1565C0",  # deep blue, matches KA Code [Opus 4.6] in Fig 1
    "ka_hps":  "#FFC107",  # amber, matches KA HPs in Fig 1
}
METHOD_DISPLAY = {
    "centaur": "Centaur",
    "ka_code": "Karpathy Agent (Code)",
    "ka_hps":  "Karpathy Agent (14 HPs)",
}
# Dash style per Claude generation index (oldest = 0). Cycles back to dot if
# more than five generations land.
DASH_BY_GEN_INDEX = ["solid", "dash", "dashdot", "longdash", "longdashdot", "dot"]


# Claude generations discoverable today. Listed oldest -> newest so the
# default-visible cut (last 2) is automatic when new generations are added.
GENERATIONS: list[dict] = [
    {
        "tag":     "Opus 4.6",
        "tag_id":  "opus_4_6",
        "base":    RESULTS / "opus46_benchmark",
        "centaur": "centaur_claude_opus_4_6",
        "ka_code": "karpathy_agent_claude_opus_4_6",
        "ka_hps":  None,  # not run for 4.6
    },
    {
        "tag":     "Opus 4.7",
        "tag_id":  "opus_4_7",
        "base":    RESULTS / "opus47_benchmark",
        "centaur": "centaur_claude_opus_4_7",
        "ka_code": "karpathy_agent_claude_opus_4_7",
        "ka_hps":  "karpathy_agent_hps_claude_opus_4_7",
    },
]

INTERP_HOURS = np.linspace(0, 24, 1000)
CAP_BUDGET_S = 86400.0
COMPLETION_THRESHOLD = 0.95  # treat seeds >=95% budget as "done enough"


def _seed_dirs(method_dir: Path) -> list[Path]:
    return sorted([p for p in method_dir.glob("seed_*") if p.name.split("_")[-1].isdigit()],
                  key=lambda p: int(p.name.split("_")[-1]))


def _seed_curve(jsonl: Path) -> tuple[np.ndarray, float] | None:
    """Return (interpolated incumbent across INTERP_HOURS, budget_frac) or
    None if no usable trials."""
    if not jsonl.is_file():
        return None
    trials = load_trials(jsonl)
    if not trials:
        return None
    times, vals = cumulative_walltime(trials, cap_hours=24.0)
    if not times:
        return None
    # Forward-fill onto the common grid (last value carried forward past
    # the seed's wall-clock end so a partial seed doesn't pull the mean up).
    interp = np.full_like(INTERP_HOURS, np.nan, dtype=float)
    t_arr = np.array(times)
    v_arr = np.array(vals)
    for i, h in enumerate(INTERP_HOURS):
        mask = t_arr <= h
        if mask.any():
            interp[i] = v_arr[mask][-1]
    last_t = max(times)
    budget = min(1.0, last_t / 24.0)
    return interp, budget


def _aggregate(method_dir: Path) -> dict | None:
    """Mean +/- std across seeds completed to threshold."""
    curves: list[np.ndarray] = []
    seeds: list[int] = []
    if not method_dir.is_dir():
        return None
    for sdir in _seed_dirs(method_dir):
        res = _seed_curve(sdir / "trials.jsonl")
        if not res:
            continue
        curve, budget = res
        if budget < COMPLETION_THRESHOLD:
            continue
        curves.append(curve)
        seeds.append(int(sdir.name.split("_")[-1]))
    if not curves:
        return None
    arr = np.stack(curves)
    # NaN-aware reduction (early grid points before any trial finished)
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr, axis=0)
    final = float(np.nanmin(mean))
    return {"x": INTERP_HOURS.tolist(), "mean": mean.tolist(),
            "std": std.tolist(), "n_seeds": len(seeds), "seeds": seeds,
            "final": final}


def _hex_to_rgba(hexstr: str, a: float) -> str:
    h = hexstr.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _safe(arr) -> list:
    """Convert numpy array (or list with NaNs) to JSON-safe Python list with
    None for missing values."""
    out = []
    for v in arr:
        if v is None:
            out.append(None)
        else:
            f = float(v)
            out.append(None if (f != f) else f)  # f != f catches NaN
    return out


def _traces_for(legend: str, color: str, agg: dict, visible: bool,
                dash: str = "solid", model_tag: str = "") -> list[dict]:
    """Three traces (lower band, upper band, mean line) for one method+version.
    The mean-line trace carries a `meta.model` field that the filter buttons
    read to decide visibility."""
    x = _safe(agg["x"])
    mean_arr = np.array(agg["mean"], dtype=float)
    std_arr = np.array(agg["std"], dtype=float)
    lower = _safe(mean_arr - std_arr)
    upper = _safe(mean_arr + std_arr)
    mean = _safe(mean_arr)
    vis = True if visible else "legendonly"
    label = f"{legend}  [{agg['final']:.4f}]  (n={agg['n_seeds']})"
    fill_rgba = _hex_to_rgba(color, 0.10)
    return [
        {"hoverinfo": "skip", "legendgroup": legend, "showlegend": False,
         "mode": "lines", "line": {"width": 0, "color": color},
         "x": x, "y": lower, "type": "scatter", "visible": vis,
         "meta": {"model": model_tag}},
        {"hoverinfo": "skip", "legendgroup": legend, "showlegend": False,
         "fill": "tonexty", "fillcolor": fill_rgba,
         "mode": "lines", "line": {"width": 0, "color": color},
         "x": x, "y": upper, "type": "scatter", "visible": vis,
         "meta": {"model": model_tag}},
        {"legendgroup": legend, "name": label,
         "mode": "lines", "line": {"color": color, "dash": dash, "width": 2.5},
         "hovertemplate": "<b>" + legend + "</b><br>hour: %{x:.2f}<br>val_bpb: %{y:.4f}<extra></extra>",
         "x": x, "y": mean, "type": "scatter", "visible": vis,
         "meta": {"model": model_tag}},
    ]


def build_traces() -> tuple[list[dict], list[str]]:
    """Returns (traces_in_best_first_order, info_lines).  Default-visible state:
    every trace (filter buttons let user narrow to a single model). Color per
    method family matches Fig 1; dash per Claude generation lets you tell
    generations apart at a glance."""
    info: list[str] = []
    # Collect (best, trace_group, label) tuples, then sort by best ascending.
    groups: list[tuple[float, list[dict], str]] = []

    tpe = _aggregate(EXP2_BENCH / "optuna")
    if tpe is None:
        info.append("WARNING: TPE has no completed seeds; skipping reference.")
    else:
        legend = "TPE (classical ref)"
        triplet = _traces_for(legend, "#2196F3", tpe, visible=True,
                              dash="dot", model_tag="classical")
        groups.append((tpe["final"], triplet, legend))
        info.append(f"TPE (optuna): n={tpe['n_seeds']} seeds, best mean={tpe['final']:.4f}")

    for gen_idx, gen in enumerate(GENERATIONS):
        dash = DASH_BY_GEN_INDEX[gen_idx % len(DASH_BY_GEN_INDEX)]
        for method_key in ("centaur", "ka_code", "ka_hps"):
            sub = gen.get(method_key)
            if not sub:
                continue
            agg = _aggregate(gen["base"] / sub)
            if not agg:
                info.append(f"  {METHOD_DISPLAY[method_key]} [{gen['tag']}]: no completed seeds")
                continue
            legend = f"{METHOD_DISPLAY[method_key]} [{gen['tag']}]"
            color = METHOD_COLOR[method_key]
            triplet = _traces_for(legend, color, agg, visible=True,
                                  dash=dash, model_tag=gen["tag_id"])
            groups.append((agg["final"], triplet, legend))
            info.append(f"  {legend}: n={agg['n_seeds']} seeds {agg['seeds']}, "
                        f"best mean={agg['final']:.4f}, color={color}, dash={dash}")

    groups.sort(key=lambda g: g[0])  # best (lowest val_bpb) first in legend
    traces: list[dict] = []
    for _, triplet, _ in groups:
        traces.extend(triplet)
    return traces, info


def _present_generations(traces: list[dict]) -> list[dict]:
    """Generations that ended up with at least one trace in the figure."""
    seen = {t.get("meta", {}).get("model") for t in traces}
    return [g for g in GENERATIONS if g["tag_id"] in seen]


def _filter_buttons(present_gens: list[dict]) -> str:
    btns = ['<button data-model="all" class="active">All</button>',
            '<button data-model="classical">Classical (TPE)</button>']
    for g in present_gens:
        btns.append(f'<button data-model="{g["tag_id"]}">{g["tag"]}</button>')
    return ("<div class=\"group-filter tracker-hero-filter\">\n"
            "  <span class=\"gf-label\">Show:</span>\n  "
            + "\n  ".join(btns) + "\n</div>")


def _plot_html(traces: list[dict]) -> str:
    """Self-contained Plotly snippet (filter buttons + plot div + init JS)."""
    div_id = "tracker-hero-plot-" + uuid.uuid4().hex[:8]
    layout = {
        "height": 700, "margin": {"l": 60, "r": 30, "t": 40, "b": 60},
        "xaxis": {"title": "Cumulative training time (hours)", "range": [0, 24]},
        "yaxis": {"title": "Best val_bpb (lower is better)"},
        "legend": {"x": 0.99, "xanchor": "right", "y": 0.99, "yanchor": "top",
                   "bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#ddd",
                   "borderwidth": 1, "font": {"size": 10}},
        "hovermode": "x unified",
    }
    config = {"responsive": True, "displayModeBar": False}
    filter_html = _filter_buttons(_present_generations(traces))
    init_js = (
        f"(function(){{\n"
        f"  const plotEl = document.getElementById('{div_id}');\n"
        f"  Plotly.newPlot('{div_id}', "
        f"{json.dumps(traces, separators=(',', ':'))}, "
        f"{json.dumps(layout, separators=(',', ':'))}, "
        f"{json.dumps(config, separators=(',', ':'))});\n"
        "  const root = plotEl.closest('.plot-container') || document;\n"
        "  const allBtns = root.querySelectorAll('.tracker-hero-filter button');\n"
        "  const allBtn = root.querySelector('.tracker-hero-filter button[data-model=\"all\"]');\n"
        "  const specBtns = Array.from(allBtns).filter(b => b.dataset.model !== 'all');\n"
        "  function applyVisibility(activeSet){\n"
        "    const data = plotEl.data || [];\n"
        "    const vis = data.map(t => activeSet === 'all' || activeSet.has((t.meta||{}).model) ? true : 'legendonly');\n"
        "    Plotly.restyle(plotEl, {visible: vis});\n"
        "  }\n"
        "  allBtns.forEach(btn => btn.addEventListener('click', () => {\n"
        "    const m = btn.dataset.model;\n"
        "    if (m === 'all'){\n"
        "      allBtns.forEach(b => b.classList.add('active'));\n"
        "      applyVisibility('all');\n"
        "      return;\n"
        "    }\n"
        "    if (allBtn.classList.contains('active')){\n"
        "      allBtn.classList.remove('active');\n"
        "      specBtns.forEach(b => b.classList.toggle('active', b === btn));\n"
        "    } else {\n"
        "      btn.classList.toggle('active');\n"
        "    }\n"
        "    const active = new Set();\n"
        "    specBtns.forEach(b => { if (b.classList.contains('active')) active.add(b.dataset.model); });\n"
        "    if (active.size === 0 || active.size === specBtns.length){\n"
        "      allBtns.forEach(b => b.classList.add('active'));\n"
        "      applyVisibility('all');\n"
        "    } else {\n"
        "      allBtn.classList.remove('active');\n"
        "      applyVisibility(active);\n"
        "    }\n"
        "  }));\n"
        "})();"
    )
    return (f'{filter_html}\n'
            f'<div id="{div_id}" class="plotly-graph-div" '
            f'style="height:700px;width:100%;"></div>\n'
            f'<script>{init_js}</script>')


def inject_into_html(snippet_html: str) -> None:
    html = HTML_PATH.read_text()
    # Replace the placeholder inside #tracker-hero-container
    pattern = re.compile(
        r'(<div class="plot-container" id="tracker-hero-container">)'
        r'.*?'
        r'(</div>\s*<h3 id="tracker-slope")',
        flags=re.DOTALL,
    )
    new = pattern.sub(rf"\1\n{snippet_html}\n\2", html, count=1)
    if new == html:
        raise RuntimeError("hero-container placeholder not found; index.html "
                           "may have changed shape")
    HTML_PATH.write_text(new)


def main() -> None:
    traces, info = build_traces()
    if not traces:
        print("No traces built (no completed seeds anywhere). Bailing.")
        return
    print(f"Built {len(traces)} traces ({len(traces)//3} method-generation curves).")
    for line in info:
        print(line)
    snippet = _plot_html(traces)
    inject_into_html(snippet)
    print(f"\nInjected hero plot into {HTML_PATH}")


if __name__ == "__main__":
    main()
