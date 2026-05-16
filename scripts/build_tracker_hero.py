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
    {
        "tag":     "Sonnet 4.6",
        "tag_id":  "sonnet_4_6",
        "base":    RESULTS / "sonnet46_benchmark",
        "centaur": "centaur_claude_sonnet_4_6",
        "ka_code": "karpathy_agent_claude_sonnet_4_6",
        "ka_hps":  "karpathy_agent_hps_claude_sonnet_4_6",
    },
]

INTERP_HOURS = np.linspace(0, 24, 1000)
CAP_BUDGET_S = 86400.0
COMPLETION_THRESHOLD = 0.95  # treat seeds >=95% budget as "done enough"


def _seed_dirs(method_dir: Path) -> list[Path]:
    return sorted([p for p in method_dir.glob("seed_*") if p.name.split("_")[-1].isdigit()],
                  key=lambda p: int(p.name.split("_")[-1]))


def _seed_finals(method_dir: Path) -> dict[int, float]:
    """{seed_num: final val_bpb} for seeds at >= COMPLETION_THRESHOLD budget.
    Used by the per-method slopegraph (section B)."""
    out: dict[int, float] = {}
    if not method_dir.is_dir():
        return out
    for sdir in _seed_dirs(method_dir):
        res = _seed_curve(sdir / "trials.jsonl")
        if not res:
            continue
        curve, budget = res
        if budget < COMPLETION_THRESHOLD:
            continue
        valid = curve[~np.isnan(curve)]
        if valid.size == 0:
            continue
        out[int(sdir.name.split("_")[-1])] = float(valid.min())
    return out


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


_DASH_TO_SVG = {
    "solid":       "",
    "dot":         ' stroke-dasharray="1 4"',
    "dash":        ' stroke-dasharray="5 3"',
    "dashdot":     ' stroke-dasharray="6 2 1 2"',
    "longdash":    ' stroke-dasharray="10 3"',
    "longdashdot": ' stroke-dasharray="10 3 1 3"',
}


def _ls_svg(dash: str) -> str:
    """Small dashed-line icon matching the linestyle used in the hero plot
    for that Claude generation (mirrors the Fig 1 filter-button style)."""
    da = _DASH_TO_SVG.get(dash, "")
    return (f'<svg class="ls-svg" width="28" height="10" aria-hidden="true">'
            f'<line x1="1" y1="5" x2="27" y2="5" stroke="currentColor" '
            f'stroke-width="2"{da}/></svg>')


def _filter_buttons(present_gens: list[dict]) -> str:
    """Buttons hook into the shared initGroupFilters / applyGroupFilter
    machinery via data-target='tracker-hero' so behaviour matches Figs 1-3.
    Each button carries a tiny SVG line icon whose dash pattern matches the
    plot trace's line.dash for that generation (TPE = dotted, Opus 4.6 =
    solid, Opus 4.7 = dashed, future = dashdot etc.)."""
    btns = [
        '<button data-group="all" class="active">All</button>',
        f'<button data-group="classical">Classical (TPE) {_ls_svg("dot")}</button>',
    ]
    for gen_idx, g in enumerate(present_gens):
        dash = DASH_BY_GEN_INDEX[gen_idx % len(DASH_BY_GEN_INDEX)]
        btns.append(f'<button data-group="{g["tag_id"]}">{g["tag"]} {_ls_svg(dash)}</button>')
    return ('<div class="group-filter" data-target="tracker-hero">\n'
            '  <span class="gf-label">Show:</span>\n  '
            + "\n  ".join(btns) + "\n</div>")


def _plot_html(traces: list[dict]) -> str:
    """Self-contained Plotly snippet (filter buttons + plot div + Plotly init).
    Filter click-handling lives in index.html's initGroupFilters() so we get
    the proven Fig 1/2/3 behaviour for free."""
    div_id = "tracker-hero-plot"  # distinct from the h3 anchor id "tracker-hero"
    # Legend below the plot so the chart keeps real estate on mobile.
    layout = {
        "height": 620, "margin": {"l": 55, "r": 20, "t": 30, "b": 60},
        "xaxis": {"title": "Cumulative training time (hours)", "range": [0, 24]},
        "yaxis": {"title": "Best val_bpb (lower is better)"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center",
                   "y": -0.18, "yanchor": "top",
                   "bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#ddd",
                   "borderwidth": 1, "font": {"size": 10},
                   "itemwidth": 50},
        "hovermode": "x unified",
    }
    config = {"responsive": True, "displayModeBar": False}
    filter_html = _filter_buttons(_present_generations(traces))
    return (f'{filter_html}\n'
            f'<div id="{div_id}" class="plotly-graph-div" '
            f'style="height:700px;width:100%;"></div>\n'
            f'<script>Plotly.newPlot("{div_id}", '
            f'{json.dumps(traces, separators=(",", ":"))}, '
            f'{json.dumps(layout, separators=(",", ":"))}, '
            f'{json.dumps(config, separators=(",", ":"))});</script>')


def _slopegraph_panel(method_key: str, gens: list[dict]) -> tuple[str, list[str]] | None:
    """Returns (div_html_snippet, info_lines) for one method panel, or None
    if the method has no data anywhere yet."""
    # seed_finals per gen
    per_gen: list[tuple[dict, dict[int, float]]] = []
    for g in gens:
        sub = g.get(method_key)
        if not sub:
            continue
        finals = _seed_finals(g["base"] / sub)
        if finals:
            per_gen.append((g, finals))
    if len(per_gen) < 1:
        return None

    # Sort seeds for stable color cycling
    all_seeds = sorted({s for _, d in per_gen for s in d})
    x_labels = [g["tag"] for g, _ in per_gen]
    color_cmap = _viridis(len(all_seeds))

    traces = []
    info = []
    for i, s in enumerate(all_seeds):
        ys = [d.get(s) for _, d in per_gen]
        present = [(xl, y) for xl, y in zip(x_labels, ys) if y is not None]
        if not present:
            continue
        # Hide individual seed legend entries — they're noise on mobile and
        # the dots+lines themselves are still drawn. Mean line stays in the
        # legend.
        traces.append({
            "x": [p[0] for p in present],
            "y": [p[1] for p in present],
            "mode": "lines+markers",
            "name": f"seed {s}",
            "line": {"color": color_cmap[i], "width": 1.3, "dash": "dot"},
            "marker": {"size": 7, "color": color_cmap[i]},
            "hovertemplate": f"<b>seed {s}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
            "type": "scatter", "showlegend": False,
            "legendgroup": f"seed_{s}",
        })
    if not traces:
        return None

    # Mean line per generation (only over present seeds)
    mean_y = []
    for _, d in per_gen:
        vals = list(d.values())
        mean_y.append(sum(vals) / len(vals) if vals else None)
    traces.insert(0, {
        "x": x_labels, "y": mean_y,
        "mode": "lines+markers", "name": "mean across seeds",
        "line": {"color": METHOD_COLOR[method_key], "width": 3.5, "dash": "solid"},
        "marker": {"size": 12, "symbol": "diamond", "color": METHOD_COLOR[method_key]},
        "hovertemplate": "<b>mean</b><br>%{x}: %{y:.4f}<extra></extra>",
        "type": "scatter",
    })

    div_id = "tracker-slope-" + method_key + "-" + uuid.uuid4().hex[:6]
    title = METHOD_DISPLAY[method_key]
    n_seeds_panel = len(traces) - 1  # mean trace + per-seed traces
    layout = {
        "title": {"text": f"{title}  ({n_seeds_panel} seeds, dotted = individual)",
                  "font": {"size": 12}, "x": 0.5, "xanchor": "center"},
        "height": 280, "margin": {"l": 55, "r": 20, "t": 40, "b": 35},
        "xaxis": {"title": "", "showgrid": False},
        "yaxis": {"title": "Final val_bpb", "showgrid": True, "gridcolor": "#eee"},
        "legend": {"font": {"size": 9}, "bgcolor": "rgba(255,255,255,0.7)",
                   "orientation": "h", "x": 0.5, "xanchor": "center",
                   "y": -0.05, "yanchor": "top"},
        "showlegend": True,
    }
    config = {"responsive": True, "displayModeBar": False}
    snippet = (f'<div id="{div_id}" class="plotly-graph-div" '
               f'style="height:280px;width:100%;"></div>\n'
               f'<script>Plotly.newPlot("{div_id}", '
               f'{json.dumps(traces, separators=(",", ":"))}, '
               f'{json.dumps(layout, separators=(",", ":"))}, '
               f'{json.dumps(config, separators=(",", ":"))});</script>')
    info.append(f"  slopegraph {method_key}: {len(traces)-1} seeds across {len(per_gen)} generations")
    return snippet, info


def _viridis(n: int) -> list[str]:
    """N hex colors along a viridis-like ramp. Stable for small N."""
    if n <= 1:
        return ["#440154"]
    stops = [(0.0,  0x44, 0x01, 0x54),
             (0.25, 0x3B, 0x52, 0x8B),
             (0.5,  0x21, 0x91, 0x8C),
             (0.75, 0x5E, 0xC9, 0x62),
             (1.0,  0xFD, 0xE7, 0x25)]
    out = []
    for i in range(n):
        t = i / (n - 1)
        for j in range(len(stops) - 1):
            t0, r0, g0, b0 = stops[j]
            t1, r1, g1, b1 = stops[j + 1]
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = int(r0 + (r1 - r0) * a)
                g = int(g0 + (g1 - g0) * a)
                b = int(b0 + (b1 - b0) * a)
                out.append(f"#{r:02X}{g:02X}{b:02X}")
                break
    return out


def build_slopegraph_html() -> tuple[str, list[str]]:
    """3-panel grid: one per method. Each is its own Plotly div."""
    info: list[str] = []
    panels: list[str] = []
    for method_key in ("centaur", "ka_code", "ka_hps"):
        res = _slopegraph_panel(method_key, GENERATIONS)
        if res is None:
            info.append(f"  slopegraph {method_key}: no data yet, skipping panel")
            continue
        snippet, panel_info = res
        info.extend(panel_info)
        panels.append(f'<div class="slope-panel">{snippet}</div>')

    if not panels:
        return "<p>(no slopegraph data yet)</p>", info

    html = ('<div class="slope-grid">\n' + "\n".join(panels) + "\n</div>")
    return html, info


_PLACEHOLDERS = {
    # Leader sits at the very top of the tracker tab; closes when the next
    # element starts (the TOC nav as of 5e18b96, the hero h3 before that).
    "leader": ("tracker-leader-container", "__leader_next__"),
    "hero":   ("tracker-hero-container",   "tracker-slope"),
    "slope":  ("tracker-slope-container",  "tracker-forest"),
    "forest": ("tracker-forest-container", "tracker-cards"),
    # Cards is the last section; close on the wrapping </div><!-- /tab-panel tracker -->.
    "cards":  ("tracker-cards-container",  None),  # special-cased below
}


def build_leader_html(min_seeds: int = 3) -> tuple[str, list[str]]:
    """Current-leader banner: the (method × Claude generation) pair with the
    lowest mean val_bpb, restricted to >= min_seeds completed seeds. Includes
    paired Wilcoxon vs TPE if computable."""
    from scipy import stats
    info: list[str] = []
    tpe_finals = _seed_finals(EXP2_BENCH / "optuna")

    # Collect every (method, gen) with enough completed seeds.
    candidates: list[dict] = []
    for gen in GENERATIONS:
        for method_key in ("centaur", "ka_code", "ka_hps"):
            sub = gen.get(method_key)
            if not sub:
                continue
            finals = _seed_finals(gen["base"] / sub)
            if len(finals) < min_seeds:
                continue
            vals = np.array(list(finals.values()))
            candidates.append({
                "label": f"{METHOD_DISPLAY[method_key]} [{gen['tag']}]",
                "method_key": method_key,
                "gen_tag": gen["tag"],
                "color": METHOD_COLOR[method_key],
                "mean": float(vals.mean()),
                "std":  float(vals.std()) if vals.size > 1 else 0.0,
                "n":    int(vals.size),
                "finals": finals,
            })
    # Always include TPE as a candidate so a strong classical can hold the crown
    if len(tpe_finals) >= min_seeds:
        vals = np.array(list(tpe_finals.values()))
        candidates.append({
            "label": "TPE (classical baseline)",
            "method_key": "tpe",
            "gen_tag": "classical",
            "color": "#2196F3",
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "n":    int(vals.size),
            "finals": tpe_finals,
        })

    if not candidates:
        info.append(f"  no method has >= {min_seeds} completed seeds yet")
        return ('<p class="tracker-leader" style="text-align:center;color:#888;font-style:italic">'
                f'No method has &ge;{min_seeds} completed seeds yet.</p>'), info

    candidates.sort(key=lambda c: c["mean"])
    winner = candidates[0]
    info.append(f"  winner: {winner['label']} @ {winner['mean']:.4f} (n={winner['n']})")

    # Wilcoxon vs TPE
    p_html = ""
    if winner["method_key"] != "tpe":
        common = sorted(set(winner["finals"]) & set(tpe_finals))
        if len(common) >= 2:
            a = np.array([winner["finals"][s] for s in common])
            b = np.array([tpe_finals[s] for s in common])
            try:
                _, p_one = stats.wilcoxon(a, b, zero_method="wilcox", alternative="less")
                p_one = float(p_one)
                p_cls = "sig" if p_one < 0.05 else ("marginal" if p_one < 0.10 else "")
                star = " *" if p_one < 0.10 else ""
                if p_one < 0.05: star = " **"
                p_html = (f'<div class="leader-metric">'
                          f'<span class="label">Δ vs TPE (paired, n={len(common)})</span>'
                          f'<span class="value {p_cls}">{(a.mean()-b.mean()):+.4f}'
                          f', p={p_one:.3f}{star}</span></div>')
                info.append(f"  vs TPE: Δ={(a.mean()-b.mean()):+.4f}, n={len(common)}, one-sided p={p_one:.4f}")
            except Exception:
                pass

    eyebrow = "Current leader" if winner["method_key"] != "tpe" else "TPE still on top"
    note = (f"Best across {len(candidates)} method &times; generation combinations with "
            f"&ge;{min_seeds} completed seeds.")
    banner = (
        f'<div class="tracker-leader">\n'
        f'  <div class="leader-eyebrow">{eyebrow}</div>\n'
        f'  <div class="leader-title" style="border-left-color:{winner["color"]}">'
        f'{winner["label"]}</div>\n'
        f'  <div class="leader-metrics">\n'
        f'    <div class="leader-metric">'
        f'<span class="label">Mean &plusmn; std</span>'
        f'<span class="value">{winner["mean"]:.4f} &plusmn; {winner["std"]:.4f}</span></div>\n'
        f'    <div class="leader-metric">'
        f'<span class="label">Seeds</span>'
        f'<span class="value">{winner["n"]}</span></div>\n'
        f'    {p_html}\n'
        f'  </div>\n'
        f'  <div class="leader-note">{note}</div>\n'
        f'</div>'
    )

    # Compact ranked overview table below the banner.
    def _short(label: str) -> str:
        return (label
                .replace("Karpathy Agent (Code)", "KA Code")
                .replace("Karpathy Agent (14 HPs)", "KA HPs")
                .replace(" [", " · ")
                .replace("]", "")
                .replace("(classical baseline)", "(classical)"))

    table_rows = []
    for i, c in enumerate(candidates):
        is_winner = (i == 0)
        short_label = _short(c["label"])
        # paired-Wilcoxon vs TPE for the table cell (skip if it IS TPE)
        p_str = "&mdash;"
        if c["method_key"] != "tpe":
            common = sorted(set(c["finals"]) & set(tpe_finals))
            if len(common) >= 2:
                a = np.array([c["finals"][s] for s in common])
                b = np.array([tpe_finals[s] for s in common])
                try:
                    _, p_one = stats.wilcoxon(a, b, zero_method="wilcox", alternative="less")
                    p_one = float(p_one)
                    star = ""
                    cls = ""
                    if p_one < 0.05: star, cls = " **", "p-sig"
                    elif p_one < 0.10: star, cls = " *", "p-marginal"
                    p_str = f'<span class="{cls}">n={len(common)}, p={p_one:.3f}{star}</span>'
                except Exception:
                    p_str = f"n={len(common)}, (n.a.)"
            else:
                p_str = f"n={len(common)}, (too few)"
        row_class = ' class="overview-winner"' if is_winner else ''
        marker = " &#9733;" if is_winner else ""
        table_rows.append(
            f'<tr{row_class}>'
            f'<td><span class="dot" style="background:{c["color"]}"></span>{short_label}{marker}</td>'
            f'<td>{c["n"]}</td>'
            f'<td><strong>{c["mean"]:.4f}</strong> &plusmn; {c["std"]:.4f}</td>'
            f'<td>{p_str}</td>'
            f'</tr>'
        )

    overview = (
        '<div class="tracker-overview">\n'
        '  <table class="overview-table">\n'
        '    <thead><tr><th>Method</th><th>Seeds</th>'
        '<th>Mean &plusmn; std</th><th>One-sided p vs TPE</th></tr></thead>\n'
        f'    <tbody>{"".join(table_rows)}</tbody>\n'
        '  </table>\n'
        '</div>'
    )

    combined = (
        '<div class="tracker-headline">\n'
        f'{banner}\n'
        f'{overview}\n'
        '</div>'
    )
    return combined, info


def build_cards_html() -> tuple[str, list[str]]:
    """Section D: one card per Claude generation, with a small per-method
    table inside (n seeds, mean +/- std, paired Wilcoxon p vs TPE) and the
    last-updated date for that generation."""
    from datetime import datetime, timezone
    from scipy import stats

    info: list[str] = []
    tpe_finals = _seed_finals(EXP2_BENCH / "optuna")

    cards_html: list[str] = []
    for gen in GENERATIONS:
        rows_html: list[str] = []
        last_mtime: float = 0.0
        any_data = False
        for method_key in ("centaur", "ka_code", "ka_hps"):
            sub = gen.get(method_key)
            if not sub:
                rows_html.append(_card_row(METHOD_DISPLAY[method_key], None, None, None, None, METHOD_COLOR[method_key]))
                continue
            method_dir = gen["base"] / sub
            m_finals = _seed_finals(method_dir)
            if not m_finals:
                rows_html.append(_card_row(METHOD_DISPLAY[method_key], 0, None, None, None, METHOD_COLOR[method_key]))
                continue
            any_data = True
            vals = np.array(list(m_finals.values()))
            mean = float(vals.mean())
            std = float(vals.std()) if vals.size > 1 else 0.0
            n = vals.size
            # mtime tracker
            for sdir in method_dir.glob("seed_*"):
                p = sdir / "trials.jsonl"
                if p.is_file():
                    last_mtime = max(last_mtime, p.stat().st_mtime)
            # paired Wilcoxon vs TPE
            common = sorted(set(tpe_finals) & set(m_finals))
            if len(common) >= 2:
                a = np.array([m_finals[s] for s in common])
                b = np.array([tpe_finals[s] for s in common])
                try:
                    _, p_one = stats.wilcoxon(a, b, zero_method="wilcox", alternative="less")
                    p_str = f"{float(p_one):.3f}"
                except Exception:
                    p_str = "-"
            else:
                p_str = f"n={len(common)}"
            rows_html.append(_card_row(METHOD_DISPLAY[method_key], n, mean, std, p_str, METHOD_COLOR[method_key]))

        if last_mtime > 0:
            updated = datetime.fromtimestamp(last_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
            updated_html = f'<span class="card-updated">Last updated {updated}</span>'
        else:
            updated_html = '<span class="card-updated">No data yet</span>'

        body_class = "tracker-card" if any_data else "tracker-card tracker-card-empty"
        cards_html.append(
            f'<div class="{body_class}">\n'
            f'  <div class="card-header">\n'
            f'    <span class="card-title">{gen["tag"]}</span>\n'
            f'    {updated_html}\n'
            f'  </div>\n'
            f'  <table class="card-table">\n'
            f'    <thead><tr><th>Method</th><th>Seeds</th><th>Mean &plusmn; std</th><th>p vs TPE</th></tr></thead>\n'
            f'    <tbody>{"".join(rows_html)}</tbody>\n'
            f'  </table>\n'
            f'</div>'
        )
        info.append(f"  card {gen['tag']}: any_data={any_data}, last_updated_mtime={last_mtime}")

    tpe_note = ""
    if tpe_finals:
        tpe_mean = float(np.mean(list(tpe_finals.values())))
        tpe_std = float(np.std(list(tpe_finals.values()))) if len(tpe_finals) > 1 else 0.0
        tpe_note = (f'<p class="cards-tpe-note">Classical reference (TPE): '
                    f'{tpe_mean:.4f} &plusmn; {tpe_std:.4f} '
                    f'across {len(tpe_finals)} seeds.</p>')

    return (f'<div class="tracker-cards-grid">\n' + "\n".join(cards_html) + f'\n</div>\n{tpe_note}', info)


def _card_row(method: str, n: int | None, mean: float | None,
              std: float | None, p: str | None, color: str) -> str:
    if n is None:
        return (f'<tr class="card-row-empty">'
                f'<td><span class="dot" style="background:{color}"></span>{method}</td>'
                f'<td>-</td><td>-</td><td>-</td></tr>')
    if n == 0 or mean is None:
        return (f'<tr class="card-row-empty">'
                f'<td><span class="dot" style="background:{color}"></span>{method}</td>'
                f'<td>0</td><td>-</td><td>-</td></tr>')
    ms = f"{mean:.4f} &plusmn; {std:.4f}" if std is not None else f"{mean:.4f}"
    p_disp = p or "-"
    # Highlight significant beat (p<0.05)
    try:
        p_f = float(p_disp)
        cls = "p-sig" if p_f < 0.05 else ("p-marginal" if p_f < 0.10 else "")
        if cls:
            p_disp = f'<span class="{cls}">{p_disp}</span>'
    except (ValueError, TypeError):
        pass
    return (f'<tr>'
            f'<td><span class="dot" style="background:{color}"></span>{method}</td>'
            f'<td>{n}</td><td>{ms}</td><td>{p_disp}</td></tr>')


def build_forest_html() -> tuple[str, list[str]]:
    """Section C: paired Wilcoxon Δ vs TPE, one bar per Claude-gen × method."""
    from scipy import stats

    info: list[str] = []
    tpe_finals = _seed_finals(EXP2_BENCH / "optuna")
    if not tpe_finals:
        return "<p>(TPE has no completed seeds yet)</p>", ["forest: no TPE data"]

    rows: list[dict] = []
    for gen in GENERATIONS:
        for method_key in ("centaur", "ka_code", "ka_hps"):
            sub = gen.get(method_key)
            if not sub:
                continue
            m_finals = _seed_finals(gen["base"] / sub)
            if not m_finals:
                continue
            common = sorted(set(tpe_finals) & set(m_finals))
            if len(common) < 2:
                info.append(f"  {METHOD_DISPLAY[method_key]} [{gen['tag']}] vs TPE: "
                            f"only {len(common)} paired seed(s), skipping")
                continue
            a = np.array([m_finals[s] for s in common])
            b = np.array([tpe_finals[s] for s in common])
            diffs = a - b
            delta = float(diffs.mean())
            signs_neg = int((diffs < 0).sum())
            try:
                _, p_one = stats.wilcoxon(a, b, zero_method="wilcox", alternative="less")
            except Exception:
                p_one = float("nan")
            rows.append({
                "label": f"{METHOD_DISPLAY[method_key]} [{gen['tag']}]",
                "color": METHOD_COLOR[method_key],
                "delta": delta,
                "n": len(common),
                "signs": f"{signs_neg}/{len(common)}",
                "p_one": float(p_one),
                "gen": gen["tag_id"],
            })
            info.append(f"  {METHOD_DISPLAY[method_key]} [{gen['tag']}] vs TPE: "
                        f"n={len(common)}, Δ={delta:+.4f}, signs={signs_neg}/{len(common)}, "
                        f"one-sided p={p_one:.4f}")

    if not rows:
        return "<p>(no paired seeds yet for any method × generation)</p>", info

    # Sort by Δ ascending so the biggest improvement (most negative) is at top
    rows.sort(key=lambda r: r["delta"])

    labels = [r["label"] for r in rows]
    deltas = [r["delta"] for r in rows]
    colors = [r["color"] if r["delta"] < 0 else "#9E9E9E" for r in rows]
    p_labels = [
        f"n={r['n']}  p={r['p_one']:.3f}"
        + (" **" if r["p_one"] < 0.05 else (" *" if r["p_one"] < 0.10 else ""))
        for r in rows
    ]

    # Shorten labels for mobile: "Karpathy Agent (Code) [Opus 4.6]" → "KA Code · Opus 4.6"
    short_labels = []
    for r in rows:
        s = (r["label"]
             .replace("Karpathy Agent (Code)", "KA Code")
             .replace("Karpathy Agent (14 HPs)", "KA HPs")
             .replace(" [", " · ")
             .replace("]", ""))
        short_labels.append(s)

    bar = {
        "type": "bar", "orientation": "h",
        "x": deltas, "y": short_labels,
        "marker": {"color": colors, "opacity": 0.85},
        "text": p_labels, "textposition": "outside",
        "cliponaxis": False,
        "hovertemplate": "%{y}<br>Δ=%{x:+.4f}<br>%{text}<extra></extra>",
    }
    xmax = max(abs(d) for d in deltas) * 1.9 if deltas else 0.005
    layout = {
        "height": max(260, 60 * len(rows) + 120),
        "margin": {"l": 130, "r": 80, "t": 55, "b": 50},
        "xaxis": {"title": {"text": "Δ = mean(method) − mean(TPE), lower is better",
                            "font": {"size": 10}},
                  "zeroline": True, "zerolinecolor": "#666", "zerolinewidth": 1,
                  "range": [-xmax, xmax], "tickfont": {"size": 9}},
        "yaxis": {"automargin": True, "tickfont": {"size": 10}},
        "showlegend": False,
        "annotations": [
            {"x": -xmax * 0.95, "y": 1.10, "xref": "x", "yref": "paper",
             "text": "← method beats TPE", "showarrow": False,
             "font": {"color": "#2E7D32", "size": 9}},
            {"x":  xmax * 0.95, "y": 1.10, "xref": "x", "yref": "paper",
             "text": "TPE beats method →", "showarrow": False,
             "font": {"color": "#C62828", "size": 9}},
        ],
    }
    config = {"responsive": True, "displayModeBar": False}
    div_id = "tracker-forest-" + uuid.uuid4().hex[:6]
    snippet = (f'<div id="{div_id}" class="plotly-graph-div" '
               f'style="height:{layout["height"]}px;width:100%;"></div>\n'
               f'<script>Plotly.newPlot("{div_id}", '
               f'{json.dumps([bar], separators=(",", ":"))}, '
               f'{json.dumps(layout, separators=(",", ":"))}, '
               f'{json.dumps(config, separators=(",", ":"))});</script>')
    return snippet, info


def inject_into_html(snippet_html: str, marker: str) -> None:
    """Replace the inner content of one placeholder plot-container in
    docs/index.html, leaving its wrapping div intact. Uses a function-based
    sub so JSON backslash escapes (e.g. \\u003c) in the snippet aren't
    interpreted as regex backrefs."""
    container_id, next_anchor = _PLACEHOLDERS[marker]
    html = HTML_PATH.read_text()
    if next_anchor is None:
        # cards section: ends at the tracker tab-panel close marker
        close = r'(</div><!--\s*/tab-panel\s+tracker\s*-->)'
    elif next_anchor == "__leader_next__":
        # leader section: matches the next element (either <nav class="toc">
        # or directly <h3 id="tracker-hero">) after the closing </div>
        close = r'(</div>\s*<(?:nav|h3))'
    else:
        close = rf'(</div>\s*<h3 id="{next_anchor}")'
    # Match either <div class="plot-container" id="..."> or <div id="...">
    # (leader banner has no plot-container class)
    pattern = re.compile(
        rf'(<div(?:\s+class="plot-container")?\s+id="{container_id}">)'
        r'.*?'
        + close,
        flags=re.DOTALL,
    )
    if not pattern.search(html):
        raise RuntimeError(f"{marker}-container placeholder not found; "
                           "index.html may have changed shape")
    def repl(m: re.Match) -> str:
        return f"{m.group(1)}\n{snippet_html}\n{m.group(2)}"
    new = pattern.sub(repl, html, count=1)
    HTML_PATH.write_text(new)


def main() -> None:
    leader_html, leader_info = build_leader_html()
    print("Current-leader banner:")
    for line in leader_info:
        print(line)
    inject_into_html(leader_html, "leader")
    print("  Injected leader banner.")

    traces, info = build_traces()
    if not traces:
        print("No traces built for hero (no completed seeds). Bailing.")
        return
    print(f"\nHero: built {len(traces)} traces ({len(traces)//3} method-generation curves).")
    for line in info:
        print(line)
    inject_into_html(_plot_html(traces), "hero")
    print(f"  Injected hero plot.")

    slope_html, slope_info = build_slopegraph_html()
    print(f"\nSlopegraph (section B):")
    for line in slope_info:
        print(line)
    inject_into_html(slope_html, "slope")
    print(f"  Injected slopegraph panels.")

    forest_html, forest_info = build_forest_html()
    print(f"\nForest plot (section C):")
    for line in forest_info:
        print(line)
    inject_into_html(forest_html, "forest")
    print(f"  Injected forest plot.")

    cards_html, cards_info = build_cards_html()
    print(f"\nSummary cards (section D):")
    for line in cards_info:
        print(line)
    inject_into_html(cards_html, "cards")
    print(f"  Injected summary cards.")


if __name__ == "__main__":
    main()
