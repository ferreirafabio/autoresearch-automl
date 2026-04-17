"""Generate Plotly traces for Opus 4.6 (Centaur + KA Code) and inject into
docs/index.html fig1. The existing demo has Gemini Pro traces with a specific
format; we mirror that format exactly.

The format is 3 traces per method:
  1. Lower band: {"hoverinfo":"skip","legendgroup":"<name>","line":{"width":0,"color":"<hex>"},"mode":"lines","showlegend":false,"x":[...],"y":[mean-std],"type":"scatter"}
  2. Upper band: {"fill":"tonexty","fillcolor":"<rgba>","hoverinfo":"skip","legendgroup":"<name>","line":{"width":0,"color":"<hex>"},"mode":"lines","showlegend":false,"x":[...],"y":[mean+std],"type":"scatter"}
  3. Mean line: {"hovertemplate":"<b><name></b>...","legendgroup":"<name>","line":{"color":"<hex>","dash":"<default>","width":2.5},"mode":"lines","name":"<name>  [<best>]","showlegend":true,"x":[...],"y":[mean],"type":"scatter"}

Uses the same interpolation as plot_convergence.plot_convergence_walltime
(INTERP_HOURS = linspace(0, 24, 1000), forward-fill last value for completed
seeds) so the demo curves match the paper figures exactly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from scripts.plot_convergence import load_trials, cumulative_walltime


HTML_PATH = Path(__file__).resolve().parent.parent / "docs" / "index.html"
RESULTS_BASE = Path("/work/dlclarge1/ferreira-autoresearch-automl/results")
INTERP_HOURS = np.linspace(0, 24, 1000)

OPUS_METHODS = [
    {
        "dir": RESULTS_BASE / "opus46_benchmark/centaur_claude_opus_4_6",
        "legend": "Centaur [Opus 4.6]",
        "color": "#C62828",
        "alpha": "rgba(198,40,40,0.1)",
    },
    {
        "dir": RESULTS_BASE / "opus46_benchmark/karpathy_agent_claude_opus_4_6",
        "legend": "Karpathy Agent (Code) [Opus 4.6]",
        "color": "#1565C0",
        "alpha": "rgba(21,101,192,0.1)",
    },
]

MIN_BUDGET_FRAC = 0.90
BUDGET_SECONDS = 86400


def build_traces(method_dir: Path, legend: str, color: str, alpha: str) -> list[dict]:
    """Build 3 Plotly traces (lower band, upper band, mean line) for one method."""
    seed_interps = []
    for seed in range(3):
        tf = method_dir / f"seed_{seed}" / "trials.jsonl"
        if not tf.exists():
            continue
        trials = load_trials(tf)
        times, values = cumulative_walltime(trials)
        if not times or values[0] == float("inf"):
            continue
        total_train_s = sum(t.get("wall_time_seconds") or 0.0 for t in trials)
        if total_train_s < BUDGET_SECONDS * MIN_BUDGET_FRAC:
            continue
        right_val = values[-1] if total_train_s >= BUDGET_SECONDS * 0.98 else np.nan
        interp = np.interp(INTERP_HOURS, times, values, left=np.nan, right=right_val)
        seed_interps.append(interp)
    if not seed_interps:
        raise RuntimeError(f"No valid seeds for {legend}")
    aligned = np.array(seed_interps)
    mean = np.nanmean(aligned, axis=0)
    std = np.nanstd(aligned, axis=0)
    best = float(np.nanmin(mean))

    def round5(arr):
        # round to 5 decimals to keep JSON compact and match existing data style
        return [round(float(v), 5) for v in arr]

    x = round5(INTERP_HOURS)
    mean_r = round5(mean)
    lower_r = round5(mean - std)
    upper_r = round5(mean + std)

    lower = {
        "hoverinfo": "skip",
        "legendgroup": legend,
        "line": {"width": 0, "color": color},
        "mode": "lines",
        "showlegend": False,
        "x": x,
        "y": lower_r,
        "type": "scatter",
    }
    upper = {
        "fill": "tonexty",
        "fillcolor": alpha,
        "hoverinfo": "skip",
        "legendgroup": legend,
        "line": {"width": 0, "color": color},
        "mode": "lines",
        "showlegend": False,
        "x": x,
        "y": upper_r,
        "type": "scatter",
    }
    mean_line = {
        "hovertemplate": f"<b>{legend}</b><br>time: %{{x:.1f}}h<br>val_bpb: %{{y:.4f}}<extra></extra>",
        "legendgroup": legend,
        "line": {"color": color, "dash": "dash", "width": 2.5},
        "mode": "lines",
        "name": f"{legend}  [{best:.4f}]",
        "showlegend": True,
        "x": x,
        "y": mean_r,
        "type": "scatter",
    }
    return [lower, upper, mean_line]


def inject_into_fig1(html: str, new_traces: list[dict]) -> str:
    """Find fig1's data array (the one with 'plotly-graph-div' id) and append
    new traces before the closing ']'."""
    # Find the Plotly.newPlot call for fig1: look for the first 'plotly-graph-div'
    # preceded by our fig1 container (we assume only fig1 is in the search area).
    # The data array is the second argument to Plotly.newPlot. We find the
    # literal "Plotly.newPlot(" after 'fig1' and then walk to the second arg.
    m = re.search(r'id="fig1"', html)
    if not m:
        raise RuntimeError("no fig1 container")
    # Find the Plotly.newPlot call after fig1
    start = html.find("Plotly.newPlot(", m.end())
    if start == -1:
        raise RuntimeError("no Plotly.newPlot after fig1")
    # Skip the first string argument (the div ID)
    # After Plotly.newPlot(, we have                        "<id>",                        [
    # Find the second comma after the start that's at depth 0 for parens
    i = start + len("Plotly.newPlot(")
    # Skip whitespace and find first "
    while i < len(html) and html[i] != '"':
        i += 1
    # skip the string
    i += 1
    while i < len(html) and html[i] != '"':
        if html[i] == '\\':
            i += 2
        else:
            i += 1
    i += 1  # past closing quote
    # skip whitespace / commas
    while i < len(html) and html[i] in ' ,\n\t':
        i += 1
    if html[i] != '[':
        raise RuntimeError(f"expected '[' for data array, got {html[i]!r}")
    array_start = i
    # Walk to the matching ']', respecting strings and nested brackets
    depth = 0
    in_str = False
    esc = False
    j = array_start
    while j < len(html):
        c = html[j]
        if esc:
            esc = False
        elif c == '\\':
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    break
        j += 1
    array_end = j  # index of matching ']'

    # Build the injected JSON fragment (comma-prefixed, no surrounding brackets)
    def compact(obj):
        return json.dumps(obj, separators=(',', ':'))
    inject = ',' + ','.join(compact(t) for t in new_traces)
    return html[:array_end] + inject + html[array_end:]


def main():
    html = HTML_PATH.read_text()
    # Purge any existing Opus traces (for idempotency)
    # Remove traces where "legendgroup":"Centaur [Opus 4.6]" or KA Code Opus
    for legend in ("Centaur [Opus 4.6]", "Karpathy Agent (Code) [Opus 4.6]"):
        while True:
            idx = html.find(f'"legendgroup":"{legend}"')
            if idx == -1:
                break
            # walk back to enclosing '{'
            depth = 0; p = idx; in_str = False; esc = False
            while p > 0:
                c = html[p]
                if esc: esc = False
                elif c == '\\': esc = True
                elif c == '"': in_str = not in_str
                elif not in_str:
                    if c == '}': depth += 1
                    elif c == '{':
                        if depth == 0: break
                        depth -= 1
                p -= 1
            start = p
            # walk forward to '}'
            depth = 0; p = idx; in_str = False; esc = False
            while p < len(html):
                c = html[p]
                if esc: esc = False
                elif c == '\\': esc = True
                elif c == '"': in_str = not in_str
                elif not in_str:
                    if c == '{': depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0: break
                p += 1
            end = p + 1
            # also absorb the leading ',' if there's one
            while start > 0 and html[start-1] in ' \n\t':
                start -= 1
            if start > 0 and html[start-1] == ',':
                start -= 1
            html = html[:start] + html[end:]

    # Build new traces
    all_traces = []
    for m in OPUS_METHODS:
        traces = build_traces(m["dir"], m["legend"], m["color"], m["alpha"])
        # Report best val for sanity
        best = min(t["name"] for t in traces if "name" in t)
        print(f"  {m['legend']}: {traces[2]['name']}")
        all_traces.extend(traces)

    html2 = inject_into_fig1(html, all_traces)
    HTML_PATH.write_text(html2)
    print(f"Injected {len(all_traces)} traces into {HTML_PATH}")


if __name__ == "__main__":
    main()
