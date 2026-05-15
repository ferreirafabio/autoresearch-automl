"""Re-sort the Fig 1 Plotly trace array in docs/index.html so legend entries
appear in best-val_bpb order (best first). Idempotent and safe to re-run after
each round of new traces (e.g. a new Opus model added by
add_opus_demo_traces.py).

The trace array layout is groups of 3 traces per method (lower band, upper
band, mean line). The mean line's name contains the best value in brackets:
"Centaur [Opus 4.6]  [0.9738]". We group traces by `legendgroup`, sort groups
by the parsed best, and stitch the array back together preserving the
internal order of each group.

Usage: python scripts/sort_fig1_legend.py
"""

import json
import re
from pathlib import Path

HTML_PATH = Path(__file__).resolve().parent.parent / "docs" / "index.html"
FIG1_UUID = "267e692c-2c84-4e3f-be11-2f11b0e56769"


def find_trace_array_span(html: str, uuid: str) -> tuple[int, int]:
    """Return (start, end) byte offsets of the JSON trace array inside the
    Plotly.newPlot("<uuid>", [ ... ]) call. End is exclusive of the trailing
    ']'."""
    needle = f'Plotly.newPlot('
    i = html.find(needle)
    while i != -1:
        # advance past 'Plotly.newPlot(' and any whitespace + quoted uuid
        j = i + len(needle)
        m = re.match(r'\s*"([a-f0-9-]+)"\s*,\s*\[', html[j:])
        if m and m.group(1) == uuid:
            start = j + m.end() - 1  # position of '['
            # walk the brackets to find the matching ']'
            depth = 0
            k = start
            in_str = False
            esc = False
            while k < len(html):
                ch = html[k]
                if esc:
                    esc = False
                elif in_str:
                    if ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return start, k + 1
                k += 1
            raise RuntimeError("unterminated array")
        i = html.find(needle, i + len(needle))
    raise RuntimeError(f"no Plotly.newPlot for uuid {uuid}")


_BEST_RE = re.compile(r"\[\s*0?\.(\d{3,5})\s*\]")


def parse_best(name: str) -> float | None:
    """Extract the final '[0.NNNN]' bracketed val_bpb from a trace name."""
    if not name:
        return None
    matches = _BEST_RE.findall(name)
    if not matches:
        return None
    # last bracket wins (handles names like "[Opus 4.6]  [0.9738]")
    return float("0." + matches[-1])


def sort_traces(traces: list[dict]) -> list[dict]:
    """Group by legendgroup, sort groups by best val_bpb (lower first), keep
    internal order. Traces without a legendgroup (or whose mean-line has no
    [best] bracket) sink to the end in original order."""
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    no_group: list[int] = []
    for idx, t in enumerate(traces):
        lg = t.get("legendgroup")
        if lg is None:
            no_group.append(idx)
            continue
        if lg not in groups:
            groups[lg] = []
            order.append(lg)
        groups[lg].append(idx)

    def group_key(lg: str) -> tuple[float, int]:
        # The mean-line trace is the one with a name (band traces have
        # showlegend=False and no name; mean-line has a name and relies on
        # Plotly's default-true showlegend, i.e. shows up as None here).
        for idx in groups[lg]:
            t = traces[idx]
            if t.get("showlegend") is False or not t.get("name"):
                continue
            b = parse_best(t["name"])
            if b is not None:
                return (b, order.index(lg))
            ys = t.get("y") or []
            ys_valid = [y for y in ys if isinstance(y, (int, float))]
            if ys_valid:
                return (float(min(ys_valid)), order.index(lg))
        return (float("inf"), order.index(lg))

    sorted_groups = sorted(order, key=group_key)
    out: list[dict] = []
    for lg in sorted_groups:
        for idx in groups[lg]:
            out.append(traces[idx])
    for idx in no_group:
        out.append(traces[idx])
    return out


def main() -> None:
    html = HTML_PATH.read_text()
    start, end = find_trace_array_span(html, FIG1_UUID)
    raw = html[start:end]
    traces = json.loads(raw)
    print(f"Fig 1: {len(traces)} traces (including bands)")
    groups_before = []
    for t in traces:
        lg = t.get("legendgroup")
        if t.get("showlegend") is not False and t.get("name") and lg:
            groups_before.append((lg, parse_best(t["name"])))
    print(f"  {len(groups_before)} legend groups, before sort:")
    for lg, b in groups_before:
        print(f"    {b}  {lg}")

    sorted_traces = sort_traces(traces)
    new_raw = json.dumps(sorted_traces, separators=(",", ":"))
    new_html = html[:start] + new_raw + html[end:]
    HTML_PATH.write_text(new_html)
    print(f"\nWrote {HTML_PATH} ({len(new_html)} bytes; was {len(html)})")

    after = []
    for t in sorted_traces:
        lg = t.get("legendgroup")
        if t.get("showlegend") is not False and t.get("name") and lg:
            after.append((lg, parse_best(t["name"])))
    print("\nAfter sort (top of legend = best):")
    for lg, b in after:
        print(f"  {b}  {lg}")


if __name__ == "__main__":
    main()
