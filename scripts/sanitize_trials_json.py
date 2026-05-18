"""Sanitize docs/data/trials.json for public release.

Goal: keep the demo's interactive plots functional while ensuring that any LLM
whose training corpus includes the public URL cannot reconstruct optimal HP
configurations for Climbmix-400B.

Transforms:
  - 14 real HP names -> opaque keys hp_a..hp_n (alphabetical permutation, stable)
  - Continuous HP values -> per-HP decile rank (1..10) across all trials
  - Categorical HP values (WINDOW_PATTERN) -> integer category index per a fixed lookup
  - val_bpb -> integer decile rank (1..10) within (method, seed)
  - Keep tid, success, wt as-is

Outputs:
  - docs/data/trials.json (overwritten with sanitized version)
  - private/trials_original.json (gitignored copy of the original)
  - private/hp_key_map.json (real HP name <-> opaque key, paper-appendix material)
  - private/hp_value_buckets.json (per-HP bucket edges, for paper appendix)
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import quantiles

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_TRIALS = ROOT / "docs" / "data" / "trials.json"
PRIVATE_DIR = ROOT / "private"
ORIGINAL_COPY = PRIVATE_DIR / "trials_original.json"
KEY_MAP_OUT = PRIVATE_DIR / "hp_key_map.json"
VALUE_BUCKETS_OUT = PRIVATE_DIR / "hp_value_buckets.json"

META_KEYS = {"tid", "val_bpb", "success", "wt"}
N_DECILES = 10


def collect_hp_names(data: dict) -> list[str]:
    seen = set()
    for method, seeds in data.items():
        for seed, trials in seeds.items():
            for t in trials:
                for k in t.keys():
                    if k not in META_KEYS:
                        seen.add(k)
    return sorted(seen)


def alpha_label(idx: int) -> str:
    if idx < 26:
        return chr(ord("a") + idx)
    # not expected for 14 HPs
    return f"x{idx}"


def value_to_decile(v: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges, start=1):
        if v <= edge:
            return i
    return N_DECILES


def main() -> None:
    PRIVATE_DIR.mkdir(exist_ok=True)
    raw = json.loads(PUBLIC_TRIALS.read_text())

    # Step 1: keep an offline copy of the original.
    if not ORIGINAL_COPY.exists():
        ORIGINAL_COPY.write_text(json.dumps(raw, indent=None))
        print(f"saved original to {ORIGINAL_COPY}")
    else:
        print(f"original already saved at {ORIGINAL_COPY} (not overwriting)")

    # Step 2: build HP key map: alphabetical real names -> hp_a..hp_n
    hp_names = collect_hp_names(raw)
    key_map = {real: f"hp_{alpha_label(i)}" for i, real in enumerate(hp_names)}
    print(f"hp_names ({len(hp_names)}): {hp_names}")
    print(f"key_map: {key_map}")

    # Step 3: per-HP value buckets. Continuous -> 10-quantile edges. Categorical -> ordered lookup.
    all_values_per_hp: dict[str, list] = defaultdict(list)
    for method, seeds in raw.items():
        for seed, trials in seeds.items():
            for t in trials:
                for k, v in t.items():
                    if k in META_KEYS or v is None:
                        continue
                    all_values_per_hp[k].append(v)

    buckets: dict[str, dict] = {}
    for hp, values in all_values_per_hp.items():
        non_null = [v for v in values if v is not None]
        if not non_null:
            buckets[hp] = {"kind": "empty"}
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
            try:
                edges = quantiles(non_null, n=N_DECILES, method="inclusive")
                buckets[hp] = {"kind": "continuous", "edges": edges, "min": min(non_null), "max": max(non_null)}
            except Exception as e:
                print(f"  {hp}: quantiles failed ({e}); falling back to single-bucket")
                buckets[hp] = {"kind": "constant", "value": non_null[0]}
        else:
            uniq = sorted({str(v) for v in non_null})
            buckets[hp] = {"kind": "categorical", "categories": uniq}

    # Step 4: per-(method, seed) val_bpb deciles.
    bpb_edges: dict[tuple, list[float]] = {}
    for method, seeds in raw.items():
        for seed, trials in seeds.items():
            vals = [t["val_bpb"] for t in trials if t.get("val_bpb") is not None]
            if len(vals) >= N_DECILES:
                bpb_edges[(method, seed)] = quantiles(vals, n=N_DECILES, method="inclusive")
            elif vals:
                bpb_edges[(method, seed)] = sorted(vals)
            else:
                bpb_edges[(method, seed)] = []

    # Step 5: build sanitized JSON.
    sanitized: dict = {}
    for method, seeds in raw.items():
        sanitized[method] = {}
        for seed, trials in seeds.items():
            edges = bpb_edges.get((method, seed), [])
            new_trials = []
            for t in trials:
                rec: dict = {"tid": t.get("tid"), "success": t.get("success"), "wt": t.get("wt")}
                v = t.get("val_bpb")
                rec["val_bpb_decile"] = value_to_decile(v, edges) if (v is not None and edges) else None
                for hp_real, hp_opaque in key_map.items():
                    val = t.get(hp_real)
                    if val is None:
                        rec[hp_opaque] = None
                        continue
                    spec = buckets.get(hp_real, {"kind": "empty"})
                    if spec["kind"] == "continuous":
                        rec[hp_opaque] = value_to_decile(val, spec["edges"])
                    elif spec["kind"] == "categorical":
                        try:
                            rec[hp_opaque] = spec["categories"].index(str(val))
                        except ValueError:
                            rec[hp_opaque] = None
                    elif spec["kind"] == "constant":
                        rec[hp_opaque] = 1
                    else:
                        rec[hp_opaque] = None
                new_trials.append(rec)
            sanitized[method][seed] = new_trials

    # Step 6: write outputs.
    PUBLIC_TRIALS.write_text(json.dumps(sanitized, separators=(",", ":")))
    KEY_MAP_OUT.write_text(json.dumps(key_map, indent=2))
    VALUE_BUCKETS_OUT.write_text(json.dumps(buckets, indent=2, default=str))
    print(f"\nwrote sanitized {PUBLIC_TRIALS} ({PUBLIC_TRIALS.stat().st_size / 1024:.1f} KiB)")
    print(f"wrote private mapping {KEY_MAP_OUT}")
    print(f"wrote private buckets {VALUE_BUCKETS_OUT}")

    # Sanity-check: a sample record.
    first_method = next(iter(sanitized))
    first_seed = next(iter(sanitized[first_method]))
    print(f"\nsample sanitized trial ({first_method} seed {first_seed} trial 0):")
    print(json.dumps(sanitized[first_method][first_seed][0], indent=2))


if __name__ == "__main__":
    main()
