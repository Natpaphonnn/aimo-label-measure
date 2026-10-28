#!/usr/bin/env python3
"""Per-case runtime of the submitted wb-K4 bundle on the validation server, from the stderr logs of the temporary queue
(runs/val_logs/<id>.log, copied from the 'Full Log' download; not in git). One row per model (gpt-oss per effort): cases, load,
generation and call time (median [min, max]), cut x K. The no-probe ablation (b018b198/24e6d3d6) gives the cost of a
prior-only case. Writes notes/results/runtime_val.md.

Usage: python analysis/runtime_table.py
"""
import pathlib, re, sys
from collections import defaultdict
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from parse_val_logs import D, parse  # noqa: E402
LOGS = {"main": ("064c9648.log", "b018b198.log"), "small": ("ef31bdf0.log", "24e6d3d6.log")}


def loads(f):   # 'case: load X' lines in call order (parse() keeps gen but not load)
    return [float(m[1]) for m in re.finditer(r"case: load ([\d.]+)s", (D / f).read_text(errors="replace"))]


def span(f):    # wall clock of the whole job from the first to the last log timestamp (h:m:s, same day)
    ts = re.findall(r"\[aimo (\d\d):(\d\d):(\d\d)\]", (D / f).read_text(errors="replace"))
    s = [int(h) * 3600 + int(m) * 60 + int(x) for h, m, x in ts]
    return (s[-1] - s[0]) / 60


def fmt(v):
    return f"{np.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]"


L = ["# Validation-server runtime of wb-K4 (analysis/runtime_table.py)", "",
     "Seconds per case, median [min, max]. call = are_robust wall time inside the process (model load + sampling + probe + stage 2);",
     "the harness adds process start and imports per case. Logs of the temporary queue (RTX PRO 6000 Blackwell, per-case fresh process).", ""]
for track, (wb, nop) in LOGS.items():
    cases, ld = parse(wb), loads(wb); assert len(cases) == len(ld)
    g = defaultdict(lambda: defaultdict(list))
    for c, l in zip(cases, ld):
        key = c["model"] + ("" if c["effort"] == "default" else f" ({c['effort']})")
        for k, v in (("load", l), ("gen", c["gen"]), ("call", c["sec"])): g[key][k].append(v)
        g[key]["budget"] = f"{c['k']} x {c['cut']}"
    nop_c = [c["sec"] for c in parse(nop) if c["loaded"] == "none"]
    L += [f"## {track}: {len(cases)} cases, whole job {span(wb):.0f} min (no-probe ablation: {span(nop):.0f} min)", "",
          "| model | cases | K x cut | load | generation | call |", "|---|---|---|---|---|---|"]
    for key, d in g.items():
        L.append(f"| {key} | {len(d['call'])} | {d['budget']} | {fmt(d['load'])} | {fmt(d['gen'])} | {fmt(d['call'])} |")
    allc = [c["sec"] for c in cases]
    L += [f"| all | {len(allc)} | | | | {fmt(allc)}; mean {np.mean(allc):.1f} |",
          f"| prior-only case (no-probe ablation, model not loaded) | {len(nop_c)} | | | | {fmt(nop_c)} |", ""]
pool = defaultdict(lambda: defaultdict(list))   # Main + Small together, gpt-oss efforts merged (the report's appendix table)
for track, (wb, _) in LOGS.items():
    for c, l in zip(parse(wb), loads(wb)):
        for k, v in (("load", l), ("gen", c["gen"]), ("call", c["sec"])): pool[c["model"]][k].append(v)
L += ["## pooled (Main + Small; report Table A1)", "", "| model | cases | load | generation | call |", "|---|---|---|---|---|"]
L += [f"| {m} | {len(d['call'])} | {fmt(d['load'])} | {fmt(d['gen'])} | {fmt(d['call'])} |" for m, d in pool.items()]
(ROOT / "notes/results/runtime_val.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
