#!/usr/bin/env python3
"""Join feature outputs with labels into the evaluation-harness table (columns: problem_id, model_id, y, f_*, solve_rate?).

Inputs:
  --labels   jsonl problem files with a `labels` dict ({"<model_id>|<effort>": {"is_robust": bool|null, ...}}) as written by
             tools/build_official_problem_files.py, and/or CSVs from src/selflabel/build_labels.py (columns model_id, problem_id, base, label_band)
  --prefill  one or more *_feats.jsonl from src/features/prefill.py (model_id, problem_id, scalar features)
  --budgeted one or more *.jsonl from src/features/budgeted.py
Output: CSV with f_ prefixed feature columns; rows restricted to (problem, model) pairs that have a label and all requested features.
"""
import argparse, json, pathlib
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--labels", nargs="+", type=pathlib.Path, required=True)
ap.add_argument("--prefill", nargs="*", type=pathlib.Path, default=[]); ap.add_argument("--budgeted", nargs="*", type=pathlib.Path, default=[])
ap.add_argument("--effort", default="default"); ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
PREFILL = ["mean_logprob", "ppl", "min_logprob", "mink10", "mink20", "mink40", "minkpp20", "first_quarter_logprob", "last_quarter_logprob", "zlib_ratio", "chars", "n_problem_tokens", "gen_pos_entropy", "gen_pos_top1", "gen_pos_top5_mass"]
BUDG = ["agree", "agree_forced", "finish_rate", "mean_commit_conf", "n_distinct", "has_nan", "mean_tokens"]
lab = []
for p in a.labels:
    if p.suffix == ".csv":
        df = pd.read_csv(p); df = df.dropna(subset=["label_band"]) if "label_band" in df else df
        for r in df.itertuples(): lab.append({"problem_id": r.problem_id, "model_id": r.model_id, "y": bool(r.label_band), "solve_rate": float(r.base)})
    else:
        for l in p.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            for key, v in r["labels"].items():
                m, eff = key.split("|"); 
                if v.get("is_robust") is None or eff != a.effort: continue
                lab.append({"problem_id": r["problem_id"], "model_id": m, "y": bool(v["is_robust"]), "solve_rate": v.get("base")})
lab = pd.DataFrame(lab).drop_duplicates(["problem_id", "model_id"]); print("labels:", len(lab), "rows;", lab.model_id.value_counts().to_dict())
table = lab
if a.prefill:
    pf = pd.concat([pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()]) for p in a.prefill]).drop_duplicates(["problem_id", "model_id"])
    pf = pf[["problem_id", "model_id"] + [c for c in PREFILL if c in pf]].rename(columns={c: f"f_pre_{c}" for c in PREFILL}); table = table.merge(pf, on=["problem_id", "model_id"])
if a.budgeted:
    bd = pd.concat([pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()]) for p in a.budgeted]).drop_duplicates(["problem_id", "model_id"])
    bd = bd[["problem_id", "model_id"] + [c for c in BUDG if c in bd]].rename(columns={c: f"f_bud_{c}" for c in BUDG}); bd = bd.astype({c: float for c in bd.columns if c.startswith("f_")}); table = table.merge(bd, on=["problem_id", "model_id"])
a.out.parent.mkdir(parents=True, exist_ok=True); table.to_csv(a.out, index=False)
print("table:", len(table), "rows,", [c for c in table.columns if c.startswith("f_")].__len__(), "features ->", a.out)
