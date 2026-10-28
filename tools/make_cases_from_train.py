#!/usr/bin/env python3
"""Convert the public train set (train-main-v2 parquet) into Codabench-style cases/labels for run_local.py.

Only labelled rows are kept (is_robust is not null). Writes data/train-cases/{input/cases.jsonl,reference/labels.jsonl}.
"""
import hashlib, json, pathlib, sys
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = sorted((ROOT / "data/hf/train-main-v2/data").glob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in src], ignore_index=True)
df = df[df["is_robust"].notna()].copy()
out = ROOT / "data" / "train-cases"; (out / "input").mkdir(parents=True, exist_ok=True); (out / "reference").mkdir(parents=True, exist_ok=True)
cases, labels = [], []
for _, r in df.iterrows():
    cid = "train-" + hashlib.sha256(json.dumps([r.model_id, r.reasoning_effort, r.problem]).encode()).hexdigest()[:16]
    case = {"id": cid, "model_id": r.model_id, "problem": r.problem}
    if r.reasoning_effort != "default":
        case["reasoning_effort"] = r.reasoning_effort
    cases.append(case); labels.append({"id": cid, "is_robust": bool(r.is_robust)})
(out / "input/cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
(out / "reference/labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
print(f"wrote {len(cases)} labelled cases to {out}; robust={sum(l['is_robust'] for l in labels)}")
