#!/usr/bin/env python3
"""Case labels for stage 2 of the two-stage estimator (DECISIONS #44) -> data/labels/case_labels.csv
(model_id, reasoning_effort, problem_id, y, source).
  byeolki : data/seed/byeolki_labels_forced.csv band labels (0.10 / 0.25 over rename/typos/distract, 8k cap, forced answers),
            excluded_low_base rows and band rows dropped (DECISIONS #18);
  official: train-main-v2 (data/hf/train-main-v2, current revision) rows with a non-null is_robust; problem ids as in
            data/official/train_main_v2_problems.jsonl ("off-" + sha256(problem)[:12])."""
import glob, hashlib, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/labels"; OUT.mkdir(parents=True, exist_ok=True)
b = pd.read_csv(ROOT / "data/seed/byeolki_labels_forced.csv")
b = b[(b.excluded_low_base != True) & b.label_band.notna()]
rows = [{"model_id": r.model_id, "reasoning_effort": "default", "problem_id": r.problem_id, "y": bool(r.label_band), "source": "byeolki"} for r in b.itertuples()]
tr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "data/hf/train-main-v2/data/*.parquet")))], ignore_index=True)
rev = (ROOT / "data/hf/train-main-v2/_revision.txt").read_text().strip()[:8]
for r in tr[tr.is_robust.notna()].itertuples():
    rows.append({"model_id": r.model_id, "reasoning_effort": r.reasoning_effort, "problem_id": "off-" + hashlib.sha256(r.problem.encode()).hexdigest()[:12],
                 "y": bool(r.is_robust), "source": f"train-main-v2@{rev}"})
df = pd.DataFrame(rows); df.to_csv(OUT / "case_labels.csv", index=False)
print(df.groupby(["source", "model_id"]).y.agg(["count", "mean"]).round(2).to_string())
