#!/usr/bin/env python3
"""Problem files (jsonl: problem_id, problem, answer=None, labels) for feature extraction on the official public data.
Writes data/official/train_main_v2_problems.jsonl (36 statements, per-config labels) and
data/official/math_deepseek_problems.jsonl (137 MATH statements, qwen3-8b:low = DeepSeek-R1-0528-Qwen3-8B labels, old rule)."""
import hashlib, json, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/official"; OUT.mkdir(parents=True, exist_ok=True)
load = lambda n: pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/hf" / n / "data").glob("*.parquet"))], ignore_index=True)
pid = lambda s: "off-" + hashlib.sha256(s.encode()).hexdigest()[:12]
tr = load("train-main-v2"); rows = {}
for r in tr.itertuples():
    row = rows.setdefault(r.problem, {"problem_id": pid(r.problem), "problem": r.problem, "answer": None, "labels": {}})
    row["labels"][f"{r.model_id}|{r.reasoning_effort}"] = {"is_robust": (None if pd.isna(r.is_robust) else bool(r.is_robust)), "max_drop": float(r.max_drop)}
(OUT / "train_main_v2_problems.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in rows.values()))
ma = load("augmented-sample-math-agg").drop_duplicates("original_problem")
recs = [{"problem_id": pid(r.original_problem), "problem": r.original_problem, "answer": None, "dataset_id": r.dataset_id,
         "labels": {"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B|default": {"is_robust": bool(r.model_is_robust), "decay": float(r.absolute_accuracy_decay), "base": float(r.base_accuracy), "types": r.permutation_type}}}
        for r in ma.itertuples()]
(OUT / "math_deepseek_problems.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in recs))
print(f"train: {len(rows)} statements; math: {len(recs)} statements ({sum(r['labels']['deepseek-ai/DeepSeek-R1-0528-Qwen3-8B|default']['is_robust'] for r in recs)} robust)")
