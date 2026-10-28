#!/usr/bin/env python3
"""Inventory byeolki's MIT-licensed generations: per (model, problem) base accuracy at the 8k budget,
family drops, truncation, restricted to our leakage-filtered pool. Output data/seed/byeolki_base.csv."""
import json, pathlib, collections
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
BY = ROOT / "external/byeolki_aimo-interp/data"
import sys; sys.path.insert(0, str(ROOT / "analysis")); from datapaths import need  # noqa: E402
need(ROOT / "data/pool/problems.jsonl")
pool = {json.loads(l)["problem_id"]: json.loads(l) for l in (ROOT / "data/pool/problems.jsonl").read_text().splitlines() if l.strip()}
rows = []
for f in sorted((BY / "generations").glob("*.jsonl")):
    model = f.stem.replace("_", "/", 1)
    rec = collections.defaultdict(dict)
    for l in f.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        rec[r["problem_id"]][(r["family"], r["variant"])] = r
    for pid, fams in rec.items():
        if pid not in pool: continue
        o = fams.get(("original", 0))
        if o is None: continue
        base = sum(o["correct"]) / len(o["correct"])
        row = {"model_id": model, "problem_id": pid, "source": pool[pid]["source_tag"], "n_orig": len(o["correct"]), "base": base,
               "trunc_rate": sum(o["truncated"]) / len(o["truncated"]), "mean_len": sum(o["lengths"]) / len(o["lengths"]),
               "n_distinct_answers": len(set(a for a in o["answers"] if a is not None)), "n_null_answers": sum(a is None for a in o["answers"])}
        for fam in ["rename", "typos", "distract"]:
            v = fams.get((fam, 0))
            row[f"acc_{fam}"] = sum(v["correct"]) / len(v["correct"]) if v else None
        rows.append(row)
df = pd.DataFrame(rows); (ROOT / "data/seed").mkdir(exist_ok=True); df.to_csv(ROOT / "data/seed/byeolki_base.csv", index=False)
pd.set_option("display.width", 200)
print("rows per model:", df.groupby("model_id").size().to_dict())
print("\nbase accuracy distribution per model (8 samples, 8k cap, forced answers):")
print(df.pivot_table(index="model_id", columns=pd.cut(df.base, [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1.0]), values="problem_id", aggfunc="count", observed=False))
print("\ntruncation rate mean per model:", df.groupby("model_id")["trunc_rate"].mean().round(2).to_dict())
print("mean generated tokens per model:", df.groupby("model_id")["mean_len"].mean().round(0).to_dict())
have = df.dropna(subset=["acc_rename"])
have["max_drop3"] = have["base"] - have[["acc_rename", "acc_typos", "acc_distract"]].min(axis=1)
print("\nproblems with variants (base>0):", have.groupby("model_id").size().to_dict())
print("share with max_drop over the 3 cheap families >= 0.25:", have.groupby("model_id")["max_drop3"].apply(lambda s: (s >= 0.25).mean()).round(2).to_dict())
# frontier: 0 < base < 1
fr = df[(df.base > 0) & (df.base < 1)]
print("\nfrontier problems (0<base<1) per model:", fr.groupby("model_id").size().to_dict(), "; base==1:", df[df.base == 1].groupby("model_id").size().to_dict(), "; base==0:", df[df.base == 0].groupby("model_id").size().to_dict())
# per-problem spread across models
piv = df.pivot_table(index="problem_id", columns="model_id", values="base")
piv = piv.dropna()
print("\nproblems with all 3 models:", len(piv), "; problems where one model has base>=0.875 and another <=0.5:", int(((piv.max(axis=1) >= 0.875) & (piv.min(axis=1) <= 0.5)).sum()))
