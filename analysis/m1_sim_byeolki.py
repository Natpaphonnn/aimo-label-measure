#!/usr/bin/env python3
"""M1 step (a): GPU-free simulation of the test-time solve-rate route on byeolki's competition-level generations (MIT).
Per (model, problem): TEST-TIME features from original samples [0, K) only, label-free (no gold): answer agreement, truncation,
mean length, number of distinct answers, null answers. LABEL from the disjoint original samples [K, 8) (base accuracy) and the
variant samples (rename/typos/distract), band rule (robust <= 0.10 max drop, non-robust >= 0.25). Budget 8k tokens with forced
answers (byeolki). Only problems with variants (solved at least once in all 8 originals) can be labelled.
Organizer convention (Discord 2026-10-16): never/rarely-solved originals are excluded from labels; argv[2] = minimum held-out base
accuracy (default 0.5, DECISIONS #2/#18; 0 reproduces the first run).
Evaluation: within-model grouped CV vs per-model prior, via analysis/eval_features.py."""
import collections, json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from eval_features import evaluate
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MIN_BASE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TAG = f"k{K}" if MIN_BASE == 0 else f"k{K}_mb{MIN_BASE:g}"
BY = pathlib.Path("external/byeolki_aimo-interp/data/generations")
from datapaths import ROOT, need  # noqa: E402
need(ROOT / "data/pool/problems.jsonl", ROOT / "external/byeolki_aimo-interp/data/generations")
pool = {json.loads(l)["problem_id"] for l in open("data/pool/problems.jsonl") if l.strip()}
rows = []
for f in sorted(BY.glob("*.jsonl")):
    model = f.stem.replace("_", "/", 1); rec = collections.defaultdict(dict)
    for l in f.read_text().splitlines():
        if l.strip():
            r = json.loads(l); rec[r["problem_id"]][r["family"]] = r
    for pid, fam in rec.items():
        if pid not in pool or "original" not in fam or not all(x in fam for x in ("rename", "typos", "distract")): continue
        o = fam["original"]; A, C, Ln, T = o["answers"], o["correct"], o["lengths"], o["truncated"]
        if len(A) < K + 4: continue
        ta = A[:K]; nn = [a for a in ta if a is not None]; cnt = collections.Counter(nn)
        feats = {"f_agree": (cnt.most_common(1)[0][1] / K) if cnt else 0.0, "f_n_distinct": float(len(cnt)), "f_null": float(K - len(nn)),
                 "f_trunc": float(np.mean(T[:K])), "f_loglen": float(np.log1p(np.mean(Ln[:K])))}
        base = float(np.mean(C[K:]))
        drops = {x: base - float(np.mean(fam[x]["correct"])) for x in ("rename", "typos", "distract")}
        md = max(drops.values()); y = True if md <= 0.10 + 1e-9 else (False if md >= 0.25 - 1e-9 else None)
        if base < MIN_BASE - 1e-9: y = None
        rows.append({"model_id": model, "problem_id": pid, "y": y, "solve_rate": base, "test_correct_rate": float(np.mean(C[:K])), **feats})
df = pd.DataFrame(rows); lab = df.dropna(subset=["y"]).copy(); lab["y"] = lab.y.astype(bool)
print(f"K={K} min_base={MIN_BASE}: cases with variants {len(df)}, labelled {len(lab)}; per model:", lab.groupby("model_id").y.agg(["size", "mean"]).round(2).to_dict("index"))
feats = ["f_agree", "f_n_distinct", "f_null", "f_trunc", "f_loglen"]
res = evaluate(lab, feats, solve_col="solve_rate")
out = {k: res[k] for k in ["n", "n_problems", "prior_rate", "baseline_prior_acc", "baseline_prior_insample_acc"]}
for k in ["within_model_features", "pooled_features", "solve_rate_only_within"]:
    out[k] = {"acc": round(res[k]["acc_mean"], 3), "acc_sd_seeds": round(res[k]["acc_sd"], 3),
              "per_model": {m: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in d.items()} for m, d in res[k]["per_model"].items()}}
    for b in ["vs_prior", "vs_prior_insample"]:
        st = res[k][b]; out[k][b] = {"delta": round(st["delta"], 3), "ci95": [round(x, 3) for x in st["ci95"]], "wins_losses": [st["wins"], st["losses"]],
                                     "p": round(st["mcnemar_p"], 3), "mde80": round(st["mde80"], 3)}
print(json.dumps(out, indent=1))
# per-model view: does agreement separate labels within each model?
from scipy.stats import mannwhitneyu
for m, g in lab.groupby("model_id"):
    a, b = g[g.y].f_agree, g[~g.y].f_agree
    auc = mannwhitneyu(a, b).statistic / (len(a) * len(b)) if len(a) and len(b) else float("nan")
    print(f"  {m}: n={len(g)} robust={int(g.y.sum())} AUC(agree)={auc:.3f}  AUC(true solve rate on held-out samples)={mannwhitneyu(g[g.y].solve_rate, g[~g.y].solve_rate).statistic/(len(a)*len(b)):.3f}")
json.dump(out, open(f"notes/results/m1_sim_byeolki_{TAG}.json", "w"), indent=1); lab.to_csv(f"runs/eval/m1_sim_byeolki_{TAG}.csv", index=False)
