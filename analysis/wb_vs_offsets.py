#!/usr/bin/env python3
"""Post-hoc check (DECISIONS #54, 2026-10-27; found by the self-review): the fairest no-hidden-state baseline for wb-K4 is stage 2
with the per-model offsets only (class-balanced; equals the K=1 bb row, whose twin is constant), not the 0.5 majority. Same cases,
stage-1 seed 0, stage-2 seeds and cluster bootstrap as analysis/wb_balanced.py at K = 4. Writes notes/results/wb_vs_offsets_v3.md."""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
from run_wb import collect
from wb_two_stage import case_table, crossfit_probe, SEEDS
from wb_balanced import oof_balanced, bacc, cluster_boot
from datapaths import need  # noqa: E402
need(ROOT / "data/labels/case_labels.csv")
base = collect(["qwen", "skywork", "deepseek"]).reset_index(drop=True)
lab = pd.read_csv(ROOT / "data/labels/case_labels.csv")[["model_id", "problem_id", "y"]].drop_duplicates(["model_id", "problem_id"])
df = base.copy(); df["t"] = df.correct.map(lambda v: None if v is None or v != v else float(v)); df["p0"] = np.nan
for m, g in df.groupby("model_id"):
    p, _ = crossfit_probe(g, 0); df.loc[g.index, "p0"] = p.values
cases = case_table(df, 4, "p0").merge(lab, on=["model_id", "problem_id"]).dropna(subset=["wb_mean"]).reset_index(drop=True)
y = cases.y.values.astype(bool); groups = cases.problem_id.values
preds = {arm: [oof_balanced(cases, cols, sd) >= 0.5 for sd in SEEDS] for arm, cols in {"offsets": [], "wb": ["wb_mean", "wb_sd"], "bb": ["bb_mean", "bb_sd"]}.items()}
preds["prior"] = [np.ones(len(y), bool)] * len(SEEDS)        # wb-no-probe in the submitted bundle = the per-model majority (robust)
L = ["# White-box ablation on held-out self-labelled cases (analysis/wb_vs_offsets.py; post hoc, DECISIONS #54)", "",
     f"N = {len(y)} cases, {len(np.unique(groups))} problems, K = 4, balanced accuracy, stage 2 class-balanced, grouped 5-fold CV.",
     "Paired cluster bootstrap over problems, 10,000 resamples per stage-2 seed; p = two-sided share of resamples on the other side of 0.", "",
     "| comparison | seed | delta | 95% CI | p |", "|---|---|---|---|---|"]
for a1, comp in (("wb", "prior"), ("wb", "offsets"), ("bb", "prior"), ("wb", "bb")):
    ds = []
    for i, sd in enumerate(SEEDS):
        bs = cluster_boot(y, {a1: preds[a1][i], comp: preds[comp][i]}, groups, n=10000, seed=i)
        v = bs[a1] - bs[comp]; d = bacc(y, preds[a1][i]) - bacc(y, preds[comp][i]); ds.append(d)
        L.append(f"| {a1} - {comp} | {sd} | {100*d:+.1f} | [{100*np.percentile(v, 2.5):+.1f}, {100*np.percentile(v, 97.5):+.1f}] | {2*min((v <= 0).mean(), (v >= 0).mean()):.3f} |")
    L.append(f"| {a1} - {comp} | mean | {100*np.mean(ds):+.1f} | | |")
L += ["", "Balanced accuracy per seed: " + "; ".join(f"{a} " + "/".join(f"{bacc(y, q):.3f}" for q in preds[a]) for a in ("wb", "bb", "offsets", "prior"))]
(ROOT / "notes/results/wb_vs_offsets_v3.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
