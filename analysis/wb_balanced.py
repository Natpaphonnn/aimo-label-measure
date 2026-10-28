#!/usr/bin/env python3
"""Stage 2 under label shift (DECISIONS #50): the validation board is balanced (Main 10/21, Small 7/14 robust) while our labelled
cases are 66% robust, and the organizers give no guarantee on the test label mix (FACTS D18). A per-model majority then scores
~0.5, and the useful quantity is balanced accuracy (mean of the two class recalls), which does not depend on the label mix.

Same samples, cross-fitted stage-1 scores and arms as analysis/wb_two_stage.run, but stage 2 is fit with class_weight='balanced'
and scored by balanced accuracy. The prior's balanced accuracy is 0.5 by construction. Uncertainty: cluster bootstrap over
problems (2000 resamples) for each arm minus prior and wb minus bb.

Usage: python analysis/wb_balanced.py --tag v3 [--models qwen skywork deepseek] [--targets correct consistent]
Writes notes/results/wb_balanced_<tag>.md and .json.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import need  # noqa: E402
from run_wb import BLOCKS, collect  # noqa: E402
from wb_two_stage import ARMS, KS, SEEDS, case_table, consistency, crossfit_probe, folds_for  # noqa: E402


def oof_balanced(cases: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    models = sorted(cases.model_id.unique()); oh = np.stack([(cases.model_id == m).values.astype(float) for m in models], 1)
    X = np.hstack([cases[cols].values.astype(float), oh]); y = cases.y.values.astype(int)
    f = folds_for(cases.problem_id.values, 5, seed); p = np.zeros(len(y))
    for k in np.unique(f):
        tr, te = f != k, f == k
        if len(np.unique(y[tr])) < 2: p[te] = 0.5; continue
        sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def bacc(y: np.ndarray, pred: np.ndarray) -> float:
    return 0.5 * (float(pred[y].mean()) + float((~pred[~y]).mean()))


def cluster_boot(y, preds: dict, groups, n=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; out = {k: [] for k in preds}
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); yy = y[s]
        if yy.all() or (~yy).all(): continue
        for k, p in preds.items(): out[k].append(bacc(yy, p[s]))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--models", nargs="+", default=["qwen", "skywork", "deepseek"])
    ap.add_argument("--targets", nargs="+", default=["correct", "consistent"])
    ap.add_argument("--s1-seed", type=int, default=0, help="stage-1 cross-fitting seed (robustness check)"); a = ap.parse_args()
    need(ROOT / "data/labels/case_labels.csv")
    base = collect(a.models).reset_index(drop=True)
    lab = pd.read_csv(ROOT / "data/labels/case_labels.csv")[["model_id", "problem_id", "y"]].drop_duplicates(["model_id", "problem_id"])
    out = {"tag": a.tag, "s1_seed": a.s1_seed, "results": {}}
    L = [f"# Stage 2 under label shift: balanced accuracy `{a.tag}` (analysis/wb_balanced.py)", "",
         "Stage 2 fit with class_weight='balanced'; metric = balanced accuracy (prior = 0.500 by construction). Mean over 3 stage-2 seeds;",
         "CI = 95% cluster bootstrap over problems on the last seed; p = two-sided bootstrap share of resamples on the other side of 0.", ""]
    for t in a.targets:
        df = base.copy(); df["t"] = df.correct.map(lambda v: None if v is None or v != v else float(v)) if t == "correct" else consistency(df)
        df["p0"] = np.nan
        for m, g in df.groupby("model_id"):
            p, _ = crossfit_probe(g, a.s1_seed); df.loc[g.index, "p0"] = p.values
        L += [f"## target = {t}", "", "| K | cases | robust share | bb | wb | hybrid | wb - prior [CI] p | wb - bb [CI] p | hybrid - prior [CI] p |", "|---|---|---|---|---|---|---|---|---|"]
        out["results"][t] = {}
        for K in KS:
            cases = case_table(df, K, "p0").merge(lab, on=["model_id", "problem_id"]).dropna(subset=["wb_mean"]).reset_index(drop=True)
            if len(cases) < 20: continue
            y = cases.y.values.astype(bool); r = {"n": int(len(cases)), "robust_share": float(y.mean())}; last = {}
            for arm, cols in ARMS.items():
                accs = []
                for s in SEEDS:
                    pred = oof_balanced(cases, cols, s) >= 0.5; accs.append(bacc(y, pred)); last[arm] = pred
                r[arm] = float(np.mean(accs))
            for arm in ARMS:   # class recalls (last seed): accuracy at any robust share pi is pi*TPR + (1-pi)*TNR
                r[f"{arm}_tpr"] = float(last[arm][y].mean()); r[f"{arm}_tnr"] = float((~last[arm][~y]).mean())
                r[f"{arm}_breakeven_pi"] = float(r[f"{arm}_tnr"] / (1 - r[f"{arm}_tpr"] + r[f"{arm}_tnr"])) if (1 - r[f"{arm}_tpr"] + r[f"{arm}_tnr"]) > 0 else None
            last["prior"] = np.ones(len(y), bool)
            bs = cluster_boot(y, last, cases.problem_id.values)
            def d(a1, a2):
                v = bs[a1] - bs[a2]; return {"delta": float(bacc(y, last[a1]) - bacc(y, last[a2])), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                          "p": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
            r["wb_vs_prior"] = d("wb", "prior"); r["wb_vs_bb"] = d("wb", "bb"); r["hybrid_vs_prior"] = d("hybrid", "prior"); r["bb_vs_prior"] = d("bb", "prior")
            out["results"][t][f"K{K}"] = r
            f = lambda x: f"{x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] p={x['p']:.3f}"
            L.append(f"| {K} | {r['n']} | {r['robust_share']:.2f} | {r['bb']:.3f} | {r['wb']:.3f} | {r['hybrid']:.3f} | {f(r['wb_vs_prior'])} | {f(r['wb_vs_bb'])} | {f(r['hybrid_vs_prior'])} |")
        L.append("")
    (ROOT / f"notes/results/wb_balanced_{a.tag}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"notes/results/wb_balanced_{a.tag}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
