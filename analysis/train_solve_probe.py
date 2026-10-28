#!/usr/bin/env python3
"""Hidden-state probe for a model's own solve rate (and, separately, for the robust label) with a layer sweep.

Inputs: one or more *_hidden.npz from src/features/prefill.py (problem_id, last_token[L+1,H], mean_problem[L+1,H]) for ONE
model, plus a labels CSV/JSONL giving per-problem targets (solve_rate in [0,1] and/or y bool).
Protocol: problem-grouped 5-fold CV x 3 seeds; ridge regression (solve rate) / logistic (label) on standardized
features with PCA inside the fold (n_components = min(64, n_train/3)); report per-layer out-of-fold Pearson r (solve rate)
and accuracy (label) for last-token and mean-pooled views, plus the best band average. Self-test on synthetic data.
Usage: python analysis/train_solve_probe.py --hidden runs/features/M/pool_hidden.npz --labels data/seed/x.csv --model M [--target solve_rate|y]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def oof_predict(X: np.ndarray, t: np.ndarray, groups: np.ndarray, kind: str, seeds=(0, 1, 2), n_splits=5, alpha=10.0, C=0.1) -> np.ndarray:
    preds = np.zeros((len(seeds), len(t)))
    for si, s in enumerate(seeds):
        rng = np.random.default_rng(s); ug = np.unique(groups); shuffled = {g: rng.random() for g in ug}
        order = np.argsort([shuffled[g] for g in groups], kind="stable"); inv = np.empty_like(order); inv[order] = np.arange(len(order))
        Xs, ts, gs = X[order], t[order], groups[order]; p = np.zeros(len(t))
        for tr, te in GroupKFold(n_splits=min(n_splits, len(ug))).split(Xs, ts, gs):
            sc = StandardScaler().fit(Xs[tr]); pca = PCA(n_components=min(64, max(2, len(tr) // 3))).fit(sc.transform(Xs[tr]))
            A, B = pca.transform(sc.transform(Xs[tr])), pca.transform(sc.transform(Xs[te]))
            if kind == "ridge":
                p[te] = Ridge(alpha=alpha).fit(A, ts[tr]).predict(B)
            else:
                if len(np.unique(ts[tr])) < 2: p[te] = ts[tr].mean(); continue
                p[te] = LogisticRegression(C=C, max_iter=3000).fit(A, ts[tr]).predict_proba(B)[:, 1]
        preds[si] = p[inv]
    return preds.mean(0)

def sweep(H: np.ndarray, t: np.ndarray, groups: np.ndarray, kind: str) -> list[dict]:
    out = []
    for L in range(H.shape[1]):
        p = oof_predict(H[:, L], t, groups, kind)
        if kind == "ridge":
            r = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0; out.append({"layer": L, "r": r, "mae": float(np.abs(p - t).mean())})
        else:
            out.append({"layer": L, "acc": float(((p >= 0.5) == t.astype(bool)).mean())})
    return out

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--hidden", nargs="+", type=pathlib.Path); ap.add_argument("--labels", type=pathlib.Path)
    ap.add_argument("--model", default=None); ap.add_argument("--target", default="solve_rate", choices=["solve_rate", "y"]); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    if not a.selftest and not (a.hidden and a.labels): ap.error("--hidden and --labels are required (or --selftest)")
    if a.selftest:
        rng = np.random.default_rng(0); n, L, Hd = 150, 6, 64
        groups = np.array([f"p{i}" for i in range(n)]); t = rng.uniform(0, 1, n)
        H = rng.normal(size=(n, L, Hd)); H[:, 3, :5] += (t[:, None] - 0.5) * 3   # layer 3 carries the signal
        res = sweep(H, t, groups, "ridge"); print(json.dumps(res)); assert res[3]["r"] > 0.7 and res[0]["r"] < 0.4, "selftest: signal layer should stand out"; print("selftest OK"); return
    lab = pd.read_csv(a.labels) if a.labels.suffix == ".csv" else pd.DataFrame([json.loads(l) for l in a.labels.read_text().splitlines() if l.strip()])
    if a.model: lab = lab[lab.model_id == a.model]
    ids, last, mean = [], [], []
    for p in a.hidden:
        z = np.load(p, allow_pickle=False); ids += list(z["problem_id"]); last.append(z["last_token"]); mean.append(z["mean_problem"])
    last, mean = np.concatenate(last).astype(np.float32), np.concatenate(mean).astype(np.float32); idx = {pid: i for i, pid in enumerate(ids)}
    col = "base" if a.target == "solve_rate" and "base" in lab else ("solve_rate" if a.target == "solve_rate" else "label_band" if "label_band" in lab else "y")
    lab = lab.dropna(subset=[col]); lab = lab[lab.problem_id.isin(idx)]
    t = lab[col].values.astype(float); groups = lab.problem_id.values; rows = [idx[p] for p in groups]
    kind = "ridge" if a.target == "solve_rate" else "logit"
    res = {"n": int(len(lab)), "target": col, "last_token": sweep(last[rows], t, groups, kind), "mean_problem": sweep(mean[rows], t, groups, kind)}
    key = "r" if kind == "ridge" else "acc"
    for view in ["last_token", "mean_problem"]:
        best = max(res[view], key=lambda d: d[key]); print(f"{view}: best layer {best['layer']} {key}={best[key]:.3f}; per-layer {key}: " + " ".join(f"{d[key]:.2f}" for d in res[view]))
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
