#!/usr/bin/env python3
"""Pre-registered evaluation harness for feature sets (decision #4).

Given a table with columns [problem_id, model_id, y (bool), feature columns..., optional solve_rate], report
  (i) pooled grouped-CV accuracy (folds by problem_id),
  (ii) WITHIN-model accuracy: a separate classifier per model (grouped CV),
  (iii) difficulty-controlled: features + solve_rate covariate vs solve_rate alone (does the feature add signal?),
  (iv) baselines: per-model prior (majority within model, leave-one-problem-out) and solve-rate-only,
  (v) paired bootstrap CI for (method - baseline) accuracy and exact McNemar p-value.
Classifier: standardized logistic regression with C chosen by inner grouped CV. Accuracy threshold 0.5 unless
`--prior-threshold` is given (then the decision threshold is set so the predicted positive rate matches the prior).
Self-test: python analysis/eval_features.py --selftest
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0, Cs=(0.01, 0.1, 1.0)) -> np.ndarray:
    """Out-of-fold P(y=1) with inner C selection; folds grouped by problem."""
    # seeded random assignment of whole problems to folds (sklearn's GroupKFold balances sizes deterministically and ignores
    # row order, so before 2026-10-16 every seed produced the same folds)
    rng = np.random.default_rng(seed); ug = np.unique(groups); k = min(n_splits, len(ug))
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k)); fold = np.array([fold_of[g] for g in groups])
    oof = np.zeros(len(y))
    for f in range(k):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        best, bestC = -1, Cs[0]
        if len(np.unique(groups[tr])) >= 3 and len(Cs) > 1:
            inner = GroupKFold(n_splits=min(3, len(np.unique(groups[tr]))))
            for C in Cs:
                acc = []
                for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                    if len(np.unique(y[tr][itr])) < 2: continue
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr][itr], y[tr][itr])
                    acc.append((m.predict(X[tr][ite]) == y[tr][ite]).mean())
                if acc and np.mean(acc) > best: best, bestC = np.mean(acc), C
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean(); continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=bestC, max_iter=2000)).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

def loo_prior(df: pd.DataFrame) -> np.ndarray:
    """Per-model majority label, leaving the row's own problem out."""
    out = np.zeros(len(df), dtype=bool)
    for i, r in enumerate(df.itertuples()):
        o = df[(df.model_id == r.model_id) & (df.problem_id != r.problem_id)]
        out[i] = (o.y.mean() >= 0.5) if len(o) else True
    return out

def insample_prior(df: pd.DataFrame) -> np.ndarray:
    """Per-model majority label over ALL rows of that model (ties -> robust). Slightly optimistic for the baseline, hence the
    conservative comparison: unlike loo_prior it does not flip when a model's robust share is near 0.5 (DECISIONS #19)."""
    maj = df.groupby("model_id").y.mean().ge(0.5)
    return df.model_id.map(maj).values.astype(bool)

def mde_paired(n: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true accuracy difference (method - baseline) detectable with the exact two-sided McNemar test at the given
    power, for n paired cases and a fixed probability that a case is discordant. Under H1 a discordant case is a win with
    probability q = (r + delta) / (2r); power = P(reject) with the number of discordant cases ~ Binomial(n, r)."""
    from scipy.stats import binom
    r = max(discordant_rate, 1.0 / n)
    m_vals = np.arange(0, n + 1); pm = binom.pmf(m_vals, n, r)
    # rejection region per m: smallest b such that two-sided exact p < alpha
    crit = {}
    for m in m_vals[pm > 1e-6]:
        crit[m] = next((b for b in range(m + 1) if m > 0 and min(1.0, 2 * binom.sf(b - 1, m, 0.5)) < alpha), None)
    for delta in np.arange(0.0, r + 1e-9, 0.0025):
        q = min(1.0, (r + delta) / (2 * r)); pw = 0.0
        for m, w in zip(m_vals, pm):
            if w <= 1e-6 or crit.get(m) is None: continue
            pw += w * binom.sf(crit[m] - 1, m, q)
        if pw >= power:
            return float(delta)
    return float("nan")   # not detectable even if the method wins every discordant case

def paired_stats(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """a = method, b = baseline. Bootstrap CI of accuracy difference (resampling cases) and exact McNemar."""
    rng = np.random.default_rng(seed); d = correct_a.astype(int) - correct_b.astype(int); n = len(d)
    boots = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    b = int(((correct_a) & (~correct_b)).sum()); c = int(((~correct_a) & (correct_b)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    disc = (b + c) / n
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "wins": b, "losses": c,
            "mcnemar_p": float(p), "n": int(n), "discordant_rate": float(disc), "mde80": mde_paired(n, disc)}

def evaluate(df: pd.DataFrame, features: list[str], solve_col: str | None = None, n_splits: int = 5, seeds=(0, 1, 2)) -> dict:
    df = df.reset_index(drop=True); y = df.y.values.astype(bool); g = df.problem_id.values
    res = {"n": int(len(df)), "n_problems": int(df.problem_id.nunique()), "models": sorted(df.model_id.unique()), "prior_rate": float(y.mean())}
    prior = loo_prior(df); res["baseline_prior_acc"] = float((prior == y).mean())
    prior_ins = insample_prior(df); res["baseline_prior_insample_acc"] = float((prior_ins == y).mean())
    def run(cols, name, within=False):
        accs = []; pred_last = None; preds = []
        for s in seeds:
            if within:
                p = np.zeros(len(df))
                for m in res["models"]:
                    mask = (df.model_id == m).values
                    if mask.sum() < 10: p[mask] = y[mask].mean(); continue
                    p[mask] = grouped_oof(df.loc[mask, cols].values.astype(float), y[mask], g[mask], n_splits, s)
            else:
                X = df[cols].values.astype(float)
                p = grouped_oof(X, y, g, n_splits, s)
            pred = p >= 0.5; accs.append((pred == y).mean()); pred_last = pred; preds.append(pred)
        st = paired_stats(pred_last == y, prior == y)
        res[name] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "vs_prior": st,
                     "vs_prior_insample": paired_stats(pred_last == y, prior_ins == y),
                     "per_model": {m: {"n": int((df.model_id == m).sum()), "acc_mean_seeds": float(np.mean([(pp[(df.model_id == m).values] == y[(df.model_id == m).values]).mean() for pp in preds])),
                                       "prior_insample_acc": float((prior_ins[(df.model_id == m).values] == y[(df.model_id == m).values]).mean())} for m in res["models"]}}
        return pred_last
    # pooled with model one-hot (so the classifier can learn per-model offsets)
    onehot = pd.get_dummies(df.model_id, prefix="m").astype(float); df = pd.concat([df, onehot], axis=1); oh = list(onehot.columns)
    run(features + oh, "pooled_features")
    run(features + oh, "within_model_features", within=True)
    if solve_col:
        run([solve_col] + oh, "solve_rate_only")
        p_solve = run([solve_col] + oh, "solve_rate_only_within", within=True)
        p_both = run(features + [solve_col] + oh, "features_plus_solve_rate_within", within=True)
        res["features_add_to_solve_rate"] = paired_stats(p_both == y, p_solve == y)
    return res

def selftest() -> None:
    rng = np.random.default_rng(0); rows = []
    for pid in range(120):
        diff = rng.normal()  # problem difficulty (shared across models)
        for m in ["A", "B", "C"]:
            off = {"A": 0.5, "B": 0.0, "C": -0.5}[m]; solve = 1 / (1 + math.exp(-(off - diff + rng.normal(0, 0.5))))
            signal = rng.normal(); y = (solve > 0.6) ^ (signal > 1.2) if rng.random() < 0.9 else rng.random() < 0.5
            rows.append({"problem_id": f"p{pid}", "model_id": m, "y": bool(y), "f_difficulty": diff + rng.normal(0, 0.3), "f_signal": signal, "f_noise": rng.normal(), "solve_rate": solve})
    df = pd.DataFrame(rows)
    r = evaluate(df, ["f_difficulty", "f_signal", "f_noise"], solve_col="solve_rate")
    print(json.dumps(r, indent=1))
    assert r["pooled_features"]["acc_mean"] > r["baseline_prior_acc"], "pooled should beat prior on synthetic data"
    print("selftest OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--table", type=pathlib.Path); ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--solve-col", default=None); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()
    if not a.selftest and a.table is None: ap.error("--table is required (or --selftest)")
    if a.selftest: selftest(); sys.exit(0)
    df = pd.read_csv(a.table) if a.table.suffix == ".csv" else pd.read_parquet(a.table)
    feats = a.features or [c for c in df.columns if c.startswith("f_")]
    r = evaluate(df, feats, a.solve_col); print(json.dumps(r, indent=1))
    if a.out: a.out.write_text(json.dumps(r, indent=1))
