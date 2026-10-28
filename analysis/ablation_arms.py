#!/usr/bin/env python3
"""Pre-registered four-arm ablation (DECISIONS #34): prior / agreement-only / probe-only / hybrid on held-out problems.

Inputs
  --cases   CSV with problem_id, model_id, y (bool), f_* agreement features (black-box, from K test-time samples)
  --states  trace_states.py output stem (<stem>.npz + <stem>.jsonl) for the same model, --cut selects the budget level
  --k       number of samples per case used for the white-box aggregates (sample_idx < k), matching the agreement features
Arms (grouped 5-fold CV by problem x 3 seeds, in-sample per-model majority as the floor):
  agreement : standardised logistic regression on f_* + model one-hot (analysis/eval_features.grouped_oof)
  probe     : PCA(16) on one candidate state (layer x {trace, answer}) fitted inside the training fold + dispersion of that
              layer's trace states + one-hot -> logistic; the candidate is chosen by inner grouped 3-fold CV on the training fold
  hybrid    : the probe's candidate selection and PCA, with the f_* features appended -> one logistic model
Paired statistics (exact McNemar, bootstrap CI, MDE80) for hybrid vs agreement (primary), probe vs prior, agreement vs prior,
hybrid vs prior, dispersion-only vs prior. Self-test: python analysis/ablation_arms.py --selftest
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_features import grouped_oof, insample_prior, paired_stats

N_PCA = 16


def folds_for(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); ug = np.unique(groups); k = min(k, len(ug))
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k)); return np.array([fold_of[g] for g in groups])


def fit_predict(Xs_tr: np.ndarray, extra_tr: np.ndarray, y_tr: np.ndarray, Xs_te: np.ndarray, extra_te: np.ndarray, C: float = 0.1) -> np.ndarray:
    """PCA on the state block (fit on train), concatenated with the extra block, standardised logistic regression."""
    n_comp = min(N_PCA, Xs_tr.shape[0] - 1, Xs_tr.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(Xs_tr)
    Ztr = np.hstack([pca.transform(Xs_tr), extra_tr]); Zte = np.hstack([pca.transform(Xs_te), extra_te])
    if len(np.unique(y_tr)) < 2:
        return np.full(len(Zte), float(y_tr.mean()))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Ztr, y_tr)
    return m.predict_proba(Zte)[:, 1]


def probe_oof(cands: dict[str, np.ndarray], extra: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold P(y) with the candidate state block chosen by inner grouped CV on each training fold."""
    fold = folds_for(groups, n_splits, seed); oof = np.zeros(len(y)); chosen = []
    for f in np.unique(fold):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = folds_for(groups[tr], 3, seed + 100); best, best_name = -1.0, next(iter(cands))
        for name, X in cands.items():
            accs = []
            for g in np.unique(inner):
                itr, ite = tr[inner != g], tr[inner == g]
                p = fit_predict(X[itr], extra[itr], y[itr], X[ite], extra[ite]); accs.append(((p >= 0.5) == y[ite]).mean())
            if np.mean(accs) > best:
                best, best_name = float(np.mean(accs)), name
        chosen.append(best_name); X = cands[best_name]
        oof[te] = fit_predict(X[tr], extra[tr], y[tr], X[te], extra[te])
    return oof, chosen


def build_whitebox(cases: pd.DataFrame, stem: pathlib.Path, cut: int, k: int) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Per-case aggregates over samples 0..k-1 at the given cut: candidate blocks {layer:trace|answer -> (n, H)} and dispersion
    per layer (n, n_layers). Rows of `cases` without k samples are dropped (reported)."""
    z = np.load(str(stem) + ".npz"); meta = pd.read_json(str(stem) + ".jsonl", lines=True)
    ci = list(z["cuts"]).index(cut); layers = list(z["layers"])
    meta = meta[(meta.family == "original") & (meta.sample_idx < k)].reset_index().rename(columns={"index": "row"})
    counts = meta.groupby("problem_id").row.count(); ok = set(counts[counts >= k].index)
    keep = cases[cases.problem_id.isin(ok)].reset_index(drop=True)
    cands = {f"L{l}:{kind}": np.zeros((len(keep), z["trace"].shape[-1]), np.float32) for l in layers for kind in ("trace", "answer")}
    disp = np.zeros((len(keep), len(layers)), np.float32)
    for i, pid in enumerate(keep.problem_id):
        rows = meta[meta.problem_id == pid].row.values[:k]
        for j, l in enumerate(layers):
            T = z["trace"][rows, ci, j].astype(np.float32); A = z["answer"][rows, ci, j].astype(np.float32)
            cands[f"L{l}:trace"][i] = T.mean(0); cands[f"L{l}:answer"][i] = A.mean(0)
            Tn = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6); S = Tn @ Tn.T
            disp[i, j] = (1 - S)[~np.eye(k, dtype=bool)].mean() if k > 1 else 0.0
    return cands, disp, keep


def run_arms(df: pd.DataFrame, feats: list[str], cands: dict[str, np.ndarray], disp: np.ndarray, seeds=(0, 1, 2)) -> dict:
    y = df.y.values.astype(bool); g = df.problem_id.values
    onehot = pd.get_dummies(df.model_id, prefix="m").astype(float).values
    prior = insample_prior(df); res = {"n": int(len(df)), "n_problems": int(df.problem_id.nunique()), "robust_share": float(y.mean()),
                                      "prior_acc": float((prior == y).mean()), "arms": {}, "chosen": {}}
    preds = {}
    for name in ("agreement", "probe", "hybrid", "dispersion"):
        accs, last = [], None
        for s in seeds:
            if name == "agreement":
                p = grouped_oof(np.hstack([df[feats].values.astype(float), onehot]), y, g, 5, s)
            elif name == "dispersion":
                p = grouped_oof(np.hstack([disp, onehot]), y, g, 5, s)
            else:
                extra = np.hstack([disp, onehot] + ([df[feats].values.astype(float)] if name == "hybrid" else []))
                p, chosen = probe_oof(cands, extra, y, g, s); res["chosen"].setdefault(name, []).extend(chosen)
            pred = p >= 0.5; accs.append(float((pred == y).mean())); last = pred
        preds[name] = last; res["arms"][name] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs))}
    comp = {"hybrid_vs_agreement (primary)": ("hybrid", "agreement"), "probe_vs_prior": ("probe", None), "agreement_vs_prior": ("agreement", None),
            "hybrid_vs_prior": ("hybrid", None), "dispersion_vs_prior": ("dispersion", None), "hybrid_vs_probe": ("hybrid", "probe")}
    res["paired"] = {k: paired_stats(preds[a] == y, (preds[b] if b else prior) == y) for k, (a, b) in comp.items()}
    return res


def selftest() -> None:
    rng = np.random.default_rng(0); n_p, H = 90, 64; rows = []; states = {}
    for pid in range(n_p):
        for m in ("A", "B"):
            agree = rng.random(); hidden = rng.normal(size=H); y = (0.6 * agree + 0.4 * (hidden[3] > 0) + rng.normal(0, 0.25)) > 0.55
            rows.append({"problem_id": f"p{pid}", "model_id": m, "y": bool(y), "f_agree": agree}); states[(f"p{pid}", m)] = hidden
    df = pd.DataFrame(rows); X = np.stack([states[(p, m)] for p, m in zip(df.problem_id, df.model_id)]).astype(np.float32)
    cands = {"L1:trace": X, "L1:answer": rng.normal(size=X.shape).astype(np.float32)}      # only L1:trace carries signal
    r = run_arms(df, ["f_agree"], cands, rng.random((len(df), 1)).astype(np.float32))
    print(json.dumps({k: v for k, v in r.items() if k != "chosen"}, indent=1, default=float))
    assert r["arms"]["hybrid"]["acc_mean"] > r["arms"]["agreement"]["acc_mean"], "hybrid should add the planted hidden signal"
    assert r["chosen"]["probe"].count("L1:trace") > len(r["chosen"]["probe"]) / 2, "inner CV should pick the informative block"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", type=pathlib.Path); ap.add_argument("--states", type=pathlib.Path)
    ap.add_argument("--cut", type=int, default=2048); ap.add_argument("--k", type=int, default=4); ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    if a.cases is None or a.states is None: ap.error("--cases and --states are required (or --selftest)")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from datapaths import need; need(pathlib.Path(a.cases).resolve())
    cases = pd.read_csv(a.cases); feats = a.features or [c for c in cases.columns if c.startswith("f_")]
    cands, disp, keep = build_whitebox(cases, a.states, a.cut, a.k)
    print(f"cases {len(cases)} -> with {a.k} samples at cut {a.cut}: {len(keep)}")
    r = run_arms(keep, feats, cands, disp); r.update(cut=a.cut, k=a.k, features=feats)
    print(json.dumps(r, indent=1, default=float))
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r, indent=1, default=float))
