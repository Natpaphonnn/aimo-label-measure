#!/usr/bin/env python3
"""Two-stage white-box solve-rate estimator and its pre-registered ablation (DECISIONS #44).

Stage 1 (per model, white-box): a sample-level correctness probe on the residual stream at the test-time cut. Inputs are the
trace_states.py arrays (trace = mean over the last 64 generated tokens, answer = state of the token that emitted the last token)
at layers L/4, L/2, 3L/4; the block (layer x kind), PCA size and C are chosen by inner grouped CV on the training problems only
(log-loss). Labels are gold-answer correctness of the sample's final answer at the cut (natural, else forced), so only problems
with a known answer train stage 1. Predictions are cross-fitted: every sample's p_correct comes from a probe that never saw its
problem (grouped folds by problem).
Black-box twin of stage 1: s_i = share of the other K - 1 samples whose answer equals sample i's answer (0 if i has no answer).
Stage 2 (shared over models): logistic regression on [mean s, sd s] over the first K samples of the case + model one-hot, C = 1,
grouped 5-fold CV by problem x 3 seeds. Arms: prior (in-sample per-model majority), bb (twin), wb (probe), hybrid (both blocks).
Paired statistics (exact McNemar, bootstrap CI, MDE80): wb vs bb and wb vs prior (primary), hybrid vs bb, bb vs prior.
Efficiency: the same at K = 1, 2, 4, 8.

Inputs (one --model block per model, repeated):
  --model MODEL_ID --records r1.jsonl [r2 ...] --states stem1 [stem2 ...] --cut C [--cuts-file f.cuts.jsonl ...]
  records: generation output (problem_id, sample_idx, gold, answer, finish_reason, forced_answer, max_tokens);
  states:  trace_states.py stems (<stem>.npz + <stem>.jsonl), the cut C must be one of the npz cuts;
  cuts-file: force_at_cuts output for records generated with a larger cap (answer at cut C), optional.
  --labels CSV(model_id, problem_id, y)   case labels for stage 2 (byeolki band labels, train-main-v2).
Output: JSON with stage-1 diagnostics (sample AUC of probe vs twin) and the stage-2 table; --out-cases writes per-case OOF scores.
Self-test: python analysis/wb_two_stage.py --selftest
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_features import insample_prior, paired_stats

KS = (1, 2, 4, 8)
PCA_SIZES = (8, 32)
CS = (0.01, 0.1)
SEEDS = (0, 1, 2)


def folds_for(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); ug = np.unique(groups); k = min(k, len(ug))
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k)); return np.array([fold_of[g] for g in groups])


# ---------------------------------------------------------------- data
def load_model_block(model_id: str, records: list[str], states: list[str], cut: int, cuts_files: list[str]) -> pd.DataFrame:
    """One row per sample: problem_id, sample_idx, answer at the cut, correct (bool or None), and the state blocks
    "S:<kind>:<layer position>" as float32 vectors."""
    recs = {}
    for f in records:
        for l in pathlib.Path(f).read_text().splitlines():
            if l.strip():
                r = json.loads(l); recs[(r["problem_id"], int(r["sample_idx"]))] = r
    at_cut = {}
    for f in cuts_files:
        for l in pathlib.Path(f).read_text().splitlines():
            if l.strip():
                r = json.loads(l); c = (r.get("cuts") or {}).get(str(cut))
                if c is not None: at_cut[(r["problem_id"], int(r["sample_idx"]))] = c
    rows = []; seen = set()   # a sample present in several state files (e.g. the Mac/Kaggle overlap check) is used once, first file wins
    for stem in states:
        z = np.load(stem + ".npz"); meta = [json.loads(l) for l in pathlib.Path(stem + ".jsonl").read_text().splitlines() if l.strip()]
        ci = list(z["cuts"]).index(cut); layers = list(z["layers"])
        for i, m in enumerate(meta):
            key = (m["problem_id"], int(m["sample_idx"])); r = recs.get(key)
            if r is None or m.get("family", "original") != "original" or key in seen: continue
            seen.add(key)
            if key in at_cut:
                ans, ok = at_cut[key]["answer"], at_cut[key]["correct"]
            elif int(r.get("max_tokens", 0)) == cut or r.get("finish_reason") != "length":
                nat = r.get("answer")
                ans = nat if (r.get("finish_reason") != "length" or r.get("forced_answer") is None) else r.get("forced_answer")
                ok = r.get("correct") if ans == nat else r.get("forced_correct")
                if r.get("finish_reason") != "length" and int(m.get("n_gen", 0)) > cut: continue   # finished after the cut: needs a cuts file
            else:
                continue
            row = {"model_id": model_id, "problem_id": m["problem_id"], "sample_idx": int(m["sample_idx"]),
                   "answer": None if ans in (None, "None") else str(ans), "correct": None if ok in (None, "None") else bool(ok in (True, "True")),
                   "n_gen": min(int(m.get("n_gen", 0)), cut)}
            for j, l in enumerate(layers):
                row[f"S:trace:{j}"] = z["trace"][i, ci, j].astype(np.float32); row[f"S:answer:{j}"] = z["answer"][i, ci, j].astype(np.float32)
            rows.append(row)
    return pd.DataFrame(rows)


def twin_scores(df: pd.DataFrame, K: int) -> pd.Series:
    """s_i = share of the other K-1 samples of the case with the same (non-null) answer; samples beyond the first K get NaN."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["model_id", "problem_id"]):
        g = g.sort_values("sample_idx").head(K); a = g.answer.tolist()
        for idx, ai in zip(g.index, a):
            out[idx] = 0.0 if (ai is None or K == 1) else sum(1 for aj in a if aj == ai) - 1
        if K > 1: out[g.index] = out[g.index] / (min(K, len(g)) - 1 if len(g) > 1 else 1)
    return out


# ---------------------------------------------------------------- stage 1
def _blocks(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith("S:")]


def _fit_probe(X: np.ndarray, y: np.ndarray, n_pca: int, C: float):
    sc = StandardScaler().fit(X); Z = sc.transform(X); n = min(n_pca, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=n, random_state=0).fit(Z); W = pca.transform(Z)
    if len(np.unique(y)) < 2: return (sc, pca, None, float(y.mean()))
    lr = LogisticRegression(C=C, max_iter=2000).fit(W, y); return (sc, pca, lr, None)


def _predict_probe(m, X: np.ndarray) -> np.ndarray:
    sc, pca, lr, const = m
    return np.full(len(X), const) if lr is None else lr.predict_proba(pca.transform(sc.transform(X)))[:, 1]


def select_and_fit(train: pd.DataFrame, seed: int) -> tuple:
    """Inner grouped 3-fold CV over (block, PCA size, C) by log-loss on the training problems; refit the best on all of them."""
    y = train.t.values.astype(int); g = train.problem_id.values; f = folds_for(g, 3, seed + 101); best = None
    for b in _blocks(train):
        X = np.stack(train[b].values)
        for n_pca in PCA_SIZES:
            for C in CS:
                p = np.zeros(len(y))
                for k in np.unique(f):
                    tr, te = f != k, f == k; p[te] = _predict_probe(_fit_probe(X[tr], y[tr], n_pca, C), X[te])
                ll = log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1])
                if best is None or ll < best[0]: best = (ll, b, n_pca, C)
    _, b, n_pca, C = best
    return b, n_pca, C, _fit_probe(np.stack(train[b].values), y, n_pca, C)


def crossfit_probe(df: pd.DataFrame, seed: int, n_folds: int = 5) -> tuple[pd.Series, list]:
    """p_correct for EVERY sample of the model (gold or not) from probes trained on other problems' gold-labelled samples."""
    out = pd.Series(np.nan, index=df.index); chosen = []
    f = folds_for(df.problem_id.values, n_folds, seed)
    for k in np.unique(f):
        te = f == k; tr = (~te) & df.t.notna().values
        if tr.sum() < 20: out[df.index[te]] = np.nan; continue
        b, n_pca, C, m = select_and_fit(df[tr], seed); chosen.append((b, n_pca, C))
        out[df.index[te]] = _predict_probe(m, np.stack(df.loc[te, b].values))
    return out, chosen


# ---------------------------------------------------------------- stage 2
def case_table(df: pd.DataFrame, K: int, p_col: str) -> pd.DataFrame:
    tw = twin_scores(df, K); d = df.assign(twin=tw)
    rows = []
    for (m, pid), g in d.groupby(["model_id", "problem_id"]):
        g = g.sort_values("sample_idx").head(K)
        if len(g) < K: continue
        rows.append({"model_id": m, "problem_id": pid, "wb_mean": g[p_col].mean(), "wb_sd": g[p_col].std(ddof=0),
                     "bb_mean": g.twin.mean(), "bb_sd": g.twin.std(ddof=0)})
    return pd.DataFrame(rows, columns=["model_id", "problem_id", "wb_mean", "wb_sd", "bb_mean", "bb_sd"])


def stage2_oof(cases: pd.DataFrame, cols: list[str], seed: int, n_folds: int = 5) -> np.ndarray:
    models = sorted(cases.model_id.unique()); oh = np.stack([(cases.model_id == m).values.astype(float) for m in models], 1)
    X = np.hstack([cases[cols].values.astype(float), oh]); y = cases.y.values.astype(int)
    f = folds_for(cases.problem_id.values, n_folds, seed); p = np.zeros(len(y))
    for k in np.unique(f):
        tr, te = f != k, f == k
        if len(np.unique(y[tr])) < 2: p[te] = y[tr].mean(); continue
        sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def label_curve(cases: pd.DataFrame, sizes=(10, 20, 50, 100), draws: int = 20, seed: int = 0) -> dict:
    """DECISIONS #47b: held-out accuracy when stage 2 is trained on only n labelled cases. Per draw: pick held-out problems (20%),
    train on n cases sampled from the other problems, score the held-out cases. Returns {arm: {n: [mean, sd]}} plus prior."""
    rng = np.random.default_rng(seed); out = {a: {} for a in list(ARMS) + ["prior"]}; probs = cases.problem_id.unique()
    models = sorted(cases.model_id.unique())
    for n in sizes:
        acc = {a: [] for a in out}
        for _ in range(draws):
            te_p = set(rng.choice(probs, max(1, len(probs) // 5), replace=False)); te = cases.problem_id.isin(te_p).values
            pool = np.where(~te)[0]
            if len(pool) < n: continue
            tr = rng.choice(pool, n, replace=False); ytr = cases.y.values[tr].astype(int); yte = cases.y.values[te].astype(bool)
            maj = {m: (cases.y.values[tr][cases.model_id.values[tr] == m].mean() >= 0.5) if (cases.model_id.values[tr] == m).any() else ytr.mean() >= 0.5 for m in models}
            acc["prior"].append(float((np.array([maj[m] for m in cases.model_id.values[te]]) == yte).mean()))
            for a, cols in ARMS.items():
                oh = np.stack([(cases.model_id == m).values.astype(float) for m in models], 1); X = np.hstack([cases[cols].values.astype(float), oh])
                if len(np.unique(ytr)) < 2: pred = np.full(te.sum(), ytr.mean() >= 0.5)
                else:
                    sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), ytr)
                    pred = lr.predict_proba(sc.transform(X[te]))[:, 1] >= 0.5
                acc[a].append(float((pred == yte).mean()))
        for a in out:
            if acc[a]: out[a][n] = [float(np.mean(acc[a])), float(np.std(acc[a]))]
    return out


ARMS = {"bb": ["bb_mean", "bb_sd"], "wb": ["wb_mean", "wb_sd"], "hybrid": ["wb_mean", "wb_sd", "bb_mean", "bb_sd"]}
COMPARISONS = [("wb", "bb"), ("wb", "prior"), ("hybrid", "bb"), ("bb", "prior"), ("hybrid", "prior")]


def consistency(df: pd.DataFrame, K: int = 8) -> pd.Series:
    """Gold-free stage-1 target (DECISIONS #45): sample i's answer equals the plurality answer of the OTHER samples of its case
    (first K by sample_idx; ties count as consistent if i's answer is among the tied; no answer -> False)."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["model_id", "problem_id"]):
        g = g.sort_values("sample_idx").head(K)
        for idx in g.index:
            ai = g.answer[idx]; others = [a for j, a in g.answer.items() if j != idx and a is not None]
            if ai is None or not others: out[idx] = 0.0; continue
            vc = pd.Series(others).value_counts(); out[idx] = float(ai in set(vc[vc == vc.max()].index))
    return out


def run(samples: pd.DataFrame, labels: pd.DataFrame, out_cases: str | None = None, target: str = "correct") -> dict:
    res = {"stage1": {"target": target}, "stage2": {}}
    samples = samples.reset_index(drop=True); pcols = []
    samples["t"] = samples.correct.map(lambda v: None if v is None or v != v else float(v)) if target == "correct" else consistency(samples)
    for s in SEEDS:   # stage-1 cross-fitting per model and seed
        col = f"p{s}"; samples[col] = np.nan; pcols.append(col)
        for m, g in samples.groupby("model_id"):
            p, chosen = crossfit_probe(g, s); samples.loc[g.index, col] = p.values
            if s == SEEDS[0]: res["stage1"].setdefault(m, {})["chosen"] = chosen
    for m, g in samples.groupby("model_id"):   # stage-1 diagnostic: AUC for correctness on gold samples, probe vs twin (K = 8)
        gg = g[g.correct.notna()]; y = gg.correct.astype(int).values; tw = twin_scores(g, 8).loc[gg.index].values
        ok = ~np.isnan(tw)
        res["stage1"][m].update({"n_samples_gold": int(len(gg)), "share_correct": float(y.mean()) if len(y) else None,
            "auc_probe": float(roc_auc_score(y, gg[pcols[0]].values)) if 0 < y.mean() < 1 else None,
            "auc_twin": float(roc_auc_score(y[ok], tw[ok])) if ok.any() and 0 < y[ok].mean() < 1 else None})
    lab = labels[["model_id", "problem_id", "y"]].drop_duplicates(["model_id", "problem_id"])
    for K in KS:
        per_seed = {a: [] for a in ARMS}; last = {}; cases = None
        for s, col in zip(SEEDS, pcols):
            cases = case_table(samples, K, col).merge(lab, on=["model_id", "problem_id"]).dropna(subset=["wb_mean"]).reset_index(drop=True)
            if len(cases) < 20: break
            y = cases.y.values.astype(bool)
            for a, cols in ARMS.items():
                pred = stage2_oof(cases, cols, s) >= 0.5; per_seed[a].append(float((pred == y).mean())); last[a] = pred == y
        if not last: res["stage2"][f"K{K}"] = {"n": 0}; continue
        y = cases.y.values.astype(bool); prior = insample_prior(cases); last["prior"] = prior == y
        r = {"n": int(len(cases)), "n_problems": int(cases.problem_id.nunique()), "robust_share": float(y.mean()),
             "per_model_n": cases.model_id.value_counts().to_dict(), "prior_acc": float(last["prior"].mean())}
        for a in ARMS: r[a] = {"acc_mean": float(np.mean(per_seed[a])), "acc_sd": float(np.std(per_seed[a]))}
        r["paired_last_seed"] = {f"{a}_vs_{b}": paired_stats(last[a], last[b]) for a, b in COMPARISONS}
        r["label_curve"] = label_curve(cases)   # stage-1 scores from the last seed
        res["stage2"][f"K{K}"] = r
        if out_cases and K == max(KS): cases.to_csv(out_cases, index=False)
    return res


# ---------------------------------------------------------------- artifact export (bundle: submissions/hybrid-v1/twostage.py)
def export_artifact(samples: pd.DataFrame, labels: pd.DataFrame, arm: str, K: int, layers: dict, seed: int = 0, run: dict | None = None,
                    class_weight: str | None = None) -> dict:
    """Stage 1 per model refit on ALL gold samples with the inner-CV choice; stage 2 fit on cross-fitted stage-1 scores (stacking),
    so deployment sees the same score distribution stage 2 was trained on. layers[model] = the npz layer ids (block index -> id)."""
    samples = samples.reset_index(drop=True); art = {"format": "two-stage", "arm": arm, "run": run or {}, "stage1": {}, "stage2": {}}
    if "t" not in samples: samples["t"] = samples.correct.map(lambda v: None if v is None or v != v else float(v))
    samples["pcf"] = np.nan
    for m, g in samples.groupby("model_id"):
        p, _ = crossfit_probe(g, seed); samples.loc[g.index, "pcf"] = p.values
        gold = g[g.t.notna()]
        if arm == "bb" or len(gold) < 20: continue
        b, n_pca, C, (sc, pca, lr, const) = select_and_fit(gold, seed)
        if lr is None: continue
        _, kind, j = b.split(":")
        art["stage1"][m] = {"kind": kind, "layer": int(layers[m][int(j)]), "block": b, "n_pca": int(pca.n_components_), "C": C,
                            "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(), "pca_mean": pca.mean_.tolist(),
                            "pca_components": pca.components_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    lab = labels[["model_id", "problem_id", "y"]].drop_duplicates(["model_id", "problem_id"])
    for a in ([arm] if arm == "bb" else [arm, "bb"]):
        cases = case_table(samples, K, "pcf").merge(lab, on=["model_id", "problem_id"])
        cases = cases.dropna(subset=ARMS[a]).reset_index(drop=True)
        models = sorted(cases.model_id.unique()); oh = np.stack([(cases.model_id == mm).values.astype(float) for mm in models], 1)
        X = np.hstack([cases[ARMS[a]].values.astype(float), oh]); y = cases.y.values.astype(int)
        sc2 = StandardScaler().fit(X); lr2 = LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight).fit(sc2.transform(X), y)   # "balanced": #50
        art["stage2"][a] = {"features": ARMS[a], "models": models, "scaler_mean": sc2.mean_.tolist(), "scaler_scale": sc2.scale_.tolist(),
                            "coef": lr2.coef_[0].tolist(), "intercept": float(lr2.intercept_[0]), "threshold": 0.5, "n_cases": int(len(cases))}
        if a == "bb":   # model-agnostic agreement stage 2 for models without labels (Olmo, gpt-oss): no offsets; leave-one-model-out
            Xa = cases[ARMS["bb"]].values.astype(float)   # balanced accuracy 0.60-0.66 at K4 on the held-out model (2026-10-25)
            sca = StandardScaler().fit(Xa); lra = LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight).fit(sca.transform(Xa), y)
            art["stage2"]["bb_any"] = {"features": ARMS["bb"], "models": [], "scaler_mean": sca.mean_.tolist(), "scaler_scale": sca.scale_.tolist(),
                                       "coef": lra.coef_[0].tolist(), "intercept": float(lra.intercept_[0]), "threshold": 0.5, "n_cases": int(len(cases))}
    return art


def check_artifact(art: dict, samples: pd.DataFrame, K: int) -> float:
    """Max |p| difference between the bundle's numpy decision (twostage.py) and sklearn on the exported parameters; also returns
    nothing silently: raises if the bundle cannot score a case."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "submissions/hybrid-v1")); import twostage
    worst = 0.0
    for (m, pid), g in samples.groupby(["model_id", "problem_id"]):
        g = g.sort_values("sample_idx").head(K)
        if len(g) < K: continue
        s1 = art["stage1"].get(m); per = None
        if s1 is not None:
            X = np.stack(g[s1["block"]].values); per = {s1["layer"]: {s1["kind"]: X}}
            Z = (X - np.array(s1["scaler_mean"])) / np.array(s1["scaler_scale"]); W = (Z - np.array(s1["pca_mean"])) @ np.array(s1["pca_components"]).T
            ref = 1 / (1 + np.exp(-(W @ np.array(s1["coef"]) + s1["intercept"])))
            worst = max(worst, float(np.abs(twostage.probe(s1, X) - ref).max()))
        d, info = twostage.decide(art, m, per, g.answer.tolist(), 0.7)
        if "p_robust" in info: assert d == (info["p_robust"] >= 0.5)
    return worst


# ---------------------------------------------------------------- self-test
def selftest() -> None:
    """Synthetic world: a latent 'knows it' direction in the states drives both correctness and robustness; agreement is a
    noisy proxy. The white-box arm must beat the twin and the prior, and the pipeline must run end to end."""
    rng = np.random.default_rng(0); H = 48; rows = []; labs = []
    for m in ("A", "B"):
        u = rng.normal(size=H); u /= np.linalg.norm(u)
        for pi in range(80):
            q = rng.uniform(0.3, 1.0); pid = f"p{pi}"; gold = pi % 7
            for s in range(8):
                ok = rng.random() < q; ans = str(gold) if ok else (str(int(rng.integers(10, 13))) if rng.random() < 0.7 else None)
                v = rng.normal(size=H) + (2.0 if ok else -2.0) * u
                rows.append({"model_id": m, "problem_id": pid, "sample_idx": s, "answer": ans, "correct": bool(ok),
                             "S:trace:0": v.astype(np.float32), "S:answer:0": (v + rng.normal(size=H)).astype(np.float32)})
            labs.append({"model_id": m, "problem_id": pid, "y": bool(q > 0.75 and rng.random() < 0.9)})
    res = run(pd.DataFrame(rows), pd.DataFrame(labs))
    s1 = res["stage1"]["A"]; k8 = res["stage2"]["K8"]; k1 = res["stage2"]["K1"]
    print(json.dumps({"auc_probe": s1["auc_probe"], "auc_twin": s1["auc_twin"], "K8": {a: k8[a]["acc_mean"] for a in ARMS}, "K8_prior": k8["prior_acc"],
                      "K1": {a: k1[a]["acc_mean"] for a in ARMS}}, indent=1))
    assert s1["auc_probe"] > 0.9 and s1["auc_probe"] > s1["auc_twin"]
    assert k8["wb"]["acc_mean"] > k8["prior_acc"] + 0.05 and k8["bb"]["acc_mean"] > k8["prior_acc"]
    df = pd.DataFrame(rows); art = export_artifact(df, pd.DataFrame(labs), "wb", 8, {"A": [7], "B": [7]})
    worst = check_artifact(art, df, 8); print("bundle vs sklearn stage-1 max |dp| =", worst); assert worst < 1e-9
    assert set(art["stage1"]) == {"A", "B"} and set(art["stage2"]) == {"wb", "bb", "bb_any"}
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", action="append", default=[]); ap.add_argument("--records", action="append", nargs="+", default=[])
    ap.add_argument("--states", action="append", nargs="+", default=[]); ap.add_argument("--cut", action="append", type=int, default=[])
    ap.add_argument("--cuts-file", action="append", nargs="*", default=None)
    ap.add_argument("--labels", type=pathlib.Path); ap.add_argument("--out", type=pathlib.Path); ap.add_argument("--out-cases")
    ap.add_argument("--target", choices=["correct", "consistent"], default="correct", help="stage-1 target (#44 primary: correct; #45: consistent)")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if not (a.model and a.labels): ap.error("--model/--records/--states/--cut and --labels are required (or --selftest); analysis/run_wb.py wraps this")
    cf = a.cuts_file or [[] for _ in a.model]
    df = pd.concat([load_model_block(m, r, s, c, f) for m, r, s, c, f in zip(a.model, a.records, a.states, a.cut, cf)], ignore_index=True)
    res = run(df, pd.read_csv(a.labels), a.out_cases, a.target)
    txt = json.dumps(res, indent=1, default=str); print(txt)
    if a.out: a.out.write_text(txt)


if __name__ == "__main__":
    main()
