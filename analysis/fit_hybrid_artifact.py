#!/usr/bin/env python3
"""Fit the artifact that submissions/hybrid-v1 reads (arms: hybrid / agreement / probe) from self-labels + saved traces.

Inputs
  --labels   build_labels.py CSV (model_id, problem_id, label_band, ...); rows with label_band null are dropped
  --records  force_at_cuts.py output (records with `cuts[c] = {answer, correct, natural}`), family original, for the SAME model
  --states   trace_states.py stem (<stem>.npz + .jsonl) for those records; --cut and --k select the test-time budget level
  --model-id key under which the bundle looks the config up (records may carry a local model path)
Feature definitions mirror submissions/hybrid-v1/features.py exactly (agreement over the first k samples at the cut, n_steps =
min(cut, longest sample) - 1, PCA of the chosen state block, per-layer dispersion); the selftest checks that the bundle's manual
logit reproduces sklearn's predict_proba on the fitted artifact.
Selection (grouped CV by problem, 3 seeds): state block (layer x trace|answer) and C for probe/hybrid; C for agreement.
Also runs analysis/ablation_arms.run_arms on the same table (pre-registered comparison, DECISIONS #34) into --report.
Usage: python analysis/fit_hybrid_artifact.py --labels runs/labels/qwen35_amc.csv --records runs/gen/Qwen_Qwen3.5-4B/qwen35_amc_originals_8k.cuts.jsonl \
   --states runs/features/Qwen_Qwen3.5-4B/amc_states --cut 2048 --k 8 --model-id Qwen/Qwen3.5-4B --arm hybrid --out submissions/hybrid-v1/artifacts/hybrid_model.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ablation_arms import N_PCA, build_whitebox, folds_for, run_arms

AGREE_FEATS = ["agree_top", "agree_natural", "n_distinct", "share_answered", "share_finished", "share_nan", "mean_len_frac", "forced_share", "log_steps"]
LAYER_FRACS = [0.25, 0.5, 0.75]


def top_share(xs: list, k: int) -> float:
    xs = [str(x) for x in xs if x is not None]
    if not xs: return 0.0
    _, c = np.unique(xs, return_counts=True); return float(c.max()) / k


def agreement_table(records: list[dict], cut: int, k: int) -> pd.DataFrame:
    """Per (model_id, problem_id): the bundle's agreement features from samples 0..k-1 at `cut`."""
    rows = []
    df = pd.DataFrame([r for r in records if r.get("family", "original") == "original" and r["sample_idx"] < k and str(cut) in r.get("cuts", {})])
    for (m, p), g in df.groupby(["model_id", "problem_id"]):
        if len(g) < k: continue
        g = g.sort_values("sample_idx").head(k)
        cuts = [r["cuts"][str(cut)] for r in g.to_dict("records")]
        final = [c["answer"] for c in cuts]; natural = [c["answer"] if c["natural"] else None for c in cuts]
        finished = [bool(c["natural"]) for c in cuts]; lengths = [min(int(n), cut) for n in g.n_tokens]
        answered = [a for a in final if a is not None]
        n_steps = max(0, min(cut, max(int(n) for n in g.n_tokens)) - 1)
        rows.append({"model_id": m, "problem_id": p, "k": float(k), "agree_top": top_share(final, k), "agree_natural": top_share(natural, k),
                     "n_distinct": float(len(set(str(a) for a in answered))), "share_answered": len(answered) / k, "share_finished": float(np.mean(finished)),
                     "share_nan": sum(1 for a in answered if a == "NaN") / k, "mean_len_frac": float(np.mean(lengths)) / cut, "cut": float(cut),
                     "forced_share": sum(1 for c in cuts if not c["natural"] and c["answer"] is not None) / k, "n_steps": float(n_steps), "log_steps": math.log1p(n_steps)})
    return pd.DataFrame(rows)


def design(arm: str, cands: dict, disp: np.ndarray, layers: list[int], block: str | None, agree: np.ndarray, pca: PCA | None) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    if arm in ("hybrid", "probe"):
        Z = pca.transform(cands[block]); parts.append(Z); names += [f"pc{i}" for i in range(Z.shape[1])]
        parts.append(disp); names += [f"disp_L{l}" for l in layers]
    if arm in ("hybrid", "agreement"):
        parts.append(agree); names += AGREE_FEATS
    if arm == "probe":
        parts.append(agree[:, [AGREE_FEATS.index("log_steps")]]); names += ["log_steps"]
    return np.hstack(parts), names


def cv_acc(arm: str, cands: dict, disp: np.ndarray, layers: list[int], block: str | None, agree: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, seeds=(0, 1, 2)) -> float:
    accs = []
    for s in seeds:
        fold = folds_for(groups, 5, s); pred = np.zeros(len(y), bool)
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            pca = None
            if block:
                pca = PCA(n_components=min(N_PCA, tr.sum() - 1, cands[block].shape[1]), random_state=0).fit(cands[block][tr])
            X, _ = design(arm, cands, disp, layers, block, agree, pca)
            if len(np.unique(y[tr])) < 2: pred[te] = y[tr].mean() >= 0.5; continue
            sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        accs.append((pred == y).mean())
    return float(np.mean(accs))


def fit(labels: pd.DataFrame, records: list[dict], stem: pathlib.Path, cut: int, k: int, model_id: str, arm: str, Cs=(0.03, 0.1, 0.3)) -> tuple[dict, dict]:
    lab = labels.dropna(subset=["label_band"]).copy(); lab["y"] = lab.label_band.astype(bool)
    ag = agreement_table(records, cut, k)
    df = lab.merge(ag.drop(columns=["model_id"]), on="problem_id", how="inner")
    if df.empty: raise SystemExit("no labelled cases with k samples at this cut")
    cands, disp, keep = build_whitebox(df, stem, cut, k)      # drops cases without k states
    layers = [int(x) for x in np.load(str(stem) + ".npz")["layers"]]
    y = keep.y.values.astype(bool); groups = keep.problem_id.values; agree = keep[AGREE_FEATS].values.astype(float)
    grid = [(b, C) for b in (list(cands) if arm != "agreement" else [None]) for C in Cs]
    scores = {f"{b}|C={C}": cv_acc(arm, cands, disp, layers, b, agree, y, groups, C) for b, C in grid}
    best = max(scores, key=scores.get); block = None if best.startswith("None") else best.split("|")[0]; C = float(best.split("C=")[1])
    pca = PCA(n_components=min(N_PCA, len(y) - 1, cands[block].shape[1]), random_state=0).fit(cands[block]) if block else None
    X, names = design(arm, cands, disp, layers, block, agree, pca)
    sc = StandardScaler().fit(X); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y)
    cfg = {"layer": int(block.split(":")[0][1:]) if block else None, "kind": block.split(":")[1] if block else None,
           "pca_mean": pca.mean_.tolist() if pca else None, "pca_components": pca.components_.tolist() if pca else None,
           "feature_order": names, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(), "coef": m.coef_[0].tolist(),
           "intercept": float(m.intercept_[0]), "threshold": 0.5, "meta": {"n_cases": int(len(y)), "n_problems": int(len(np.unique(groups))), "robust_share": float(y.mean()),
           "cv_acc_selected": scores[best], "selected": best, "cv_grid": scores, "train_acc": float((m.predict(sc.transform(X)) == y).mean())}}
    art = {"arm": arm, "run": {"k": k, "cut": cut, "layers_frac": LAYER_FRACS, "window": int(np.load(str(stem) + ".npz")["window"])}, "configs": {model_id: cfg}}
    report = run_arms(keep, AGREE_FEATS, cands, disp); report.update(cut=cut, k=k, model_id=model_id)
    return art, report


def bundle_prob(art: dict, model_id: str, feats: dict) -> float:
    """What submissions/hybrid-v1/solution.py computes for the same case (duplicated here to keep the selftest independent)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "submissions/hybrid-v1")); import importlib
    sol = importlib.import_module("solution_for_test") if False else None
    cfg = art["configs"][model_id]; arm = art["arm"]; vals = {}
    if arm in ("hybrid", "probe"):
        wb = feats["wb"]; x = np.asarray(wb[cfg["layer"]][f"{cfg['kind']}_mean"], float)
        z = (x - np.asarray(cfg["pca_mean"])) @ np.asarray(cfg["pca_components"]).T; vals.update({f"pc{i}": float(v) for i, v in enumerate(z)})
        vals.update({f"disp_L{l}": float(wb[l]["dispersion"]) for l in wb})
    if arm in ("hybrid", "agreement"): vals.update(feats["agreement"])
    vals["n_steps"] = float(feats["n_steps"]); vals["log_steps"] = math.log1p(vals["n_steps"])
    v = [vals[n] for n in cfg["feature_order"]]
    zz = sum(c * (x - mu) / (s if s else 1.0) for c, x, mu, s in zip(cfg["coef"], v, cfg["scaler_mean"], cfg["scaler_scale"])) + cfg["intercept"]
    return 1 / (1 + math.exp(-zz))


def selftest(tmp: pathlib.Path) -> None:
    rng = np.random.default_rng(1); H, n_p, k, cut = 32, 60, 4, 2048; layers = [3, 7, 11]; cuts = [1024, 2048]
    recs, meta, trace, answer = [], [], [], []
    labels = []
    for pi in range(n_p):
        pid = f"p{pi:03d}"; robust = rng.random() < 0.6; labels.append({"model_id": "M", "problem_id": pid, "label_band": robust, "base": 1.0})
        hidden = rng.normal(size=H) + (1.5 if robust else -1.5) * np.eye(H)[0]
        for s in range(k):
            ans = 7 if (rng.random() < (0.9 if robust else 0.5)) else int(rng.integers(0, 5)); nat = rng.random() < 0.5; n_tok = 900 if nat else 8192
            recs.append({"model_id": "M", "problem_id": pid, "family": "original", "variant": 0, "sample_idx": s, "n_tokens": n_tok, "gold": 7,
                         "cuts": {str(c): {"answer": ans, "correct": ans == 7, "natural": nat} for c in cuts}})
            meta.append({"problem_id": pid, "family": "original", "variant": 0, "sample_idx": s, "n_gen": n_tok})
            trace.append([[hidden + rng.normal(0, 0.3, H) for _ in layers] for _ in cuts]); answer.append([[rng.normal(size=H) for _ in layers] for _ in cuts])
    stem = tmp / "states"; np.savez(str(stem) + ".npz", trace=np.array(trace, np.float16), answer=np.array(answer, np.float16), layers=np.array(layers), cuts=np.array(cuts), window=64)
    (tmp / "states.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    for arm in ("hybrid", "agreement", "probe"):
        art, rep = fit(pd.DataFrame(labels), recs, stem, cut, k, "M", arm)
        cfg = art["configs"]["M"]; print(arm, "selected", cfg["meta"]["selected"], "cv", round(cfg["meta"]["cv_acc_selected"], 3), "train", round(cfg["meta"]["train_acc"], 3))
        # the bundle's manual logit must equal sklearn's probability for a case rebuilt in the bundle's feature format
        ag = agreement_table(recs, cut, k); cands, disp, keep = build_whitebox(pd.DataFrame(labels).assign(y=True), stem, cut, k)
        i = 5; pid = keep.problem_id[i]; row = ag[ag.problem_id == pid].iloc[0]
        feats = {"agreement": {f: float(row[f]) for f in AGREE_FEATS if f != "log_steps"}, "n_steps": row.n_steps,
                 "wb": {l: {"trace_mean": cands[f"L{l}:trace"][i], "answer_mean": cands[f"L{l}:answer"][i], "dispersion": disp[i, j]} for j, l in enumerate(layers)}}
        p_bundle = bundle_prob(art, "M", feats)
        pca = PCA(n_components=len([n for n in cfg["feature_order"] if n.startswith("pc")]), random_state=0).fit(cands[f"L{cfg['layer']}:{cfg['kind']}"]) if cfg["layer"] is not None else None
        X, _ = design(arm, cands, disp, layers, f"L{cfg['layer']}:{cfg['kind']}" if pca else None, keep.merge(ag, on="problem_id")[AGREE_FEATS].values.astype(float), pca)
        z = (np.array(cfg["coef"]) * (X[i] - np.array(cfg["scaler_mean"])) / np.array(cfg["scaler_scale"])).sum() + cfg["intercept"]
        assert abs(p_bundle - 1 / (1 + math.exp(-z))) < 1e-6, (arm, p_bundle, 1 / (1 + math.exp(-z)))
        if arm == "hybrid": assert cfg["meta"]["cv_acc_selected"] > 0.75 and rep["arms"]["hybrid"]["acc_mean"] > rep["arms"]["agreement"]["acc_mean"] - 0.05
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", type=pathlib.Path); ap.add_argument("--records", type=pathlib.Path, nargs="*")
    ap.add_argument("--states", type=pathlib.Path); ap.add_argument("--cut", type=int, default=2048); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B"); ap.add_argument("--arm", choices=["hybrid", "agreement", "probe"], default="hybrid")
    ap.add_argument("--out", type=pathlib.Path); ap.add_argument("--report", type=pathlib.Path); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest and not (a.labels and a.records and a.states): ap.error("--labels, --records and --states are required (or --selftest)")
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as t: selftest(pathlib.Path(t)); sys.exit(0)
    recs = [json.loads(l) for p in a.records for l in p.read_text().splitlines() if l.strip()]
    art, rep = fit(pd.read_csv(a.labels), recs, a.states, a.cut, a.k, a.model_id, a.arm)
    print(json.dumps({"arm": a.arm, **art["configs"][a.model_id]["meta"]}, indent=1, default=float))
    print(json.dumps({"arms": rep["arms"], "paired": rep["paired"]}, indent=1, default=float))
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(art))
    if a.report: a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(rep, indent=1, default=float))
