#!/usr/bin/env python3
"""What the stage-1 probe reads, and how much of the robustness label any solve-rate estimate can explain (report §5-6).

Per model (samples from analysis/run_wb.py BLOCKS, gold correctness at the test-time cut):
  A. probe AUC per block (layer x state kind), fixed PCA 32 / C 0.1, cross-fitted by problem;
  B. within-problem AUC (sample pairs of the same problem only) vs between-problem rank correlation, probe and twin (K = 8);
  C. truncation control: AUC of 'finished before the cut' alone, and the probe's AUC inside capped / finished samples;
  D. transfer across problem sets: train on the AMC generations, test on the test-time wave (and the reverse);
Per labelled case:
  E. label decomposition: AUC for the label (robust = 1) and pooled CV accuracy (+ model offsets) of the gold solve rate at the cut
     (what stage 1 estimates), the probe mean, the twin mean, and the labellers' own base accuracy at 8k (byeolki generations;
     in-sample for byeolki labels, so an optimistic bound for any solve-rate estimator).

Usage: python analysis/wb_diagnostics.py --tag v1all3 [--models qwen skywork deepseek]
Writes notes/results/wb_diagnostics_<tag>.md and .json.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from run_wb import BLOCKS, collect  # noqa: E402
from wb_two_stage import _blocks, _fit_probe, _predict_probe, crossfit_probe, folds_for, stage2_oof, twin_scores  # noqa: E402
from eval_features import insample_prior  # noqa: E402

from datapaths import need  # noqa: E402
need(ROOT / "data/pool/tt_wave_problems.jsonl", ROOT / "data/seed/byeolki_base.csv")
TT = {json.loads(l)["problem_id"] for l in (ROOT / "data/pool/tt_wave_problems.jsonl").read_text().splitlines() if l.strip()}


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 5 and 0 < y[ok].mean() < 1 else None


def fixed_crossfit(g: pd.DataFrame, block: str, seed: int = 0, n_pca: int = 32, C: float = 0.1) -> np.ndarray:
    X = np.stack(g[block].values); y = g.t.values; f = folds_for(g.problem_id.values, 5, seed); p = np.full(len(g), np.nan)
    for k in np.unique(f):
        te = f == k; tr = (~te) & ~np.isnan(y)
        p[te] = _predict_probe(_fit_probe(X[tr], y[tr].astype(int), n_pca, C), X[te])
    return p


def within_auc(g: pd.DataFrame, col: str):
    """Pooled AUC over (correct, incorrect) pairs of the SAME problem."""
    num = den = 0.0
    for _, h in g.groupby("problem_id"):
        y = h.correct.astype(int).values; s = h[col].values
        if 0 < y.mean() < 1 and not np.isnan(s).any():
            a = roc_auc_score(y, s); w = y.sum() * (1 - y).sum(); num += a * w; den += w
    return float(num / den) if den else None


def transfer(g: pd.DataFrame, block: str, train_tt: bool) -> float | None:
    tt = g.problem_id.isin(TT).values; tr = (tt if train_tt else ~tt) & g.correct.notna().values; te = (~tt if train_tt else tt) & g.correct.notna().values
    if tr.sum() < 50 or te.sum() < 50: return None
    X = np.stack(g[block].values); y = g.correct.astype(float).values
    m = _fit_probe(X[tr], y[tr].astype(int), 32, 0.1); return auc(y[te], _predict_probe(m, X[te]))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--models", nargs="+", default=list(BLOCKS))
    a = ap.parse_args()
    need(ROOT / "data/labels/case_labels.csv")
    df = collect(a.models).reset_index(drop=True)
    df["t"] = df.correct.map(lambda v: np.nan if v is None or v != v else float(v))
    out = {"tag": a.tag, "models": {}}; L = [f"# Stage-1 diagnostics and label decomposition `{a.tag}` (analysis/wb_diagnostics.py)", ""]
    df["p"] = np.nan; df["twin"] = twin_scores(df, 8)
    for m, g in df.groupby("model_id"):
        p, _ = crossfit_probe(g, 0); df.loc[g.index, "p"] = p.values; g = df.loc[g.index]
        gold = g[g.correct.notna()].copy(); y = gold.correct.astype(int).values; r = {"n_gold": int(len(gold)), "share_correct": float(y.mean())}
        r["blocks"] = {}
        for b in _blocks(g):
            pb = fixed_crossfit(g, b); r["blocks"][b] = auc(y, pb[g.correct.notna().values])
        r["auc_probe_selected"] = auc(y, gold.p); r["auc_twin8"] = auc(y, gold.twin)
        r["within_probe"] = within_auc(gold, "p"); r["within_twin"] = within_auc(gold.dropna(subset=["twin"]), "twin")
        pm = gold.assign(c=y.astype(float)).groupby("problem_id").agg(sr=("c", "mean"), p=("p", "mean"), tw=("twin", "mean")).astype(float)
        r["between_rho_probe"] = float(spearmanr(pm.sr, pm.p).correlation); r["between_rho_twin"] = float(spearmanr(pm.sr, pm.tw, nan_policy="omit").correlation)
        fin = (gold.n_gen < gold.n_gen.max()).values   # finished before the cut
        r["share_finished"] = float(fin.mean()); r["auc_finished_only_feature"] = auc(y, fin.astype(float))
        r["auc_probe_in_capped"] = auc(y[~fin], gold.p.values[~fin]); r["auc_probe_in_finished"] = auc(y[fin], gold.p.values[fin])
        r["acc_correct_capped"] = float(y[~fin].mean()) if (~fin).any() else None; r["acc_correct_finished"] = float(y[fin].mean()) if fin.any() else None
        best = max(r["blocks"], key=lambda b: r["blocks"][b] or 0)
        r["transfer_amc_to_tt"] = transfer(g, best, False); r["transfer_tt_to_amc"] = transfer(g, best, True); r["transfer_block"] = best
        out["models"][m] = r
    # --- A-D tables
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    L += ["## A. Probe AUC by block (correct vs incorrect sample; fixed PCA 32, C 0.1; cross-fitted by problem)", ""]
    blocks = _blocks(df); L += ["| model | " + " | ".join(blocks) + " | selected (inner CV) | twin K=8 |", "|---|" + "---|" * (len(blocks) + 2)]
    for m, r in out["models"].items():
        L.append(f"| {m} | " + " | ".join(f(r['blocks'].get(b)) for b in blocks) + f" | {f(r['auc_probe_selected'])} | {f(r['auc_twin8'])} |")
    L += ["", "Block names: S:<kind>:<j>, kind trace = mean of the last 64 generated tokens, answer = state of the last token; j = 0,1,2 -> layer L/4, L/2, 3L/4.", "",
          "## B. Within-problem vs between-problem signal", "",
          "| model | within-problem AUC probe | within-problem AUC twin | between-problem Spearman (problem mean vs solve rate) probe | twin |", "|---|---|---|---|---|"]
    for m, r in out["models"].items():
        L.append(f"| {m} | {f(r['within_probe'])} | {f(r['within_twin'])} | {f(r['between_rho_probe'])} | {f(r['between_rho_twin'])} |")
    L += ["", "## C. Truncation control (samples that hit the cut get a forced answer)", "",
          "| model | share finished before cut | acc finished / capped | AUC of 'finished' alone | probe AUC inside capped | inside finished |", "|---|---|---|---|---|---|"]
    for m, r in out["models"].items():
        L.append(f"| {m} | {r['share_finished']:.2f} | {f(r['acc_correct_finished'])} / {f(r['acc_correct_capped'])} | {f(r['auc_finished_only_feature'])} | {f(r['auc_probe_in_capped'])} | {f(r['auc_probe_in_finished'])} |")
    L += ["", "## D. Transfer across problem sets (probe on the best block, trained on one set, tested on the other)", "",
          "| model | block | AMC -> test-time wave | test-time wave -> AMC |", "|---|---|---|---|"]
    for m, r in out["models"].items():
        L.append(f"| {m} | {r['transfer_block']} | {f(r['transfer_amc_to_tt'])} | {f(r['transfer_tt_to_amc'])} |")
    # --- E. label decomposition
    lab = pd.read_csv(ROOT / "data/labels/case_labels.csv")[["model_id", "problem_id", "y", "source"]].drop_duplicates(["model_id", "problem_id"])
    by = pd.read_csv(ROOT / "data/seed/byeolki_base.csv")[["model_id", "problem_id", "base"]]
    rows = []
    for (m, pid), h in df.groupby(["model_id", "problem_id"]):
        h = h.sort_values("sample_idx").head(8)
        if len(h) < 8: continue
        rows.append({"model_id": m, "problem_id": pid, "solve_cut": h.correct.astype(float).mean() if h.correct.notna().all() else np.nan,
                     "probe_mean": h.p.mean(), "twin_mean": h.twin.mean()})
    cases = pd.DataFrame(rows).merge(lab, on=["model_id", "problem_id"]).merge(by, on=["model_id", "problem_id"], how="left").reset_index(drop=True)
    cases["y"] = cases.y.astype(bool)
    preds = [("solve rate at cut (gold, 8 samples)", "solve_cut"), ("probe mean (stage 1)", "probe_mean"), ("twin mean (agreement)", "twin_mean"),
             ("labellers' base accuracy at 8k (byeolki)", "base")]
    E = {"n_cases": int(len(cases)), "per_model_n": cases.model_id.value_counts().to_dict(), "rows": {}}
    L += ["", f"## E. What explains the robustness label? ({len(cases)} labelled cases with 8 samples; robust = positive class)", "",
          "| predictor | " + " | ".join(f"AUC {m.split('/')[-1]}" for m in sorted(cases.model_id.unique())) + " | pooled AUC | n | CV acc (+model offset) | prior on same cases |",
          "|---|" + "---|" * (cases.model_id.nunique() + 4)]
    for name, col in preds:
        c = cases.dropna(subset=[col]).reset_index(drop=True)
        if len(c) < 20: continue
        per = {m: auc(h.y, h[col]) for m, h in c.groupby("model_id")}
        accs = [float(((stage2_oof(c.assign(_x=c[col]), ["_x"], s) >= 0.5) == c.y.values).mean()) for s in (0, 1, 2)]
        prior = float((insample_prior(c) == c.y.values).mean())
        E["rows"][col] = {"per_model_auc": per, "pooled_auc": auc(c.y, c[col]), "n": int(len(c)), "cv_acc": float(np.mean(accs)), "prior": prior}
        L.append(f"| {name} | " + " | ".join(f(per.get(m)) for m in sorted(cases.model_id.unique())) + f" | {f(auc(c.y, c[col]))} | {len(c)} | {np.mean(accs):.3f} | {prior:.3f} |")
    cb = cases.dropna(subset=["base", "solve_cut"])
    if len(cb) > 10:
        E["rho_solve_cut_vs_base"] = float(spearmanr(cb.solve_cut, cb.base).correlation)
        E["mean_solve_cut"] = float(cb.solve_cut.mean()); E["mean_base"] = float(cb.base.mean())
        L += ["", f"Solve rate at our cut vs the labellers' base accuracy at 8k on the same {len(cb)} cases: Spearman {E['rho_solve_cut_vs_base']:.2f}, "
              f"means {E['mean_solve_cut']:.2f} vs {E['mean_base']:.2f}."]
    out["label_decomposition"] = E
    (ROOT / f"notes/results/wb_diagnostics_{a.tag}.json").write_text(json.dumps(out, indent=1, default=str))
    (ROOT / f"notes/results/wb_diagnostics_{a.tag}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
