#!/usr/bin/env python3
"""What the robustness label is a function of, with nested stage-1 scores (revision of #21; pre-registered P2 and P3).

Inputs: the per-sample nested stage-1 scores written by analysis/wb_nested.py (notes/results/stage1_nested_scores.csv.gz),
the per-sample table notes/results/cache/samples.csv.gz (answers, gold correctness, tokens generated; written here from the
logged generations when they are present, otherwise read from the cache), data/labels/case_labels.csv, data/seed/byeolki_base.csv.

P2  same-model solve-rate-matched comparison: pooled within-stratum AUC of the wb case mean for the label, strata = (model,
    correct samples out of 8 at the cut); nested class-balanced stage 2: [solve rate] + offsets vs [solve rate, wb mean, wb sd] + offsets.
P3  monotone AUC, band AUC |x - 0.5|, nested two-threshold band rule and [x, x^2] logistic, for the gold solve rate at the cut,
    the wb mean, the agreement (twin) mean and the labellers' base accuracy at 8k; by label source (byeolki vs train-main-v2).
Usage: python analysis/label_decomp_revision.py      Writes notes/results/label_decomp_revision.md and .json.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import RESULTS, need  # noqa: E402
from wb_two_stage import SEEDS, folds_for, twin_scores  # noqa: E402

CACHE = RESULTS / "cache"; K = 8; B = 10000


def read_samples_cache(f) -> pd.DataFrame:
    """Exact round trip of the answers: a missing answer is a float NaN (twin_scores gives it -1/(K-1); None would give 0),
    while some extracted answers are the literal string "NaN", which pandas would otherwise read as missing. answer_kind
    (value / nan / none) and literal reading keep cached runs identical to the runs on the logged records."""
    d = pd.read_csv(f).drop(columns=["answer", "answer_kind"])   # literal strings: pandas would read "none"/"nan"/"" as missing
    d = d.join(pd.read_csv(f, usecols=["answer", "answer_kind"], dtype=str, keep_default_na=False))
    d["answer"] = pd.Series([None if k == "none" else np.nan if k == "nan" else v for v, k in zip(d.answer, d.answer_kind)], index=d.index, dtype=object)
    return d.drop(columns="answer_kind")


def samples() -> pd.DataFrame:
    f = CACHE / "samples.csv.gz"
    if f.exists(): return read_samples_cache(f)
    from run_wb import collect
    d = collect(["qwen", "skywork", "deepseek"]).reset_index(drop=True)
    cut = {"Qwen/Qwen3.5-4B": 3072, "Skywork/Skywork-OR1-Math-7B": 3072, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 2560}
    out = d[["model_id", "problem_id", "sample_idx", "answer", "correct", "n_gen"]].copy(); out["cut"] = out.model_id.map(cut)
    out["finished_before_cut"] = out.n_gen < out.cut
    out["answer_kind"] = ["none" if v is None else "nan" if isinstance(v, float) and v != v else "value" for v in out.answer]
    CACHE.mkdir(parents=True, exist_ok=True); out.to_csv(f, index=False); return read_samples_cache(f)


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 5 and 0 < y[ok].mean() < 1 else np.nan


def strat_auc(c: pd.DataFrame, col: str) -> float:
    num = den = 0.0
    for _, h in c.groupby(["model_id", "n_correct"]):
        y = h.y.values.astype(int)
        if 0 < y.mean() < 1:   # every (robust, non-robust) pair inside the stratum counts, however small the stratum
            w = y.sum() * (1 - y).sum(); num += roc_auc_score(y, h[col].values) * w; den += w
    return num / den if den else np.nan


def boot(c: pd.DataFrame, fn, n=B, seed=0):
    rng = np.random.default_rng(seed); ug = c.problem_id.unique(); idx = {g: np.where(c.problem_id.values == g)[0] for g in ug}; v = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); x = fn(c.iloc[s])
        if not np.isnan(x): v.append(x)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bacc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool); return 0.5 * (p[y].mean() + (~p[~y]).mean())


def case_feats(s: pd.DataFrame, score: pd.Series, lab: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    d = s.assign(p=score.values, twin=twin_scores(s, K).values); rows = []
    for (m, pid), h in d.groupby(["model_id", "problem_id"]):
        h = h.sort_values("sample_idx").head(K)
        if len(h) < K: continue
        gold = h.correct.notna().all()
        rows.append({"model_id": m, "problem_id": pid, "wb_mean": h.p.mean(), "wb_sd": h.p.std(ddof=0), "twin_mean": h.twin.mean(),
                     "solve": h.correct.astype(float).mean() if gold else np.nan, "n_correct": int(h.correct.astype(float).sum()) if gold else -1})
    return pd.DataFrame(rows).merge(lab, on=["model_id", "problem_id"]).merge(base, on=["model_id", "problem_id"], how="left")


def fit(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1] >= 0.5


def band_rule(xtr, ytr, xte):
    grid = np.round(np.arange(0, 1.0001, 0.05), 2); best = (-1, 0, 0)
    for a in grid:
        for b in grid[grid > a]:
            v = bacc(ytr, ~((xtr > a) & (xtr < b)))
            if v > best[0]: best = (v, a, b)
    return ~((xte > best[1]) & (xte < best[2]))


def main():
    need(RESULTS / "stage1_nested_scores.csv.gz", ROOT / "data/labels/case_labels.csv", ROOT / "data/seed/byeolki_base.csv")
    s = samples().sort_values(["model_id", "problem_id", "sample_idx"]).reset_index(drop=True)
    s["correct"] = s.correct.map(lambda v: np.nan if pd.isna(v) else float(v in (True, "True", 1, 1.0)))
    # missing answers stay float NaN, as in the logged records and analysis/wb_nested.py (twin = -1/(K-1) for such a sample)
    sc = pd.read_csv(RESULTS / "stage1_nested_scores.csv.gz")
    lab = pd.read_csv(ROOT / "data/labels/case_labels.csv")[["model_id", "problem_id", "y", "source"]].drop_duplicates(["model_id", "problem_id"])
    lab["y"] = lab.y.astype(bool); lab["official"] = lab.source.str.startswith("train-main")
    base = pd.read_csv(ROOT / "data/seed/byeolki_base.csv")[["model_id", "problem_id", "base"]]
    key = ["model_id", "problem_id", "sample_idx"]
    # held-out score per sample (the outer fold in which its problem is outer-test), stage-2 seed 0
    ho = sc[(sc.seed == 0) & (sc.role == "outer_test")][key + ["p_correct"]]
    c0 = case_feats(s, s[key].merge(ho, on=key, how="left").p_correct, lab, base)
    out = {"n_cases": int(len(c0)), "n_gold_cases": int(c0.solve.notna().sum())}
    L = ["# Label decomposition with nested stage-1 scores (analysis/label_decomp_revision.py; pre-registration P2/P3)", "",
         f"{len(c0)} labelled cases with 8 samples ({c0.official.sum()} official train-main-v2, {(~c0.official).sum()} byeolki); "
         f"{c0.solve.notna().sum()} with 8 gold-graded samples. Robust = positive class. 95% cluster bootstrap over problems ({B}).", ""]
    # ---- P3 (a, b): monotone and band AUCs, by source
    preds = [("gold solve rate at the cut", "solve"), ("wb mean (nested probe)", "wb_mean"), ("agreement mean", "twin_mean"), ("labellers' base accuracy at 8k", "base")]
    L += ["## Monotone vs band (two-sided) AUC", "", "| subset | predictor | n | AUC x [CI] | AUC abs(x-0.5) [CI] |", "|---|---|---|---|---|"]
    out["auc"] = {}
    for sub, m in (("all", np.ones(len(c0), bool)), ("byeolki", ~c0.official.values), ("official train-main-v2", c0.official.values)):
        for name, col in preds:
            c = c0[m].dropna(subset=[col]).reset_index(drop=True)
            if len(c) < 15 or c.y.nunique() < 2: continue
            c = c.assign(band=(c[col] - 0.5).abs())
            a1, a2 = auc(c.y, c[col]), auc(c.y, c.band)
            ci1, ci2 = boot(c, lambda h: auc(h.y, h[col])), boot(c, lambda h: auc(h.y, h.band))
            out["auc"][f"{sub}|{col}"] = {"n": int(len(c)), "auc": a1, "ci": ci1, "band_auc": a2, "band_ci": ci2}
            L.append(f"| {sub} | {name} | {len(c)} | {a1:.3f} [{ci1[0]:.2f}, {ci1[1]:.2f}] | {a2:.3f} [{ci2[0]:.2f}, {ci2[1]:.2f}] |")
    # ---- nested rules (P3 c, d) and P2 secondary, per (seed, outer fold) features
    labelled = np.array(sorted(s.merge(lab[["model_id", "problem_id"]], on=["model_id", "problem_id"]).problem_id.unique()))   # as wb_nested.py
    res = {k: [] for k in ("band_solve", "quad_solve", "band_wb", "quad_wb", "band_twin", "quad_twin", "solve", "solve_wb", "offsets")}
    cases_seed = {}
    for sd in SEEDS:
        fold_of = dict(zip(labelled, folds_for(labelled, 5, sd)))
        parts = []
        for k in range(5):
            sk = sc[(sc.seed == sd) & (sc.outer_fold == k)][key + ["p_correct"]]
            c = case_feats(s, s[key].merge(sk, on=key, how="left").p_correct, lab, base).dropna(subset=["solve"]).reset_index(drop=True)
            c["fold"] = c.problem_id.map(fold_of); tr = (c.fold != k).values; te = ~tr
            y = c.y.values; models = sorted(c.model_id.unique()); oh = np.stack([(c.model_id == mm).values.astype(float) for mm in models], 1)
            o = c[te][["model_id", "problem_id", "y"]].copy()
            for col, nm in (("solve", "solve"), ("wb_mean", "wb"), ("twin_mean", "twin")):
                x = c[col].values
                o[f"band_{nm}"] = band_rule(x[tr], y[tr], x[te])
                Xq = np.c_[x, x ** 2, oh]; o[f"quad_{nm}"] = fit(Xq[tr], y[tr], Xq[te])
            Xs = np.c_[c.solve.values, oh]; o["solve"] = fit(Xs[tr], y[tr], Xs[te])
            Xw = np.c_[c.solve.values, c.wb_mean.values, c.wb_sd.values, oh]; o["solve_wb"] = fit(Xw[tr], y[tr], Xw[te])
            o["offsets"] = fit(oh[tr], y[tr], oh[te]); parts.append(o)
        cs = pd.concat(parts).sort_values(["model_id", "problem_id"]).reset_index(drop=True); cases_seed[sd] = cs
        for k2 in res: res[k2].append(bacc(cs.y, cs[k2]))
    cs = cases_seed[SEEDS[0]]
    L += ["", f"## Nested rules on the {len(cs)} cases with 8 gold-graded samples (balanced accuracy, class-balanced fits, grouped 5-fold x 3 seeds)", "",
          "| rule | balanced accuracy (seeds) | mean | CI (seed 0) |", "|---|---|---|---|"]
    out["nested"] = {}
    for k2, v in res.items():
        ci = boot(cs, lambda h: bacc(h.y, h[k2]) if h.y.nunique() == 2 else np.nan, n=2000)
        out["nested"][k2] = {"seeds": v, "mean": float(np.mean(v)), "ci_seed0": ci}
        L.append(f"| {k2} | {' / '.join(f'{x:.3f}' for x in v)} | {np.mean(v):.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
    d = [bacc(cases_seed[sd].y, cases_seed[sd].solve_wb) - bacc(cases_seed[sd].y, cases_seed[sd].solve) for sd in SEEDS]
    dci = boot(cs, lambda h: bacc(h.y, h.solve_wb) - bacc(h.y, h.solve) if h.y.nunique() == 2 else np.nan, n=2000)
    out["solve_wb_minus_solve"] = {"seeds": d, "mean": float(np.mean(d)), "ci_seed0": dci}
    L += ["", f"Adding the probe to the gold solve rate (P2 secondary): {' / '.join(f'{100*x:+.1f}' for x in d)} points, mean {100*np.mean(d):+.1f}, "
          f"seed-0 CI [{100*dci[0]:+.1f}, {100*dci[1]:+.1f}]."]
    # ---- P2 primary: within-stratum AUC (held-out scores)
    cg = c0.dropna(subset=["solve"]).reset_index(drop=True)
    for col in ("wb_mean", "twin_mean"):
        a = strat_auc(cg, col); ci = boot(cg, lambda h: strat_auc(h, col))
        out[f"within_stratum_auc_{col}"] = {"auc": a, "ci": ci}
        L.append(f"Within-stratum AUC ({col}; same model, same number of correct samples out of 8; {len(cg)} cases): {a:.3f} [{ci[0]:.2f}, {ci[1]:.2f}].")
    (RESULTS / "label_decomp_revision.json").write_text(json.dumps(out, indent=1, default=float))
    (RESULTS / "label_decomp_revision.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
