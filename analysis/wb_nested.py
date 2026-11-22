#!/usr/bin/env python3
"""Nested two-stage evaluation (revision of #21, R-Q1/M2; pre-registered in notes/PREREG_REVISION_2026-11-22.md, P1).

The October analysis cross-fitted stage 1 over ALL problems of a model before the stage-2 folds were drawn, so the stage-2
training cases of an outer fold carried scores from probes that had seen the outer-test problems' correctness labels. Here,
for every outer stage-2 fold, stage 1 only sees samples of problems outside the outer test fold: outer-training cases are
scored by inner cross-fitting inside that pool, outer-test cases by a probe fitted on the whole pool. Stage 2 is then fitted
on the outer-training cases and scored on the outer-test cases.

Arms: prior (per-model majority of the training cases), offsets (per-model offsets only), bb, wb, hybrid; unweighted stage 2
(plain accuracy) and class-balanced stage 2 (balanced accuracy); per-class recalls and the share predicted robust.
--probe-subset truncated: stage 1 is trained on samples that reached the cut only (P2 variant), and scores every sample.
--from-cache: skip stage 1 and rebuild every stage-2 table from the shipped cache (notes/results/cache/samples.csv.gz and
stage1_nested_scores[_truncated].csv.gz); no residual states needed. Outputs get the suffix _fromcache.

Usage: python analysis/wb_nested.py [--probe-subset all|truncated] [--jobs 6] [--from-cache]
Writes notes/results/wb_nested[_truncated].json (+ per-case predictions notes/results/wb_nested[_truncated]_cases.csv.gz,
and the per-sample nested stage-1 scores notes/results/stage1_nested_scores[_truncated].csv.gz) and a markdown table.
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import RESULTS, need  # noqa: E402
from run_wb import collect  # noqa: E402
from wb_two_stage import KS, SEEDS, _predict_probe, case_table, folds_for, select_and_fit  # noqa: E402

ARMS = {"offsets": [], "bb": ["bb_mean", "bb_sd"], "wb": ["wb_mean", "wb_sd"], "hybrid": ["wb_mean", "wb_sd", "bb_mean", "bb_sd"]}
MODELS = ["qwen", "skywork", "deepseek"]
CUT = {"Qwen/Qwen3.5-4B": 3072, "Skywork/Skywork-OR1-Math-7B": 3072, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 2560}


def stage1_pool(g: pd.DataFrame, pool: np.ndarray, test: np.ndarray, seed: int, train_mask: np.ndarray) -> np.ndarray:
    """Scores for one model and one outer fold. pool/test: boolean masks over g's rows (problems outside / inside the outer
    test fold). train_mask: rows usable as stage-1 training examples (gold-labelled, and truncated-only for the variant)."""
    p = np.full(len(g), np.nan)
    f = folds_for(g.problem_id.values[pool], 5, seed); idx = np.where(pool)[0]
    for k in np.unique(f):   # inner cross-fitting inside the pool
        te = idx[f == k]; tr = idx[f != k]; tr = tr[train_mask[tr]]
        if len(tr) < 20: continue
        b, n_pca, C, m = select_and_fit(g.iloc[tr], seed)
        p[te] = _predict_probe(m, np.stack(g.iloc[te][b].values))
    tr = np.where(pool & train_mask)[0]
    b, n_pca, C, m = select_and_fit(g.iloc[tr], seed)
    te = np.where(test)[0]
    if len(te): p[te] = _predict_probe(m, np.stack(g.iloc[te][b].values))
    return p


G: dict = {}   # model_id -> its samples; each worker loads them once (spawn: forking after BLAS init segfaults on macOS)


def init_worker():
    d = collect(MODELS).reset_index(drop=True)
    d["t"] = d.correct.map(lambda v: np.nan if v is None or v != v else float(v)); d["cut"] = d.model_id.map(CUT)
    G.update({m: g for m, g in d.groupby("model_id")})


def job(args):
    m, seed, test_problems, subset = args; g = G[m]
    train_mask = g.t.notna().values
    if subset == "truncated":
        train_mask = train_mask & (g.n_gen.values >= g.cut.values)
    test = g.problem_id.isin(test_problems).values
    return stage1_pool(g, ~test, test, seed, train_mask)


def fit_predict(Xtr, ytr, Xte, balanced: bool) -> np.ndarray:
    if len(np.unique(ytr)) < 2: return np.full(len(Xte), float(ytr.mean()))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced" if balanced else None).fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def design(cases: pd.DataFrame, cols: list[str], models: list[str]) -> np.ndarray:
    oh = np.stack([(cases.model_id == m).values.astype(float) for m in models], 1)
    return np.hstack([cases[cols].values.astype(float), oh]) if cols else oh


def bacc(y, pred):
    return 0.5 * (float(pred[y].mean()) + float((~pred[~y]).mean()))


def cluster_boot_delta(y, pa, pb, groups, n=10000, seed=0, metric=bacc):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; v = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); yy = y[s]
        if yy.all() or (~yy).all(): continue
        v.append(metric(yy, pa[s]) - metric(yy, pb[s]))
    v = np.array(v)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))


def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--probe-subset", choices=["all", "truncated"], default="all")
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--from-cache", action="store_true"); a = ap.parse_args()
    need(ROOT / "data/labels/case_labels.csv")
    suffix = "" if a.probe_subset == "all" else "_truncated"
    if a.from_cache:
        need(RESULTS / "cache/samples.csv.gz", RESULTS / f"stage1_nested_scores{suffix}.csv.gz")
        from label_decomp_revision import read_samples_cache
        df = read_samples_cache(RESULTS / "cache/samples.csv.gz")
        df["correct"] = df.correct.map(lambda v: None if pd.isna(v) else bool(v in (True, "True", 1, 1.0)))
    else:
        df = collect(MODELS).reset_index(drop=True)
    df["t"] = df.correct.map(lambda v: np.nan if v is None or v != v else float(v)); df["cut"] = df.model_id.map(CUT)
    lab = pd.read_csv(ROOT / "data/labels/case_labels.csv")[["model_id", "problem_id", "y", "source"]].drop_duplicates(["model_id", "problem_id"])
    labelled = df.merge(lab[["model_id", "problem_id"]], on=["model_id", "problem_id"]).problem_id.unique()
    models = sorted(df.model_id.unique())
    # outer folds: on the problems of the labelled cases (all K use the same problems; K = 8 drops cases with < 8 samples)
    scores = {}   # (seed, fold) -> Series of scores over df.index
    tasks, keys = [], []
    for s in SEEDS:
        f = folds_for(np.array(sorted(labelled)), 5, s); fold_of = dict(zip(sorted(labelled), f))
        for k in range(5):
            test_problems = {p for p, ff in fold_of.items() if ff == k}
            for m, g in df.groupby("model_id"):
                tasks.append((m, s, test_problems, a.probe_subset)); keys.append((s, k, m, g.index))
    if a.from_cache:
        key = ["model_id", "problem_id", "sample_idx"]; cs = pd.read_csv(RESULTS / f"stage1_nested_scores{suffix}.csv.gz")
        for (s, k), h in cs.groupby(["seed", "outer_fold"]):
            scores[(s, k)] = pd.Series(df[key].merge(h[key + ["p_correct"]], on=key, how="left").p_correct.values, index=df.index)
        suffix += "_fromcache"
    else:
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn"), initializer=init_worker) as ex:
            outs = list(ex.map(job, tasks))
        for (s, k, m, gi), p in zip(keys, outs):
            scores.setdefault((s, k), pd.Series(np.nan, index=df.index)).loc[gi] = p
    res = {"probe_subset": a.probe_subset, "per_K": {}}; case_rows = []; s1_rows = []
    for s in SEEDS:
        f = folds_for(np.array(sorted(labelled)), 5, s); fold_of = dict(zip(sorted(labelled), f))
        for k in range(5):
            sc = scores[(s, k)]
            tp = {p for p, ff in fold_of.items() if ff == k}
            s1_rows.append(pd.DataFrame({"seed": s, "outer_fold": k, "model_id": df.model_id, "problem_id": df.problem_id,
                                         "sample_idx": df.sample_idx, "role": np.where(df.problem_id.isin(tp), "outer_test", "outer_train"),
                                         "p_correct": sc.values}))
    for K in KS:
        per_seed = {}
        for s in SEEDS:
            f = folds_for(np.array(sorted(labelled)), 5, s); fold_of = dict(zip(sorted(labelled), f))
            preds = {(arm, bal): None for arm in list(ARMS) + ["prior"] for bal in (False, True)}; y_all = None; parts = []
            for k in range(5):
                d = df.assign(pk=scores[(s, k)])
                cases = case_table(d, K, "pk").merge(lab, on=["model_id", "problem_id"]).dropna(subset=["wb_mean"]).reset_index(drop=True)
                cases["fold"] = cases.problem_id.map(fold_of); tr = (cases.fold != k).values; te = ~tr
                y = cases.y.values.astype(int); out = {"model_id": cases.model_id[te].values, "problem_id": cases.problem_id[te].values, "y": y[te]}
                for arm, cols in ARMS.items():
                    for bal in (False, True):
                        X = design(cases, cols, models); out[f"{arm}_{'bal' if bal else 'plain'}"] = fit_predict(X[tr], y[tr], X[te], bal) >= 0.5
                maj = {m: cases.y[tr & (cases.model_id == m).values].mean() >= 0.5 for m in models}
                pr = np.array([maj[m] for m in cases.model_id[te]]); out["prior_plain"] = pr; out["prior_bal"] = pr
                out["wb_mean"] = cases.wb_mean[te].values; out["wb_sd"] = cases.wb_sd[te].values
                out["bb_mean"] = cases.bb_mean[te].values; out["bb_sd"] = cases.bb_sd[te].values
                parts.append(pd.DataFrame(out))
            c = pd.concat(parts, ignore_index=True).sort_values(["model_id", "problem_id"]).reset_index(drop=True)
            per_seed[s] = c; c2 = c.assign(K=K, seed=s); case_rows.append(c2)
        n = len(per_seed[SEEDS[0]]); y = per_seed[SEEDS[0]].y.values.astype(bool)
        r = {"n": int(n), "n_problems": int(per_seed[SEEDS[0]].problem_id.nunique()), "robust_share": float(y.mean()), "arms": {}}
        for arm in list(ARMS) + ["prior"]:
            for v in ("plain", "bal"):
                acc, ba, tpr, tnr, share = [], [], [], [], []
                for s in SEEDS:
                    c = per_seed[s]; yy = c.y.values.astype(bool); p = c[f"{arm}_{v}"].values.astype(bool)
                    acc.append(float((p == yy).mean())); ba.append(bacc(yy, p)); tpr.append(float(p[yy].mean())); tnr.append(float((~p[~yy]).mean())); share.append(float(p.mean()))
                r["arms"][f"{arm}_{v}"] = {"acc": float(np.mean(acc)), "bacc": float(np.mean(ba)), "recall_robust": float(np.mean(tpr)),
                                           "recall_nonrobust": float(np.mean(tnr)), "share_pred_robust": float(np.mean(share)),
                                           "acc_seeds": acc, "bacc_seeds": ba}
        # primary family (plain accuracy, exact McNemar on the last seed as in October) and the K = 4 balanced comparisons
        c = per_seed[SEEDS[-1]]; yy = c.y.values.astype(bool)
        r["mcnemar_last_seed"] = {f"{a1}_vs_{a2}": mcnemar(c[f"{a1}_plain"].values == yy, c[f"{a2}_plain"].values == yy)
                                  for a1, a2 in (("wb", "bb"), ("wb", "prior"), ("hybrid", "bb"), ("bb", "prior"))}
        r["plain_delta_last_seed"] = {f"{a1}_vs_{a2}": float((c[f"{a1}_plain"].values == yy).mean() - (c[f"{a2}_plain"].values == yy).mean())
                                      for a1, a2 in (("wb", "bb"), ("wb", "prior"))}
        if K == 4:
            r["boot_bal"] = {}
            for a1, a2 in (("wb", "prior"), ("wb", "offsets"), ("bb", "prior"), ("wb", "bb"), ("hybrid", "prior"), ("bb", "offsets")):
                rows = []
                for i, s in enumerate(SEEDS):
                    c = per_seed[s]; yy = c.y.values.astype(bool); pa = c[f"{a1}_bal"].values.astype(bool); pb = c[f"{a2}_bal"].values.astype(bool)
                    ci, p = cluster_boot_delta(yy, pa, pb, c.problem_id.values, seed=i)
                    rows.append({"seed": s, "delta": bacc(yy, pa) - bacc(yy, pb), "ci": ci, "p": p})
                r["boot_bal"][f"{a1}_vs_{a2}"] = rows
        res["per_K"][f"K{K}"] = r
    fam = [(K, c, res["per_K"][K]["mcnemar_last_seed"][c]) for K in res["per_K"] for c in ("wb_vs_bb", "wb_vs_prior")]
    order = sorted(range(len(fam)), key=lambda i: fam[i][2]); adj = [0.0] * len(fam); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(fam) - rank) * fam[i][2])); adj[i] = run
    res["holm_primary"] = {f"{K}_{c}": {"p": p, "p_holm": q} for (K, c, p), q in zip(fam, adj)}
    (RESULTS / f"wb_nested{suffix}.json").write_text(json.dumps(res, indent=1))
    pd.concat(case_rows).to_csv(RESULTS / f"wb_nested{suffix}_cases.csv.gz", index=False)
    if not a.from_cache: pd.concat(s1_rows).to_csv(RESULTS / f"stage1_nested_scores{suffix}.csv.gz", index=False)
    L = [f"# Nested two-stage evaluation (analysis/wb_nested.py, probe subset = {a.probe_subset})", "",
         "Stage 1 inside the outer stage-2 folds (pre-registration P1). Means over stage-2 seeds 0-2. rec+ = recall of robust,",
         "rec- = recall of non-robust, pred+ = share predicted robust.", "",
         "| K | n | arm | acc | rec+ | rec- | pred+ | bal (class-weighted) | rec+ | rec- | pred+ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for K, r in res["per_K"].items():
        for arm in ["prior"] + list(ARMS):
            P, B = r["arms"][f"{arm}_plain"], r["arms"][f"{arm}_bal"]
            L.append(f"| {K} | {r['n']} | {arm} | {P['acc']:.3f} | {P['recall_robust']:.2f} | {P['recall_nonrobust']:.2f} | {P['share_pred_robust']:.2f} | "
                     f"{B['bacc']:.3f} | {B['recall_robust']:.2f} | {B['recall_nonrobust']:.2f} | {B['share_pred_robust']:.2f} |")
    L += ["", "Primary family (plain accuracy, exact McNemar, last seed; Holm over 8): " +
          "; ".join(f"{k} p={v['p']:.3f} holm={v['p_holm']:.3f}" for k, v in res["holm_primary"].items()), "",
          "K = 4, balanced accuracy, paired cluster bootstrap over problems (10^4 per seed): delta in points [95% CI] p", ""]
    for comp, rows in res["per_K"]["K4"]["boot_bal"].items():
        L.append(f"- {comp}: " + "; ".join(f"seed {x['seed']} {100*x['delta']:+.1f} [{100*x['ci'][0]:+.1f}, {100*x['ci'][1]:+.1f}] p={x['p']:.3f}" for x in rows)
                 + f"; mean {100*np.mean([x['delta'] for x in rows]):+.1f}")
    (RESULTS / f"wb_nested{suffix}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
