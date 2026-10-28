#!/usr/bin/env python3
"""Does a mean-pooled hidden-state probe add signal beyond problem length? DeepSeek, official MATH labels (137).
Nested: the layer is selected on inner folds only. Compares (a) prior, (b) length-only logistic, (c) probe, (d) probe + length,
with paired bootstrap / McNemar of (d) vs (b)."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from eval_features import paired_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from datapaths import ROOT, need  # noqa: E402
need(ROOT / "runs/kaggle/kx-feat-deepseek-prefill/out/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B__math_deepseek_problems_hidden.npz", ROOT / "runs/eval/math_deepseek_prefill.csv")
z = np.load("runs/kaggle/kx-feat-deepseek-prefill/out/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B__math_deepseek_problems_hidden.npz")
tab = pd.read_csv("runs/eval/math_deepseek_prefill.csv"); idx = {p: i for i, p in enumerate(z["problem_id"])}
tab = tab[tab.problem_id.isin(idx)].reset_index(drop=True); rows = [idx[p] for p in tab.problem_id]
H = z["mean_problem"][rows].astype(np.float32); y = tab.y.values.astype(int)
L = np.log(tab[["f_pre_chars", "f_pre_n_problem_tokens"]].values) ; L = np.c_[L, tab.f_pre_zlib_ratio.values]
LAYERS = list(range(2, 37, 2))
def fit_pred(Xtr, ytr, Xte, pca=True, C=0.1):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    if pca: p = PCA(n_components=min(32, len(Xtr) // 3)).fit(A); A, B = p.transform(A), p.transform(B)
    return LogisticRegression(C=C, max_iter=3000).fit(A, ytr).predict_proba(B)[:, 1]
res = {k: [] for k in ["length", "probe", "probe_len"]}; probs = {k: np.zeros((3, len(y))) for k in res}; chosen = []
for s in range(3):
    outer = StratifiedKFold(5, shuffle=True, random_state=s)
    for tr, te in outer.split(H, y):
        # inner layer selection
        inner = StratifiedKFold(3, shuffle=True, random_state=100 + s); best, bl = -1, LAYERS[0]
        for l in LAYERS:
            acc = np.mean([((fit_pred(H[tr][a, l], y[tr][a], H[tr][b, l]) >= 0.5) == y[tr][b]).mean() for a, b in inner.split(H[tr], y[tr])])
            if acc > best: best, bl = acc, l
        chosen.append(bl)
        probs["length"][s, te] = fit_pred(L[tr], y[tr], L[te], pca=False, C=1.0)
        probs["probe"][s, te] = fit_pred(H[tr][:, bl], y[tr], H[te][:, bl])
        X2tr = np.c_[PCA(n_components=min(32, len(tr)//3)).fit(StandardScaler().fit_transform(H[tr][:, bl])).transform(StandardScaler().fit(H[tr][:, bl]).transform(H[tr][:, bl])), L[tr]]
        sc = StandardScaler().fit(H[tr][:, bl]); pc = PCA(n_components=min(32, len(tr)//3)).fit(sc.transform(H[tr][:, bl]))
        A = np.c_[pc.transform(sc.transform(H[tr][:, bl])), L[tr]]; B = np.c_[pc.transform(sc.transform(H[te][:, bl])), L[te]]
        s2 = StandardScaler().fit(A); probs["probe_len"][s, te] = LogisticRegression(C=0.1, max_iter=3000).fit(s2.transform(A), y[tr]).predict_proba(s2.transform(B))[:, 1]
prior = np.zeros(len(y), bool)  # always non-robust (majority)
out = {"n": int(len(y)), "robust": int(y.sum()), "prior_acc": float((prior == y).mean()), "layers_chosen": chosen}
for k, P in probs.items():
    accs = [((P[s] >= 0.5) == y).mean() for s in range(3)]; aucs = [roc_auc_score(y, P[s]) for s in range(3)]
    out[k] = {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "auc": float(np.mean(aucs))}
pm = lambda k: (probs[k].mean(0) >= 0.5) == y
out["probe_len_vs_length"] = paired_stats(pm("probe_len"), pm("length")); out["probe_vs_prior"] = paired_stats(pm("probe"), prior == y)
print(json.dumps(out, indent=1)); json.dump(out, open("notes/results/probe_vs_length_math_deepseek.json", "w"), indent=1)
