#!/usr/bin/env python3
"""Within-problem ranking test on byeolki-derived self-labels (forced mode, 8k cap, 3 models).
Question: on problems whose labels differ across models, does ranking configs by their base solve rate on the
ORIGINAL problem recover which model is robust? Compare with per-model thresholds and the model-identity prior."""
import pandas as pd, numpy as np, itertools
df = pd.read_csv("data/seed/byeolki_labels_forced.csv")
lab = df.dropna(subset=["label_band"]).copy(); lab["y"] = lab.label_band.astype(bool)
print("labelled cases:", len(lab), "robust:", int(lab.y.sum()))
piv = lab.pivot_table(index="problem_id", columns="model_id", values="y", aggfunc="first")
multi = piv.dropna(thresh=2); mixed = multi[(multi.sum(axis=1) > 0) & (multi.sum(axis=1) < multi.notna().sum(axis=1))]
print("problems with >=2 labelled models:", len(multi), "| mixed labels:", len(mixed))
m = lab[lab.problem_id.isin(mixed.index)]
print("rows on mixed problems:", len(m), "robust share:", round(m.y.mean(), 3))
# model identity prior (leave-one-problem-out): predict robust iff model's robust share on other problems >= 0.5
acc_prior = []
for pid, g in m.groupby("problem_id"):
    others = m[m.problem_id != pid]
    for r in g.itertuples():
        p = others[others.model_id == r.model_id].y.mean(); acc_prior.append((p >= 0.5) == r.y)
print("model-identity prior (LOO):", round(np.mean(acc_prior), 3))
# within-problem rank by base: top-k by base where k = number of robust labels on that problem (oracle k) and k = ceil(n/2)
acc_rank_oracle, acc_rank_half, acc_thr = [], [], []
for pid, g in m.groupby("problem_id"):
    g = g.sort_values("base", ascending=False); n = len(g); k = int(g.y.sum())
    pred = [i < k for i in range(n)]; acc_rank_oracle += list(np.array(pred) == g.y.values)
    kh = int(np.ceil(n / 2)); predh = [i < kh for i in range(n)]; acc_rank_half += list(np.array(predh) == g.y.values)
print("within-problem rank by base, oracle k:", round(np.mean(acc_rank_oracle), 3), "| k=ceil(n/2):", round(np.mean(acc_rank_half), 3))
# absolute threshold on base (within-model, LOO over problems for threshold choice)
for thr in [0.5, 0.75, 0.875, 1.0]:
    print(f"  robust iff base >= {thr}: acc on mixed rows {round(((m.base >= thr) == m.y).mean(), 3)} | on all labelled rows {round(((lab.base >= thr) == lab.y).mean(), 3)}")
# pairwise: within a mixed problem, does the robust model have higher base than the non-robust one?
pairs = []
for pid, g in m.groupby("problem_id"):
    for a, b in itertools.product(g[g.y].itertuples(), g[~g.y].itertuples()):
        pairs.append(np.sign(a.base - b.base))
pairs = np.array(pairs); print("robust-vs-nonrobust pairs within problem:", len(pairs), "| robust has higher base:", round((pairs > 0).mean(), 3), "| tie:", round((pairs == 0).mean(), 3))
# what drives nonrobust among base==1 cases?
b1 = lab[lab.base == 1.0]; print("\nbase==1 cases:", len(b1), "nonrobust among them:", int((~b1.y).sum()), "| worst family counts:", b1[~b1.y].worst_family.value_counts().to_dict())
fr = lab[(lab.base > 0) & (lab.base < 1)]; print("frontier (0<base<1) cases:", len(fr), "nonrobust:", int((~fr.y).sum()), "robust:", int(fr.y.sum()))
