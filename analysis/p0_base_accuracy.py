#!/usr/bin/env python3
"""P0: how does base accuracy relate to the reconstructed label on AIMO-level data?"""
import pandas as pd, pathlib, json
pd.set_option("display.width", 250)
HF = pathlib.Path(__file__).resolve().parents[1] / "data" / "hf"
load = lambda n: pd.concat([pd.read_parquet(f) for f in sorted((HF/n/"data").glob("*.parquet"))], ignore_index=True)
sf = load("aimo-interp-challenge-sample-full"); vs = load("val-sample"); tr = load("train-main-v2")
sf["drop"] = sf["base_accuracy"] - sf["permuted_accuracy"]
agg = sf.groupby(["model_id","problem_id"]).agg(base=("base_accuracy","first"), nb=("n_base_predictions","first"), max_drop=("drop","max"), ntypes=("permutation_type","nunique")).reset_index()
agg["label"] = pd.cut(agg["max_drop"], [-9, 0.10+1e-9, 0.25-1e-9, 9], labels=["robust","null","nonrobust"])
print("sample-full reconstructed, EXCLUDING base==0:")
a = agg[agg.base > 0]
print(pd.crosstab(a["base"].round(1), a["label"], margins=True))
print("\nrule 'robust iff base==1.0' accuracy on labelled reconstructed cases (base>0):")
lab = a[a.label != "null"]
print("  n=", len(lab), " acc=", round(((lab.base >= 0.999) == (lab.label == "robust")).mean(), 3), " always-robust acc=", round((lab.label=="robust").mean(),3))
print("\nval-sample: same rule:")
v = vs.drop_duplicates(["model_id","problem_id"])
print("  n=", len(v), " acc=", round(((v.base_accuracy >= 0.999) == v.model_is_robust).mean(),3), " always-true=", round(v.model_is_robust.mean(),3))
print("\nPer-problem balance in sample-full (base>0, labelled): robust/nonrobust per problem")
print(pd.crosstab(lab["problem_id"], lab["label"]))
print("\nHow often is the per-problem ordering by base accuracy consistent with labels? (within problem, robust models have base >= nonrobust models)")
ok=tot=0
for pid, g in lab.groupby("problem_id"):
    r = g[g.label=="robust"].base.tolist(); nr = g[g.label=="nonrobust"].base.tolist()
    for x in r:
        for y in nr:
            tot += 1; ok += (x >= y)
print(f"  pairs={tot} consistent={ok} ({ok/max(tot,1):.2f})")
print("\nWhich types cause the max drop for nonrobust cases (base>0)?")
sfl = sf[sf.base_accuracy>0].copy()
idx = sfl.groupby(["model_id","problem_id"])["drop"].idxmax()
top = sfl.loc[idx]; top = top[top["drop"]>=0.25]
print(top["permutation_type"].value_counts().to_dict())
print("\nn_permuted_predictions per type (AIMO sample):"); print(sf.groupby("permutation_type")["n_permuted_predictions"].describe()[["count","mean","min","max"]])
# train-main-v2 negative drops => base<1 rows that are robust
print("\ntrain-main-v2 rows with max_drop<0 (perturbed > base, so base<1) and their labels:")
print(tr[tr.max_drop<0][["model_id","reasoning_effort","max_drop","is_robust"]].to_string())
# train-main-v2 overlap problem with val-sample
ov = set(tr.problem) & set(vs.original_problem)
print("\noverlapping problem train<->val-sample:", [p[:80] for p in ov])
for p in ov:
    print(tr[tr.problem==p][["model_id","reasoning_effort","is_robust","max_drop"]].to_string())
    print(vs[vs.original_problem==p][["model_id","base_accuracy","absolute_accuracy_decay","model_is_robust"]].to_string())
