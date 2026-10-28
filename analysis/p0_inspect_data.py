#!/usr/bin/env python3
"""P0: inspect every public aimo-interp dataset; priors, label rule, overlaps."""
import json, re, sys, pathlib
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
HF = pathlib.Path(__file__).resolve().parents[1] / "data" / "hf"
def load(name):
    files = sorted((HF / name / "data").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df

print("="*100); print("TRAIN-MAIN-V2 (official public train set)")
tr = load("train-main-v2")
print(tr.dtypes); print("rows", len(tr), "unique problems", tr["problem"].nunique())
tr["cfg"] = tr["model_id"] + " · " + tr["reasoning_effort"]
tr["label"] = tr["is_robust"].map({True: "robust", False: "nonrobust"}).fillna("null")
print(pd.crosstab(tr["cfg"], tr["label"], margins=True))
print("\nmax_drop summary by label:"); print(tr.groupby("label")["max_drop"].describe())
print("\nmax_drop values (sorted):", sorted(round(x, 3) for x in tr["max_drop"]))
# per-problem label mixing
g = tr[tr["is_robust"].notna()].groupby("problem")["is_robust"].agg(["count", "sum"])
g["mixed"] = (g["sum"] > 0) & (g["sum"] < g["count"])
print("\nproblems with >=2 labelled configs:", int((g['count']>=2).sum()), " of which mixed labels:", int(g[g['count']>=2]['mixed'].sum()))
print("configs per problem distribution:", g["count"].value_counts().sort_index().to_dict())
tr["nchar"] = tr["problem"].str.len()
print("\nproblem length (chars) by label:"); print(tr.groupby("label")["nchar"].describe()[["count","mean","50%","min","max"]])
print("\nFirst 3 problems:"); 
for p in tr["problem"].unique()[:3]: print("  -", repr(p[:300]))
# save a clean csv for later
tr.drop(columns=["cfg","label","nchar"]).to_csv(HF.parent / "train_main_v2.csv", index=False)

print("\n" + "="*100); print("VAL-SAMPLE (public)")
vs = load("val-sample")
print("rows", len(vs), "unique problems", vs["original_problem"].nunique(), "dataset_ids", vs["dataset_id"].unique().tolist())
print(vs.groupby("model_id")["model_is_robust"].agg(["count","sum"]))
print("\nbase_accuracy vs label:"); print(pd.crosstab(vs["base_accuracy"].round(2), vs["model_is_robust"]))
print("n_base_predictions:", vs["n_base_predictions"].value_counts().to_dict(), " n_permuted_predictions:", vs["n_permuted_predictions"].value_counts().to_dict())
print("\nper-row: model, base_acc, perm_acc, decay, n_detr, types, label")
for _, r in vs.sort_values(["original_problem","model_id"]).iterrows():
    print(f"  {r.problem_id} {r.model_id[:34]:34s} base={r.base_accuracy:.2f} perm={r.permuted_accuracy:.2f} decay={r.absolute_accuracy_decay:.2f} ndet={r.n_detrimental_permutations} types={r.permutation_type[:60]} robust={r.model_is_robust}")
# rule check: robust == (n_detrimental == 0)?
print("\nrobust == (n_detrimental_permutations==0):", (vs["model_is_robust"] == (vs["n_detrimental_permutations"]==0)).all())
# per-problem mixing in val-sample
g = vs.drop_duplicates(["model_id","original_problem"]).groupby("original_problem")["model_is_robust"].agg(["count","sum"])
print("val-sample problems:", len(g), "mixed:", int(((g['sum']>0)&(g['sum']<g['count'])).sum()))

print("\n" + "="*100); print("AIMO-INTERP-CHALLENGE-SAMPLE-FULL (per-perturbation, unlabelled)")
sf = load("aimo-interp-challenge-sample-full")
print("rows", len(sf), "problems", sf["original_problem"].nunique(), "models", sf["model_id"].unique().tolist())
print("dataset_ids", sf["dataset_id"].unique().tolist())
print("permutation types:", sf["permutation_type"].value_counts().to_dict())
print("n_base_predictions:", sf["n_base_predictions"].value_counts().to_dict())
print("n_permuted_predictions:", sf["n_permuted_predictions"].value_counts().to_dict())
# reconstruct labels with the train-main-v2 rule: max over types of (base - permuted)
sf["drop"] = sf["base_accuracy"] - sf["permuted_accuracy"]
agg = sf.groupby(["model_id","original_problem","problem_id"]).agg(base=("base_accuracy","first"), max_drop=("drop","max"), ntypes=("permutation_type","nunique"), ndet=("n_detrimental_permutations","max")).reset_index()
agg["label"] = pd.cut(agg["max_drop"], [-9, 0.10+1e-9, 0.25-1e-9, 9], labels=["robust","null","nonrobust"])
print("\nreconstructed labels (rule: max_drop<=0.10 robust, >=0.25 nonrobust):"); print(pd.crosstab(agg["model_id"], agg["label"]))
print("\nbase accuracy distribution of reconstructed cases:"); print(agg["base"].round(1).value_counts().sort_index().to_dict())
print("\nbase<1 vs label:"); print(pd.crosstab(agg["base"]<1, agg["label"]))
print("\nper problem (rows=problem_id): robust/null/nonrobust counts across models")
print(pd.crosstab(agg["problem_id"], agg["label"]))
# which types drive drops >= 0.25
sf["big"] = sf["drop"] >= 0.25
print("\nshare of (model,problem,type) rows with drop>=0.25 by type:"); print(sf.groupby("permutation_type")["big"].mean().round(3).to_dict())
print("\nmean drop by type and model:"); print(sf.pivot_table(index="permutation_type", columns="model_id", values="drop", aggfunc="mean").round(2))
# overlap with val-sample problems and train problems
print("\noverlap sample-full problems with val-sample:", len(set(sf.original_problem) & set(vs.original_problem)), " with train-main-v2:", len(set(sf.original_problem) & set(tr.problem)))
print("overlap val-sample with train-main-v2:", len(set(vs.original_problem) & set(tr.problem)))
# gpt-oss low rows in sample-full = competition config
print("\ngpt-oss-120b:low reconstructed:"); print(agg[agg.model_id.str.contains("gpt-oss-120b")][["model_id","problem_id","base","max_drop","ntypes","label"]].to_string())
agg.to_csv(HF.parent / "sample_full_reconstructed.csv", index=False)

print("\n" + "="*100); print("AUGMENTED-SAMPLE-MATH-AGG (MATH, qwen3-8b:low alias)")
ma = load("augmented-sample-math-agg")
print("rows", len(ma), "unique problems", ma["original_problem"].nunique(), "models", ma["model_id"].unique().tolist(), "dataset_ids", ma["dataset_id"].unique().tolist())
print("label counts:", ma["model_is_robust"].value_counts().to_dict())
print("base_accuracy dist:", ma["base_accuracy"].round(1).value_counts().sort_index().to_dict())
print("robust == (n_detrimental==0):", (ma["model_is_robust"] == (ma["n_detrimental_permutations"]==0)).all())
print("rule check (decay<=0.10 → robust; >=0.25 → nonrobust):"); print(pd.crosstab(pd.cut(ma["absolute_accuracy_decay"],[-9,0.1+1e-9,0.25-1e-9,9],labels=["<=0.10","band",">=0.25"]), ma["model_is_robust"]))
print("n_base_predictions:", ma["n_base_predictions"].value_counts().to_dict())
for name in ["augmented-sample-math","augmented-sample-math-agg-filtered","augmented-sample-math-full-filtered"]:
    d = load(name); 
    print(f"\n{name}: rows {len(d)}, problems {d['original_problem'].nunique()}, models {d['model_id'].unique().tolist()}, types {d['permutation_type'].astype(str).value_counts().to_dict() if 'permutation_type' in d else None}, labels {d['model_is_robust'].value_counts().to_dict() if 'model_is_robust' in d else None}")
