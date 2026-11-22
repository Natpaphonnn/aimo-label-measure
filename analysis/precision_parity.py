#!/usr/bin/env python3
"""Precision parity of the residual states (revision of #21, R-S6; pre-registered P5). Kaggle T4 GPUs compute in fp16, the
evaluation GPUs (compute capability >= 8, src/features/budgeted.py) and the Mac in bf16. Qwen3.5-4B has 24 test-time samples
whose states were computed on both (Mac bf16 / Kaggle fp16): per block cosine and the change of a probe's score; and the
cross-fitted probe AUC on the bf16 Qwen samples vs the fp16 ones. Writes notes/results/precision_parity.md."""
import json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import RESULTS, need  # noqa: E402
from run_wb import collect  # noqa: E402
from wb_two_stage import _fit_probe, _predict_probe, crossfit_probe  # noqa: E402

MAC = ROOT / "runs/features/Qwen_Qwen3.5-4B/tt_wave_states"; KAG = ROOT / "runs/kaggle/kx-states-tt-qwen-rest/output/qwen_tt_rest_states"
need(MAC, KAG)
za, zb = np.load(str(MAC) + ".npz"), np.load(str(KAG) + ".npz")
ma = [json.loads(l) for l in open(str(MAC) + ".jsonl")]; mb = [json.loads(l) for l in open(str(KAG) + ".jsonl")]
key = lambda m: (m["problem_id"], int(m["sample_idx"]), m.get("family", "original"))
ia = {key(m): i for i, m in enumerate(ma)}; pairs = [(ia[key(m)], j) for j, m in enumerate(mb) if key(m) in ia]
L = ["# Precision parity: bf16 (Mac, as the evaluation GPUs) vs fp16 (Kaggle T4) states (analysis/precision_parity.py)", "",
     f"{len(pairs)} Qwen3.5-4B samples recomputed on both. Cosine between the two copies per block (cut 3072):", "",
     "| block | min cosine | median | max abs relative L2 difference |", "|---|---|---|---|"]
for kind in ("trace", "answer"):
    for j in range(za[kind].shape[2]):
        A = np.stack([za[kind][i, 0, j].astype(np.float64) for i, _ in pairs]); B = np.stack([zb[kind][k, 0, j].astype(np.float64) for _, k in pairs])
        cos = (A * B).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(B, axis=1); rel = np.linalg.norm(A - B, axis=1) / np.linalg.norm(A, axis=1)
        L.append(f"| S:{kind}:{j} | {cos.min():.6f} | {np.median(cos):.6f} | {rel.max():.4f} |")
# probe trained on all other Qwen gold samples (fixed PCA 32, C 0.1, best block S:trace:1-type choice by the paper: trace states)
df = collect(["qwen"]).reset_index(drop=True); df["t"] = df.correct.map(lambda v: np.nan if v is None or v != v else float(v))
over = {(ma[i]["problem_id"], int(ma[i]["sample_idx"])) for i, _ in pairs}
is_over = np.array([(p, s) in over for p, s in zip(df.problem_id, df.sample_idx)])
tr = df.t.notna().values & ~df.problem_id.isin({p for p, _ in over}).values
L += ["", "Probe score of the same sample from its bf16 and its fp16 states (probe: PCA 32, C 0.1, trained on the Qwen gold samples of other problems):", ""]
for b in ("S:trace:0", "S:trace:1", "S:trace:2"):
    j = int(b[-1]); m = _fit_probe(np.stack(df.loc[tr, b].values), df.t[tr].values.astype(int), 32, 0.1)
    pa = _predict_probe(m, np.stack([za["trace"][i, 0, j].astype(np.float32) for i, _ in pairs]))
    pb = _predict_probe(m, np.stack([zb["trace"][k, 0, j].astype(np.float32) for _, k in pairs]))
    L.append(f"- {b}: max |p_bf16 - p_fp16| = {np.abs(pa - pb).max():.4f}, mean {np.abs(pa - pb).mean():.4f}")
# AUC by precision of the stored states: samples whose states come from the Kaggle file are fp16, all other Qwen samples were computed on the Mac (bf16)
p, _ = crossfit_probe(df, 0); df["p"] = p.values
tt_mac = {(m["problem_id"], int(m["sample_idx"])) for m in ma}
kag = {(m["problem_id"], int(m["sample_idx"])) for m in mb} - tt_mac
src = np.where([(pp, s) in kag for pp, s in zip(df.problem_id, df.sample_idx)], "fp16", "bf16")
g = df[df.t.notna()]
L += ["", "Cross-fitted probe AUC (inner model selection, as in the paper) on Qwen samples by the precision of their stored states:", ""]
for s in ("bf16", "fp16"):
    h = g[src[g.index] == s]
    L.append(f"- {s}: {len(h)} gold samples, AUC {roc_auc_score(h.t.astype(int), h.p):.3f}" if h.t.nunique() == 2 else f"- {s}: {len(h)} samples")
(RESULTS / "precision_parity.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
