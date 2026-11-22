#!/usr/bin/env python3
"""Stage 1 on naturally finished vs truncated samples, fully separated (revision of #21, R-Q2; pre-registered P2).
finished = the sample ended before the per-model cut (its last-64-token window can contain natural final-answer tokens);
truncated = it reached the cut (the window lies inside the reasoning). Inside each subset the probe is selected, fitted and
evaluated separately (grouped cross-fitting by problem, same grid as the paper); AUC per model and subset with a 95% cluster
bootstrap over problems. Writes notes/results/stage1_finished_truncated.md and .json."""
import json, pathlib, sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import RESULTS  # noqa: E402
from run_wb import collect  # noqa: E402
from wb_two_stage import crossfit_probe  # noqa: E402
from wb_nested import CUT, MODELS  # noqa: E402


def boot_auc(y, p, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in u}; out = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        if len(np.unique(y[s])) == 2: out.append(roc_auc_score(y[s], p[s]))
    return np.percentile(out, [2.5, 97.5])


def main():
    L = ["# Stage 1 on finished vs truncated samples, probes fitted separately (analysis/stage1_finished_truncated.py)", "",
         "| model | subset | gold samples | correct share | problems | AUC [95% CI] |", "|---|---|---|---|---|---|"]; out = {}
    for m in MODELS:
        df = collect([m]).reset_index(drop=True)
        df["t"] = df.correct.map(lambda v: np.nan if v is None or v != v else float(v)); cut = CUT[df.model_id.iloc[0]]
        for name, mask in (("finished", df.n_gen.values < cut), ("truncated", df.n_gen.values >= cut)):
            sub = df[mask].reset_index(drop=True); p, _ = crossfit_probe(sub, 0)
            g = sub.t.notna().values & p.notna().values; y = sub.t.values[g].astype(int); s = p.values[g]
            if len(np.unique(y)) < 2:
                L.append(f"| {m} | {name} | {g.sum()} | {y.mean():.2f} | {sub.problem_id[g].nunique()} | one class only |"); continue
            auc = roc_auc_score(y, s); lo, hi = boot_auc(y, s, sub.problem_id.values[g])
            out[f"{m}_{name}"] = {"n": int(g.sum()), "correct_share": float(y.mean()), "auc": float(auc), "ci": [float(lo), float(hi)]}
            L.append(f"| {m} | {name} | {g.sum()} | {y.mean():.2f} | {sub.problem_id[g].nunique()} | {auc:.3f} [{lo:.3f}, {hi:.3f}] |")
    (RESULTS / "stage1_finished_truncated.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "stage1_finished_truncated.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
