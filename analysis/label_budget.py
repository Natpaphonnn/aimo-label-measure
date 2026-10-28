#!/usr/bin/env python3
"""How many samples per prompt make the robustness label a reliable instrument? Same simulated world as
analysis/label_noise_ceiling.py (empirical base accuracies of aimo-interp-challenge-sample-full, each perturbation family truly
degrades with probability pi_t = 0.5 x observed share, drop ~ U(0.3, 1.0); robust iff max drop <= 0.10, non-robust iff >= 0.25,
band dropped), but the number of samples per prompt K varies. Reports, per K: P(non-robust) for a case with no true degradation
and solve probability 0.9, test-retest agreement of the labels, the accuracy of an oracle that knows the true degradations, and
how far the true solve rate alone moves accuracy above the majority class. Pure numpy; writes notes/results/label_budget.md.
"""
import pathlib
import numpy as np
rng = np.random.default_rng(20261027)
OUT = pathlib.Path(__file__).resolve().parents[1] / "notes" / "results" / "label_budget.md"
OUT.parent.mkdir(parents=True, exist_ok=True)
OBS_SHARE = np.array([0.327, 0.122, 0.122, 0.111, 0.089, 0.078, 0.044])   # as in label_noise_ceiling.py
BASE_VALS = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); BASE_CNT = np.array([1, 2, 3, 1, 5, 3, 5, 7, 33])


def label(md):
    return np.where(md <= 0.10 + 1e-9, 1, np.where(md >= 0.25 - 1e-9, 0, -1))


def sampled(K, p0, drops, reps):
    pt = np.clip(p0[:, None] - drops, 0.0, 1.0); n = len(p0)
    b = rng.binomial(K, np.broadcast_to(p0, (reps, n))) / K
    a = rng.binomial(K, np.broadcast_to(pt, (reps, n, pt.shape[1]))) / K
    return label((b[:, :, None] - a).max(axis=2))


def world(share, n=60000):
    p0 = rng.choice(BASE_VALS, size=n, p=BASE_CNT / BASE_CNT.sum())
    has = rng.random((n, len(share))) < np.clip(0.5 * share, 0, 1)
    return p0, np.where(has, rng.uniform(0.3, 1.0, has.shape), 0.0)


L = ["# Label reliability vs samples per prompt (analysis/label_budget.py)", "",
     "World of label_noise_ceiling.py section B (pi scale 0.5, all p0 > 0 kept). 'false NR' = P(non-robust) for a case with no true",
     "degradation at p0 = 0.9. Ceiling = accuracy on the sampled labels of a predictor that knows every true degradation.",
     "'solve-rate gain' = best-threshold accuracy of the true solve rate alone minus the majority class.", ""]
for name, share in (("all 7 families", OBS_SHARE), ("6 families, no expert_no_solution", OBS_SHARE[1:])):
    L += [f"## {name}", "", "| samples per prompt K | false NR at p0=0.9 | kept share | robust share | test-retest | oracle ceiling | majority | solve-rate gain |",
          "|---|---|---|---|---|---|---|---|"]
    p0, drops = world(share); tmd = drops.max(axis=1); tl = label(tmd); truth = np.where(tl == -1, (tmd < 0.175).astype(int), tl)
    for K in (10, 20, 40, 100):
        fnr = float(np.mean(sampled(K, np.full(20000, 0.9), np.zeros((20000, len(share))), 1)[0] == 0))
        l1, l2 = sampled(K, p0, drops, 2); kept = l1 != -1; both = kept & (l2 != -1)
        maj = max(np.mean(l1[kept] == 1), np.mean(l1[kept] == 0))
        sr = max(np.mean((p0[kept] >= t).astype(int) == l1[kept]) for t in (0.55, 0.65, 0.75, 0.85, 0.95, 0.999))
        L.append(f"| {K} | {fnr:.2f} | {kept.mean():.2f} | {np.mean(l1[kept] == 1):.2f} | {np.mean(l1[both] == l2[both]):.3f} | "
                 f"{np.mean(truth[kept] == l1[kept]):.3f} | {maj:.3f} | {sr - maj:+.3f} |")
    L.append("")
OUT.write_text("\n".join(L) + "\n"); print("\n".join(L))
