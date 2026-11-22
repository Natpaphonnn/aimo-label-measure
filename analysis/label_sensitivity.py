#!/usr/bin/env python3
"""Sensitivity of the simulated label-reliability numbers to the simulation's assumptions (revision of #21, M3/R-S5/R-Q3;
pre-registered in notes/PREREG_REVISION_2026-11-22.md, P4). Same world as analysis/label_noise_ceiling.py section B and
analysis/label_budget.py:

  A1 each prompt's accuracy is a mean of K independent Bernoulli samples (K = 10 in the challenge);
  A2 perturbation families degrade independently given the case;
  A3 family t truly degrades with probability scale x (share of sample-full rows with an observed drop >= 0.25), scale 0.5;
  A4 a true drop is U(0.3, 1.0);
  A5 base accuracies follow the empirical sample-full distribution (base > 0);
  A6 sample-full = 10 problems x 8 settings of OTHER models than the challenge's five.

Varied here: scale {0.25, 0.5, 0.75, 1.0}; per-family shares bootstrapped over the 10 sample-full problems (200 replicates);
drop size U(0.1, 1) / U(0.3, 1) / U(0.5, 1); families correlated through a Gaussian copula (rho 0.5); base accuracies
bootstrapped over problems. Also: selection effects of dropping the 0.10-0.25 band, and P(robust) of low-accuracy cases.
Pure numpy/pandas; writes notes/results/label_sensitivity.md and .json.
"""
import glob, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import RESULTS, need  # noqa: E402
rng = np.random.default_rng(20261122)
FAMS = ["expert_no_solution", "rephrase", "expert_perturbations", "domain", "typos", "rename", "distract"]


def label(md):
    return np.where(md <= 0.10 + 1e-9, 1, np.where(md >= 0.25 - 1e-9, 0, -1))


def load():
    f = sorted(glob.glob(str(ROOT / "data/hf/aimo-interp-challenge-sample-full/data/*.parquet")))
    if not f: need(ROOT / "data/hf/aimo-interp-challenge-sample-full/data")
    d = pd.read_parquet(f[0])
    d = d[d.permutation_type.isin(FAMS)].copy(); d["deg"] = d.absolute_accuracy_decay >= 0.25 - 1e-9
    base = d.drop_duplicates(["model_id", "problem_id"])[["model_id", "problem_id", "base_accuracy"]]
    return d, base


def shares(d, problems):
    sub = pd.concat([d[d.problem_id == p] for p in problems])
    return np.array([sub[sub.permutation_type == t].deg.mean() for t in FAMS])


def world(n, share, base_vals, scale=0.5, lo=0.3, rho=0.0):
    p0 = rng.choice(base_vals, size=n)
    pi = np.clip(scale * share, 0, 1)
    if rho > 0:   # Gaussian copula: one shared factor, marginals unchanged
        z = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, len(pi)))
        has = norm.cdf(z) < pi
    else:
        has = rng.random((n, len(pi))) < pi
    return p0, np.where(has, rng.uniform(lo, 1.0, has.shape), 0.0)


def sampled(K, p0, drops, reps=2):
    pt = np.clip(p0[:, None] - drops, 0, 1); n = len(p0)
    b = rng.binomial(K, np.broadcast_to(p0, (reps, n))) / K
    a = rng.binomial(K, np.broadcast_to(pt, (reps, n, pt.shape[1]))) / K
    return label((b[:, :, None] - a).max(axis=2))


def metrics(p0, drops, K=10):
    tmd = drops.max(axis=1); tl = label(tmd); truth = np.where(tl == -1, (tmd < 0.175).astype(int), tl)
    l1, l2 = sampled(K, p0, drops); kept = l1 != -1; both = kept & (l2 != -1)
    maj = max(np.mean(l1[kept] == 1), np.mean(l1[kept] == 0))
    sr = max(np.mean((p0[kept] >= t).astype(int) == l1[kept]) for t in (0.55, 0.65, 0.75, 0.85, 0.95, 0.999))
    deg = tmd > 0
    return {"oracle": float(np.mean(truth[kept] == l1[kept])), "test_retest": float(np.mean(l1[both] == l2[both])),
            "kept": float(kept.mean()), "robust_share_kept": float(np.mean(l1[kept] == 1)), "solve_rate_gain": float(sr - maj),
            "truly_degraded_all": float(deg.mean()), "truly_degraded_kept": float(deg[kept].mean()),
            "truly_degraded_dropped": float(deg[~kept].mean()) if (~kept).any() else None,
            "mean_p0_kept": float(p0[kept].mean()), "mean_p0_dropped": float(p0[~kept].mean()) if (~kept).any() else None,
            "oracle_on_truth_robust_share_all": float(np.mean(truth == 1))}


def main():
    d, base = load(); probs = sorted(d.problem_id.unique()); bvals = base.base_accuracy.values; bvals = bvals[bvals > 0]
    share0 = shares(d, probs); n = 40000; out = {"assumptions": __doc__.split("\n\n")[1], "share_observed": dict(zip(FAMS, share0.round(3).tolist()))}
    L = ["# Sensitivity of the simulated label reliability (analysis/label_sensitivity.py)", "",
         "Base world = label_noise_ceiling.py section B (K = 10, scale 0.5, drop U(0.3, 1), independent families, empirical base).",
         "oracle = agreement of a predictor that knows every true degradation with the sampled labels (kept cases);",
         "test-retest = agreement of two independent labellings; gain = best-threshold true solve rate minus majority.", ""]
    rows = []
    def add(name, **kw):
        m = metrics(*world(n, kw.pop("share", share0), kw.pop("bvals", bvals), **kw)); m["setting"] = name; rows.append(m); return m
    add("base (scale 0.5, U(0.3,1), independent)")
    for sc in (0.25, 0.75, 1.0): add(f"scale {sc}", scale=sc)
    for lo in (0.1, 0.5): add(f"drop U({lo},1)", lo=lo)
    add("correlated families (copula rho 0.5)", rho=0.5)
    add("6 families (no expert_no_solution)", share=np.r_[0.0, share0[1:]])
    L += ["| setting | oracle | test-retest | kept | robust share (kept) | solve-rate gain |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['setting']} | {r['oracle']:.3f} | {r['test_retest']:.3f} | {r['kept']:.2f} | {r['robust_share_kept']:.2f} | {r['solve_rate_gain']:+.3f} |")
    # bootstrap over the 10 sample-full problems: family shares and base accuracies together
    boot = []
    for _ in range(200):
        bp = rng.choice(probs, len(probs))
        bb = np.concatenate([base[base.problem_id == p].base_accuracy.values for p in bp]); bb = bb[bb > 0]
        boot.append(metrics(*world(20000, shares(d, bp), bb)))
    q = lambda k: (float(np.percentile([b[k] for b in boot], 5)), float(np.percentile([b[k] for b in boot], 95)))
    out["bootstrap_5_95"] = {k: q(k) for k in ("oracle", "test_retest", "kept", "robust_share_kept", "solve_rate_gain")}
    L += ["", "Bootstrap over the 10 sample-full problems (family shares and base accuracies resampled together, 200 replicates), 5-95% range: " +
          "; ".join(f"{k} {v[0]:.3f}-{v[1]:.3f}" for k, v in out["bootstrap_5_95"].items())]
    # false non-robust at p0 = 0.9 with no degradation does not depend on A2-A5 (only on K and the rule)
    fnr = {K: float(np.mean(sampled(K, np.full(40000, 0.9), np.zeros((40000, 7)), 1)[0] == 0)) for K in (10, 20, 40)}
    out["false_nr_p09"] = fnr
    L += ["", "False non-robust at p0 = 0.9 with no true degradation (depends only on A1, K and the rule): " +
          ", ".join(f"K={k}: {v:.3f}" for k, v in fnr.items())]
    # selection effects of the dropped band and construct validity
    r0 = rows[0]
    L += ["", "## Selection by the 0.10-0.25 band (base world)", "",
          f"kept share {r0['kept']:.2f}; truly degraded: all {r0['truly_degraded_all']:.2f}, kept {r0['truly_degraded_kept']:.2f}, dropped {r0['truly_degraded_dropped']:.2f}; "
          f"mean p0: kept {r0['mean_p0_kept']:.2f}, dropped {r0['mean_p0_dropped']:.2f}."]
    low = {}
    for p in (0.0, 0.1, 0.3, 0.5, 0.9):   # P(robust) when the family accuracies cannot fall further (true drop 0 everywhere)
        lab = sampled(10, np.full(40000, p), np.zeros((40000, 7)), 1)[0]; low[p] = {"robust": float(np.mean(lab == 1)), "dropped": float(np.mean(lab == -1))}
    out["no_degradation_label_by_p0"] = low
    L += ["", "## Construct validity: label of a case with no true degradation, by base accuracy p0 (K = 10)", "",
          "| p0 | P(robust) | P(dropped) |", "|---|---|---|"] + [f"| {p} | {v['robust']:.2f} | {v['dropped']:.2f} |" for p, v in low.items()]
    L += ["", "A model that never solves the problem (p0 = 0) is labelled robust with probability 1: the rule measures a drop, not competence."]
    out["rows"] = rows
    (RESULTS / "label_sensitivity.json").write_text(json.dumps(out, indent=1, default=str))
    (RESULTS / "label_sensitivity.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
