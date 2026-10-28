#!/usr/bin/env python3
"""Label-noise ceiling: simulate the organizers' labelling process (K samples per original and per
perturbation type, label robust iff max_drop <= 0.10, non-robust iff >= 0.25, else dropped) and
measure (a) how often a case with NO true degradation is labelled non-robust, (b) the accuracy a
predictor that knows the TRUE solve probabilities can reach on the sampled labels (the ceiling),
(c) test-retest agreement of the labels, (d) the minimum detectable effect on a 100-240 case test.
Also evaluates the older 'any variant degrades' rule. Pure numpy.
"""
import numpy as np, itertools, pathlib, math
from scipy.stats import binom
rng = np.random.default_rng(20261015)
OUT = pathlib.Path(__file__).resolve().parents[1] / "notes" / "results" / "label_noise_ceiling.md"
OUT.parent.mkdir(parents=True, exist_ok=True)
K, T = 10, 7
TYPES = ["no_solution", "rephrase", "expert", "domain", "typos", "rename", "distract"]
# share of (model,problem,type) rows with observed drop >= 0.25 in aimo-interp-challenge-sample-full
OBS_SHARE = np.array([0.327, 0.122, 0.122, 0.111, 0.089, 0.078, 0.044])
lines = []
def P(s=""): print(s); lines.append(s)

def label(maxdrop):  # 1 robust, 0 non-robust, -1 dropped
    return np.where(maxdrop <= 0.10 + 1e-9, 1, np.where(maxdrop >= 0.25 - 1e-9, 0, -1))

def simulate(p0, drops, reps=1, rule="band"):
    """p0 (n,), drops (n,T) true drops -> sampled labels (reps, n)."""
    n = p0.shape[0]
    pt = np.clip(p0[:, None] - drops, 0.0, 1.0)
    base_hat = rng.binomial(K, np.broadcast_to(p0, (reps, n))) / K
    acc_t = rng.binomial(K, np.broadcast_to(pt, (reps, n, pt.shape[1]))) / K
    md = (base_hat[:, :, None] - acc_t).max(axis=2)
    if rule == "band":
        return label(md)
    # old rule: non-robust iff any type accuracy below base accuracy
    return np.where(md > 1e-9, 0, 1)

P("# Label-noise ceiling (simulation, K=10 samples per prompt, 7 perturbation types)\n")
P("Labelling rule: max_drop = max over types of (base_acc - type_acc); robust iff <= 0.10, non-robust iff >= 0.25, band dropped (train-main-v2 card). ")
P("'Old rule' = non-robust iff any type accuracy is below base accuracy (val-sample / MATH sets).\n")

# (a) false non-robust rate when NO type truly degrades
P("## A. Cases with no true degradation: probability of each sampled label, by true solve probability p0\n")
P("| p0 | P(robust) | P(dropped) | P(non-robust) | old rule P(non-robust) |")
P("|---|---|---|---|---|")
for p0v in [1.0, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3]:
    p0 = np.full(20000, p0v); d = np.zeros((20000, T))
    lab = simulate(p0, d)[0]; old = simulate(p0, d, rule="old")[0]
    P(f"| {p0v:.2f} | {np.mean(lab==1):.2f} | {np.mean(lab==-1):.2f} | {np.mean(lab==0):.2f} | {np.mean(old==0):.2f} |")
P("\nReading: unless the model is essentially perfect on the original (p0 >= 0.97), a case with zero true degradation is labelled non-robust a large share of the time. "
  "The band rule removes some of these but far from all. Under the labelling, 'robust' is close to 'solves the original with probability ~1 and no family truly degrades'.\n")

# (b) AIMO-like world: empirical p0 from sample-full (base>0), true drops as mixture
P("## B. AIMO-like world: ceiling of a predictor that knows the truth\n")
P("p0 drawn from the empirical base-accuracy distribution of aimo-interp-challenge-sample-full (cases with base > 0, which the organizers appear to keep); ")
P("each type truly degrades with probability pi_t (drop ~ U(0.3, 1.0)), pi_t = scale x observed share of rows with drop >= 0.25 (scale 0.5 corrects for sampling noise; 1.0 is an upper bound).\n")
base_vals = np.array([0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0]); base_cnt = np.array([1,2,3,1,5,3,5,7,33])
P("| types | pi scale | min p0 kept | label mix robust/dropped/non-robust | truth ambiguous among kept | majority acc | CEILING acc (knows truth) | solve-rate-only oracle acc (best thr) | test-retest agreement |")
P("|---|---|---|---|---|---|---|---|---|")
for tname, share in [("all 7", OBS_SHARE), ("6, no expert_no_solution (train-main-v2 convention)", OBS_SHARE[1:])]:
  for scale in [0.5, 1.0]:
    for minp0 in [0.0, 0.5]:
        n = 60000
        vals, cnt = base_vals[base_vals >= max(minp0, 1e-9)], base_cnt[base_vals >= max(minp0, 1e-9)]
        p0 = rng.choice(vals, size=n, p=cnt / cnt.sum())
        pi = np.clip(share * scale, 0, 1); Tt = len(share)
        has = rng.random((n, Tt)) < pi
        drops = np.where(has, rng.uniform(0.3, 1.0, (n, Tt)), 0.0)
        true_md = drops.max(axis=1); true_lab = label(true_md)  # on TRUE drops
        labs = simulate(p0, drops, reps=2)
        l1, l2 = labs[0], labs[1]
        kept = l1 != -1
        mix = f"{np.mean(l1==1):.2f}/{np.mean(l1==-1):.2f}/{np.mean(l1==0):.2f}"
        amb = np.mean(true_lab[kept] == -1)
        # Bayes-ish ceiling: predictor outputs the TRUE label (ambiguous truth -> predict robust if true_md<0.175)
        pred_true = np.where(true_lab == -1, (true_md < 0.175).astype(int), true_lab)
        ceiling = np.mean(pred_true[kept] == l1[kept])
        # solve-rate-only oracle: predict robust iff p0 >= thr, best thr
        best = max(((np.mean((p0[kept] >= thr).astype(int) == l1[kept]), thr) for thr in [0.55,0.65,0.75,0.85,0.95,0.999]))
        both = kept & (l2 != -1)
        rr = np.mean(l1[both] == l2[both])
        maj = max(np.mean(l1[kept] == 1), np.mean(l1[kept] == 0))
        P(f"| {tname} | {scale} | {minp0} | {mix} | {amb:.2f} | {maj:.3f} | {ceiling:.3f} | {best[0]:.3f} (thr {best[1]}) | {rr:.3f} |")
P("\nReading (corrected 2026-10-22; the v0 text said solve rate 'recovers most' of the ceiling, which the table does not show): even a predictor that knows the true "
  "per-type degradation is capped by label noise (ceiling column). Knowing only the true solve rate on the original moves accuracy from the majority "
  "baseline only part of the way to the ceiling; the rest needs information about genuine degradation, which with all 7 types is dominated by "
  "expert_no_solution. Compare the two 'types' blocks for how much of the label solve rate explains under the train-main-v2 convention.\n")

# (c) MDE on the test set
P("## C. What a test of N cases can resolve\n")
P("| N cases | 95% CI half-width at acc 0.6 | cases per 1 pt | discordant pairs m needed so a method winning 65% of them is significant (exact McNemar, two-sided 0.05) |")
P("|---|---|---|---|")
def min_m_for_win(frac):
    for m in range(4, 400):
        b = math.ceil(frac * m)
        if binom.sf(b - 1, m, 0.5) * 2 < 0.05:
            return m
    return None
for N in [100, 150, 210, 240]:
    hw = 1.96 * math.sqrt(0.6 * 0.4 / N)
    P(f"| {N} | {hw:.3f} | {N/100:.2f} | {min_m_for_win(0.65)} |")
P("\nNet lead needed over a constant baseline when the method is never wrong where the baseline is right (minimum discordance): " +
  ", ".join(f"{d} cases -> p={2**(1-d):.3f}" for d in [4,5,6,7,8]))
P("\nReading: a 2-3 point improvement on a ~200-case test is inside the noise; only differences of about 6+ points with many discordant cases can be shown significant.")
OUT.write_text("\n".join(lines) + "\n")
print(f"\n[written {OUT}]")
