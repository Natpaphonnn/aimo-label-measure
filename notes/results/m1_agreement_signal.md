# M1 progress (v2, 2026-10-16): test-time answer agreement against the per-model prior, under the organizers' label convention

v2 replaces the first version of this note (same day). Three things changed: (1) the organizers confirmed that never/rarely-solved
originals are excluded from labels (FACTS D3), so cases are now dropped when the held-out base accuracy is below a cut (primary 0.5,
DECISIONS #18); (2) the baseline is the in-sample per-model majority, because the leave-one-problem-out majority used in v1 flips when a
model's robust share is near 0.5 and then scores below the base rate (DECISIONS #19a); (3) seeds now change the folds (#19b; in v1 all
three seeds used identical folds, so "mean over seeds" was one run).

Data: byeolki/aimo-interp generations (MIT) for Qwen3.5-4B, Skywork-OR1-Math-7B, DeepSeek-R1-0528-Qwen3-8B on our leakage-filtered
competition-level pool (AIME 2022-23, AMC, HMMT/BRUMO/SMT/CMIMC 2025-26); T = 1.0, top-k 40, top-p 0.95, 8192-token cap, forced answer
on truncation. Variants: rename / typos / distract only (6 samples each; no expert_no_solution, matching train-main-v2, FACTS D9).

Protocol (no leakage between features and labels): test-time features use original samples [0, K) and NO gold answer (answer agreement,
distinct answers, null answers, truncation, log length); the label uses the disjoint original samples [K, 8) as base accuracy and the variant
samples, band rule 0.10/0.25, cases with held-out base < cut dropped. Within-model = a separate classifier per model; pooled = one classifier
with model one-hot (shared slope, per-model offset). Grouped 5-fold CV by problem x 3 seeds (sd over seeds in brackets); paired statistics
for the last seed against the in-sample per-model majority. Code: `analysis/m1_sim_byeolki.py K CUT`; JSON per cell in
`notes/results/m1_sim_byeolki_k{K}[_mb{CUT}].json`.

| K | min base | n | robust share | prior (in-sample) | within-model acc (sd) | delta | 95% CI | W/L | p | MDE80 | pooled acc (sd) | delta | 95% CI | p | MDE80 | pooled gain per model (Qwen / Skywork / DeepSeek) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 145 | 0.64 | 0.641 | 0.614 (0.015) | -0.034 | [-0.110, +0.041] | 11/16 | 0.442 | 0.105 | 0.616 (0.003) | -0.021 | [-0.062, +0.021] | 0.508 | 0.060 | +0.00 (n=56) / -0.01 (n=56) / -0.10 (n=33) |
| 1 | 0.3 | 128 | 0.66 | 0.664 | 0.680 (0.022) | +0.031 | [-0.008, +0.070] | 5/1 | 0.219 | nan | 0.648 (0.011) | +0.000 | [-0.055, +0.055] | 1.000 | 0.085 | +0.01 (n=48) / -0.04 (n=51) / -0.01 (n=29) |
| 1 | 0.5 | 112 | 0.71 | 0.714 | 0.699 (0.011) | -0.027 | [-0.071, +0.009] | 1/4 | 0.375 | nan | 0.708 (0.004) | -0.009 | [-0.027, +0.000] | 1.000 | nan | +0.00 (n=47) / +0.00 (n=43) / -0.03 (n=22) |
| 1 | 0.7 | 105 | 0.73 | 0.733 | 0.721 (0.018) | +0.000 | [+0.000, +0.000] | 0/0 | 1.000 | nan | 0.727 (0.004) | -0.010 | [-0.029, +0.000] | 1.000 | nan | +0.00 (n=44) / +0.00 (n=43) / -0.04 (n=18) |
| 2 | 0 | 141 | 0.65 | 0.645 | 0.697 (0.007) | +0.043 | [-0.021, +0.113] | 15/9 | 0.307 | 0.103 | 0.700 (0.003) | +0.057 | [-0.028, +0.142] | 0.243 | 0.125 | -0.04 (n=55) / +0.09 (n=54) / +0.16 (n=32) |
| 2 | 0.3 | 132 | 0.62 | 0.636 | 0.730 (0.014) | +0.083 | [+0.008, +0.159] | 19/8 | 0.052 | 0.115 | 0.780 (0.000) | +0.144 | [+0.053, +0.235] | 0.003 | 0.138 | +0.14 (n=51) / +0.11 (n=53) / +0.21 (n=28) |
| 2 | 0.5 | 124 | 0.65 | 0.669 | 0.707 (0.004) | +0.040 | [-0.024, +0.105] | 11/6 | 0.332 | 0.095 | 0.766 (0.000) | +0.097 | [+0.008, +0.185] | 0.043 | 0.130 | +0.06 (n=48) / +0.06 (n=48) / +0.21 (n=28) |
| 2 | 0.7 | 95 | 0.71 | 0.705 | 0.688 (0.018) | +0.000 | [-0.053, +0.053] | 3/3 | 1.000 | nan | 0.740 (0.005) | +0.032 | [-0.032, +0.095] | 0.508 | 0.090 | +0.00 (n=40) / +0.06 (n=37) / +0.05 (n=18) |
| 3 | 0 | 147 | 0.65 | 0.653 | 0.671 (0.012) | +0.007 | [-0.068, +0.082] | 17/16 | 1.000 | 0.115 | 0.685 (0.012) | +0.027 | [-0.048, +0.109] | 0.618 | 0.120 | +0.02 (n=59) / +0.04 (n=54) / +0.05 (n=34) |
| 3 | 0.3 | 134 | 0.62 | 0.627 | 0.764 (0.018) | +0.149 | [+0.067, +0.231] | 29/9 | 0.002 | 0.135 | 0.776 (0.006) | +0.149 | [+0.067, +0.239] | 0.002 | 0.135 | +0.11 (n=55) / +0.15 (n=50) / +0.22 (n=29) |
| 3 | 0.5 | 119 | 0.66 | 0.655 | 0.717 (0.014) | +0.042 | [-0.034, +0.126] | 15/10 | 0.424 | 0.122 | 0.762 (0.022) | +0.118 | [+0.050, +0.193] | 0.004 | 0.115 | +0.05 (n=49) / +0.09 (n=46) / +0.25 (n=24) |
| 3 | 0.7 | 101 | 0.66 | 0.663 | 0.703 (0.029) | +0.000 | [-0.059, +0.059] | 5/5 | 1.000 | 0.090 | 0.762 (0.000) | +0.099 | [+0.030, +0.178] | 0.021 | 0.115 | +0.07 (n=42) / +0.10 (n=38) / +0.14 (n=21) |
| 4 | 0 | 167 | 0.58 | 0.593 | 0.687 (0.003) | +0.090 | [+0.012, +0.162] | 29/14 | 0.032 | 0.115 | 0.699 (0.006) | +0.114 | [+0.030, +0.192] | 0.009 | 0.122 | +0.09 (n=67) / +0.08 (n=60) / +0.16 (n=40) |
| 4 | 0.3 | 130 | 0.62 | 0.646 | 0.710 (0.018) | +0.077 | [+0.000, +0.146] | 18/8 | 0.076 | 0.115 | 0.741 (0.013) | +0.100 | [+0.008, +0.192] | 0.047 | 0.138 | +0.05 (n=55) / +0.07 (n=47) / +0.23 (n=28) |
| 4 | 0.5 | 130 | 0.62 | 0.646 | 0.710 (0.018) | +0.077 | [+0.000, +0.146] | 18/8 | 0.076 | 0.115 | 0.741 (0.013) | +0.100 | [+0.008, +0.192] | 0.047 | 0.138 | +0.05 (n=55) / +0.07 (n=47) / +0.23 (n=28) |
| 4 | 0.7 | 108 | 0.66 | 0.657 | 0.707 (0.019) | +0.074 | [+0.000, +0.148] | 13/5 | 0.096 | 0.113 | 0.738 (0.004) | +0.083 | [+0.000, +0.167] | 0.108 | 0.135 | +0.06 (n=47) / +0.02 (n=40) / +0.24 (n=21) |

Per-model AUC of agreement at cut 0.5: K = 2: Qwen 0.66, Skywork 0.67, DeepSeek 0.76; K = 4: 0.72 / 0.67 / 0.82 (the held-out solve rate
itself has AUC 0.53-0.59 for Qwen and Skywork: agreement is not a mere solve-rate proxy for them).

Reading:
- With one sample (K = 1) nothing beats the prior. From K = 2 on, under the organizers' convention (cut 0.3-0.7), the **pooled** model
  beats the per-model prior in all 9 cells (K = 2-4 x cut 0.3/0.5/0.7), by +3 to +15 points; 7 of the 9 have p < 0.05. At the primary cut 0.5: K = 2
  +9.7 points (p = 0.043, MDE80 0.13), K = 3 +11.8 (p = 0.004, MDE80 0.12), K = 4 +10.0 (p = 0.047, MDE80 0.14). The gain is positive
  in every model in those cells and largest for DeepSeek (the model whose robust share is near 0.5).
- The **separate per-model** classifiers are weaker and unstable (20-55 cases per model): no cell with cut 0.5 is significant. A
  shared agreement-to-robustness mapping with per-model offsets is the better-supported form, and it is the one that could transfer to
  gpt-oss-120b, for which we have no self-labels.
- Choosing among 16 cells x 2 classifiers is a forking-paths problem: this simulation is exploratory. The confirmatory test is
  pre-registered in DECISIONS #21 (pooled, cut 0.5, K fixed by the runtime) and will run on our own generations.

Caveats: (1) 8k-token traces; the per-case server limit allows about 0.8-1.4k tokens per sample for the 7-8B models (DECISIONS #20),
so the signal must be re-measured at 512-2048-token cuts (night-1 + force_at_cuts); (2) cheap families only; (3) K = 4 leaves only
4 base samples, so its cuts 0.3 and 0.5 coincide; (4) 59-64 problems, 3 models; not yet tested against the official labels.
