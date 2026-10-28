# Label-noise ceiling (simulation, K=10 samples per prompt, 7 perturbation types)

Labelling rule: max_drop = max over types of (base_acc - type_acc); robust iff <= 0.10, non-robust iff >= 0.25, band dropped (train-main-v2 card). 
'Old rule' = non-robust iff any type accuracy is below base accuracy (val-sample / MATH sets).

## A. Cases with no true degradation: probability of each sampled label, by true solve probability p0

| p0 | P(robust) | P(dropped) | P(non-robust) | old rule P(non-robust) |
|---|---|---|---|---|
| 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 0.97 | 0.83 | 0.15 | 0.02 | 0.70 |
| 0.95 | 0.70 | 0.26 | 0.05 | 0.73 |
| 0.90 | 0.53 | 0.30 | 0.17 | 0.78 |
| 0.85 | 0.46 | 0.27 | 0.27 | 0.78 |
| 0.80 | 0.42 | 0.25 | 0.32 | 0.80 |
| 0.70 | 0.38 | 0.22 | 0.39 | 0.80 |
| 0.60 | 0.37 | 0.21 | 0.42 | 0.81 |
| 0.50 | 0.37 | 0.22 | 0.41 | 0.81 |
| 0.30 | 0.43 | 0.24 | 0.33 | 0.79 |

Reading: unless the model is essentially perfect on the original (p0 >= 0.97), a case with zero true degradation is labelled non-robust a large share of the time. The band rule removes some of these but far from all. Under the labelling, 'robust' is close to 'solves the original with probability ~1 and no family truly degrades'.

## B. AIMO-like world: ceiling of a predictor that knows the truth

p0 drawn from the empirical base-accuracy distribution of aimo-interp-challenge-sample-full (cases with base > 0, which the organizers appear to keep); 
each type truly degrades with probability pi_t (drop ~ U(0.3, 1.0)), pi_t = scale x observed share of rows with drop >= 0.25 (scale 0.5 corrects for sampling noise; 1.0 is an upper bound).

| types | pi scale | min p0 kept | label mix robust/dropped/non-robust | truth ambiguous among kept | majority acc | CEILING acc (knows truth) | solve-rate-only oracle acc (best thr) | test-retest agreement |
|---|---|---|---|---|---|---|---|---|
| all 7 | 0.5 | 0.0 | 0.49/0.09/0.42 | 0.00 | 0.537 | 0.886 | 0.628 (thr 0.999) | 0.895 |
| all 7 | 0.5 | 0.5 | 0.50/0.08/0.43 | 0.00 | 0.537 | 0.910 | 0.649 (thr 0.999) | 0.914 |
| all 7 | 1.0 | 0.0 | 0.31/0.07/0.62 | 0.00 | 0.669 | 0.915 | 0.536 (thr 0.999) | 0.923 |
| all 7 | 1.0 | 0.5 | 0.30/0.06/0.65 | 0.00 | 0.683 | 0.944 | 0.539 (thr 0.999) | 0.943 |
| 6, no expert_no_solution (train-main-v2 convention) | 0.5 | 0.0 | 0.59/0.10/0.32 | 0.00 | 0.650 | 0.884 | 0.672 (thr 0.85) | 0.886 |
| 6, no expert_no_solution (train-main-v2 convention) | 0.5 | 0.5 | 0.60/0.08/0.32 | 0.00 | 0.651 | 0.904 | 0.701 (thr 0.85) | 0.905 |
| 6, no expert_no_solution (train-main-v2 convention) | 1.0 | 0.0 | 0.44/0.08/0.47 | 0.00 | 0.517 | 0.902 | 0.590 (thr 0.999) | 0.905 |
| 6, no expert_no_solution (train-main-v2 convention) | 1.0 | 0.5 | 0.44/0.07/0.49 | 0.00 | 0.525 | 0.925 | 0.609 (thr 0.999) | 0.926 |

Reading (corrected 2026-10-22; the v0 text said solve rate 'recovers most' of the ceiling, which the table does not show): even a predictor that knows the true per-type degradation is capped by label noise (ceiling column). Knowing only the true solve rate on the original moves accuracy from the majority baseline only part of the way to the ceiling; the rest needs information about genuine degradation, which with all 7 types is dominated by expert_no_solution. Compare the two 'types' blocks for how much of the label solve rate explains under the train-main-v2 convention.

## C. What a test of N cases can resolve

| N cases | 95% CI half-width at acc 0.6 | cases per 1 pt | discordant pairs m needed so a method winning 65% of them is significant (exact McNemar, two-sided 0.05) |
|---|---|---|---|
| 100 | 0.096 | 1.00 | 37 |
| 150 | 0.078 | 1.50 | 37 |
| 210 | 0.066 | 2.10 | 37 |
| 240 | 0.062 | 2.40 | 37 |

Net lead needed over a constant baseline when the method is never wrong where the baseline is right (minimum discordance): 4 cases -> p=0.125, 5 cases -> p=0.062, 6 cases -> p=0.031, 7 cases -> p=0.016, 8 cases -> p=0.008

Reading: a 2-3 point improvement on a ~200-case test is inside the noise; only differences of about 6+ points with many discordant cases can be shown significant.
