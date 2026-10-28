# Label reliability vs samples per prompt (analysis/label_budget.py)

World of label_noise_ceiling.py section B (pi scale 0.5, all p0 > 0 kept). 'false NR' = P(non-robust) for a case with no true
degradation at p0 = 0.9. Ceiling = accuracy on the sampled labels of a predictor that knows every true degradation.
'solve-rate gain' = best-threshold accuracy of the true solve rate alone minus the majority class.

## all 7 families

| samples per prompt K | false NR at p0=0.9 | kept share | robust share | test-retest | oracle ceiling | majority | solve-rate gain |
|---|---|---|---|---|---|---|---|
| 10 | 0.18 | 0.91 | 0.54 | 0.896 | 0.888 | 0.540 | +0.089 |
| 20 | 0.06 | 0.90 | 0.56 | 0.937 | 0.938 | 0.561 | +0.039 |
| 40 | 0.00 | 0.89 | 0.60 | 0.985 | 0.984 | 0.599 | -0.018 |
| 100 | 0.00 | 0.94 | 0.63 | 1.000 | 0.997 | 0.626 | -0.041 |

## 6 families, no expert_no_solution

| samples per prompt K | false NR at p0=0.9 | kept share | robust share | test-retest | oracle ceiling | majority | solve-rate gain |
|---|---|---|---|---|---|---|---|
| 10 | 0.16 | 0.90 | 0.65 | 0.884 | 0.886 | 0.652 | +0.023 |
| 20 | 0.05 | 0.89 | 0.68 | 0.929 | 0.936 | 0.681 | -0.017 |
| 40 | 0.00 | 0.89 | 0.72 | 0.984 | 0.984 | 0.725 | -0.048 |
| 100 | 0.00 | 0.94 | 0.75 | 1.000 | 0.997 | 0.750 | -0.065 |

