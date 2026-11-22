# Sensitivity of the simulated label reliability (analysis/label_sensitivity.py)

Base world = label_noise_ceiling.py section B (K = 10, scale 0.5, drop U(0.3, 1), independent families, empirical base).
oracle = agreement of a predictor that knows every true degradation with the sampled labels (kept cases);
test-retest = agreement of two independent labellings; gain = best-threshold true solve rate minus majority.

| setting | oracle | test-retest | kept | robust share (kept) | solve-rate gain |
|---|---|---|---|---|---|
| base (scale 0.5, U(0.3,1), independent) | 0.888 | 0.897 | 0.90 | 0.54 | +0.083 |
| scale 0.25 | 0.869 | 0.876 | 0.90 | 0.68 | +0.016 |
| scale 0.75 | 0.897 | 0.908 | 0.91 | 0.42 | -0.004 |
| scale 1.0 | 0.912 | 0.921 | 0.93 | 0.34 | -0.130 |
| drop U(0.1,1) | 0.865 | 0.873 | 0.89 | 0.57 | +0.069 |
| drop U(0.5,1) | 0.892 | 0.903 | 0.92 | 0.53 | +0.093 |
| correlated families (copula rho 0.5) | 0.875 | 0.886 | 0.90 | 0.63 | +0.039 |
| 6 families (no expert_no_solution) | 0.874 | 0.882 | 0.90 | 0.64 | +0.036 |

Bootstrap over the 10 sample-full problems (family shares and base accuracies resampled together, 200 replicates), 5-95% range: oracle 0.825-0.932; test_retest 0.836-0.933; kept 0.872-0.934; robust_share_kept 0.465-0.603; solve_rate_gain 0.038-0.120

False non-robust at p0 = 0.9 with no true degradation (depends only on A1, K and the rule): K=10: 0.175, K=20: 0.058, K=40: 0.002

## Selection by the 0.10-0.25 band (base world)

kept share 0.90; truly degraded: all 0.37, kept 0.39, dropped 0.23; mean p0: kept 0.86, dropped 0.64.

## Construct validity: label of a case with no true degradation, by base accuracy p0 (K = 10)

| p0 | P(robust) | P(dropped) |
|---|---|---|
| 0.0 | 1.00 | 0.00 |
| 0.1 | 0.75 | 0.18 |
| 0.3 | 0.43 | 0.24 |
| 0.5 | 0.38 | 0.21 |
| 0.9 | 0.52 | 0.31 |

A model that never solves the problem (p0 = 0) is labelled robust with probability 1: the rule measures a drop, not competence.
