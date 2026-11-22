# Label decomposition with nested stage-1 scores (analysis/label_decomp_revision.py; pre-registration P2/P3)

139 labelled cases with 8 samples (34 official train-main-v2, 105 byeolki); 105 with 8 gold-graded samples. Robust = positive class. 95% cluster bootstrap over problems (10000).

## Monotone vs band (two-sided) AUC

| subset | predictor | n | AUC x [CI] | AUC abs(x-0.5) [CI] |
|---|---|---|---|---|
| all | gold solve rate at the cut | 105 | 0.651 [0.50, 0.81] | 0.521 [0.42, 0.63] |
| all | wb mean (nested probe) | 139 | 0.642 [0.51, 0.76] | 0.517 [0.41, 0.63] |
| all | agreement mean | 139 | 0.638 [0.50, 0.77] | 0.567 [0.45, 0.69] |
| all | labellers' base accuracy at 8k | 105 | 0.608 [0.48, 0.75] | 0.608 [0.48, 0.75] |
| byeolki | gold solve rate at the cut | 105 | 0.651 [0.50, 0.81] | 0.521 [0.42, 0.63] |
| byeolki | wb mean (nested probe) | 105 | 0.631 [0.48, 0.77] | 0.561 [0.43, 0.69] |
| byeolki | agreement mean | 105 | 0.644 [0.50, 0.80] | 0.581 [0.44, 0.73] |
| byeolki | labellers' base accuracy at 8k | 105 | 0.608 [0.48, 0.75] | 0.608 [0.48, 0.75] |
| official train-main-v2 | wb mean (nested probe) | 34 | 0.688 [0.51, 0.85] | 0.367 [0.18, 0.53] |
| official train-main-v2 | agreement mean | 34 | 0.642 [0.33, 0.83] | 0.496 [0.29, 0.81] |

## Nested rules on the 105 cases with 8 gold-graded samples (balanced accuracy, class-balanced fits, grouped 5-fold x 3 seeds)

| rule | balanced accuracy (seeds) | mean | CI (seed 0) |
|---|---|---|---|
| band_solve | 0.483 / 0.483 / 0.483 | 0.483 | [0.42, 0.55] |
| quad_solve | 0.593 / 0.593 / 0.607 | 0.598 | [0.46, 0.72] |
| band_wb | 0.457 / 0.527 / 0.563 | 0.516 | [0.32, 0.59] |
| quad_wb | 0.543 / 0.597 / 0.547 | 0.562 | [0.41, 0.67] |
| band_twin | 0.580 / 0.580 / 0.553 | 0.571 | [0.46, 0.70] |
| quad_twin | 0.603 / 0.620 / 0.657 | 0.627 | [0.48, 0.73] |
| solve | 0.613 / 0.607 / 0.633 | 0.618 | [0.48, 0.74] |
| solve_wb | 0.580 / 0.537 / 0.623 | 0.580 | [0.48, 0.68] |
| offsets | 0.550 / 0.570 / 0.570 | 0.563 | [0.46, 0.63] |

Adding the probe to the gold solve rate (P2 secondary): -3.3 / -7.0 / -1.0 points, mean -3.8, seed-0 CI [-11.6, +4.5].
Within-stratum AUC (wb_mean; same model, same number of correct samples out of 8; 105 cases): 0.552 [0.30, 0.73].
Within-stratum AUC (twin_mean; same model, same number of correct samples out of 8; 105 cases): 0.548 [0.44, 0.71].
