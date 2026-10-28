# White-box ablation on held-out self-labelled cases (analysis/wb_vs_offsets.py; post hoc, DECISIONS #54)

N = 158 cases, 78 problems, K = 4, balanced accuracy, stage 2 class-balanced, grouped 5-fold CV.
Paired cluster bootstrap over problems, 10,000 resamples per stage-2 seed; p = two-sided share of resamples on the other side of 0.

| comparison | seed | delta | 95% CI | p |
|---|---|---|---|---|
| wb - prior | 0 | +11.8 | [+2.7, +20.8] | 0.014 |
| wb - prior | 1 | +13.3 | [+3.4, +23.0] | 0.010 |
| wb - prior | 2 | +10.6 | [+0.7, +20.2] | 0.036 |
| wb - prior | mean | +11.9 | | |
| wb - offsets | 0 | +6.9 | [-3.3, +17.0] | 0.188 |
| wb - offsets | 1 | +7.4 | [-2.3, +16.9] | 0.136 |
| wb - offsets | 2 | +6.8 | [-4.4, +18.2] | 0.238 |
| wb - offsets | mean | +7.0 | | |
| bb - prior | 0 | +17.1 | [+7.8, +26.5] | 0.000 |
| bb - prior | 1 | +16.2 | [+6.9, +25.0] | 0.001 |
| bb - prior | 2 | +13.7 | [+3.8, +23.1] | 0.009 |
| bb - prior | mean | +15.7 | | |
| wb - bb | 0 | -5.3 | [-15.1, +3.6] | 0.264 |
| wb - bb | 1 | -2.9 | [-12.4, +5.7] | 0.538 |
| wb - bb | 2 | -3.0 | [-12.3, +6.0] | 0.519 |
| wb - bb | mean | -3.8 | | |

Balanced accuracy per seed: wb 0.618/0.633/0.606; bb 0.671/0.662/0.637; offsets 0.549/0.559/0.538; prior 0.500/0.500/0.500
