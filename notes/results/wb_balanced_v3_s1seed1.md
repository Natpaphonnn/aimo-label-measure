# Stage 2 under label shift: balanced accuracy `v3_s1seed1` (analysis/wb_balanced.py)

Stage 2 fit with class_weight='balanced'; metric = balanced accuracy (prior = 0.500 by construction). Mean over 3 stage-2 seeds;
CI = 95% cluster bootstrap over problems on the last seed; p = two-sided bootstrap share of resamples on the other side of 0.

## target = correct

| K | cases | robust share | bb | wb | hybrid | wb - prior [CI] p | wb - bb [CI] p | hybrid - prior [CI] p |
|---|---|---|---|---|---|---|---|---|
| 1 | 158 | 0.70 | 0.549 | 0.594 | 0.594 | +0.085 [+0.001, +0.172] p=0.048 | +0.046 [-0.045, +0.148] p=0.321 | +0.085 [+0.001, +0.172] p=0.048 |
| 2 | 158 | 0.70 | 0.535 | 0.612 | 0.581 | +0.081 [-0.019, +0.181] p=0.106 | +0.010 [-0.076, +0.094] p=0.822 | +0.082 [-0.013, +0.178] p=0.096 |
| 4 | 158 | 0.70 | 0.657 | 0.600 | 0.593 | +0.092 [+0.002, +0.183] p=0.048 | -0.044 [-0.138, +0.048] p=0.377 | +0.059 [-0.027, +0.151] p=0.167 |
| 8 | 139 | 0.71 | 0.609 | 0.559 | 0.584 | +0.052 [-0.044, +0.149] p=0.302 | -0.071 [-0.188, +0.035] p=0.195 | +0.093 [-0.019, +0.213] p=0.106 |

