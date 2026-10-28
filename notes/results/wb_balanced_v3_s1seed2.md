# Stage 2 under label shift: balanced accuracy `v3_s1seed2` (analysis/wb_balanced.py)

Stage 2 fit with class_weight='balanced'; metric = balanced accuracy (prior = 0.500 by construction). Mean over 3 stage-2 seeds;
CI = 95% cluster bootstrap over problems on the last seed; p = two-sided bootstrap share of resamples on the other side of 0.

## target = correct

| K | cases | robust share | bb | wb | hybrid | wb - prior [CI] p | wb - bb [CI] p | hybrid - prior [CI] p |
|---|---|---|---|---|---|---|---|---|
| 1 | 158 | 0.70 | 0.549 | 0.630 | 0.630 | +0.113 [+0.019, +0.208] p=0.016 | +0.075 [-0.026, +0.187] p=0.143 | +0.113 [+0.019, +0.208] p=0.016 |
| 2 | 158 | 0.70 | 0.535 | 0.614 | 0.580 | +0.115 [+0.017, +0.211] p=0.032 | +0.044 [-0.043, +0.134] p=0.324 | +0.073 [-0.031, +0.176] p=0.162 |
| 4 | 158 | 0.70 | 0.657 | 0.622 | 0.611 | +0.118 [+0.032, +0.209] p=0.014 | -0.019 [-0.115, +0.073] p=0.678 | +0.115 [+0.030, +0.199] p=0.006 |
| 8 | 139 | 0.71 | 0.609 | 0.573 | 0.606 | +0.087 [-0.019, +0.193] p=0.109 | -0.036 [-0.152, +0.064] p=0.512 | +0.128 [+0.021, +0.240] p=0.021 |

