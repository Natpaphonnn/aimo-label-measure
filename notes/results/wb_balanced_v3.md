# Stage 2 under label shift: balanced accuracy `v3` (analysis/wb_balanced.py)

Stage 2 fit with class_weight='balanced'; metric = balanced accuracy (prior = 0.500 by construction). Mean over 3 stage-2 seeds;
CI = 95% cluster bootstrap over problems on the last seed; p = two-sided bootstrap share of resamples on the other side of 0.

## target = correct

| K | cases | robust share | bb | wb | hybrid | wb - prior [CI] p | wb - bb [CI] p | hybrid - prior [CI] p |
|---|---|---|---|---|---|---|---|---|
| 1 | 158 | 0.70 | 0.549 | 0.576 | 0.576 | +0.070 [-0.021, +0.162] p=0.129 | +0.032 [-0.060, +0.132] p=0.499 | +0.070 [-0.021, +0.162] p=0.129 |
| 2 | 158 | 0.70 | 0.535 | 0.593 | 0.569 | +0.087 [-0.006, +0.182] p=0.068 | +0.016 [-0.063, +0.094] p=0.685 | +0.072 [-0.029, +0.175] p=0.168 |
| 4 | 158 | 0.70 | 0.657 | 0.619 | 0.610 | +0.106 [+0.008, +0.203] p=0.033 | -0.030 [-0.125, +0.060] p=0.531 | +0.084 [-0.004, +0.182] p=0.063 |
| 8 | 139 | 0.71 | 0.609 | 0.533 | 0.594 | +0.022 [-0.070, +0.113] p=0.637 | -0.101 [-0.207, +0.002] p=0.055 | +0.083 [-0.029, +0.198] p=0.137 |

## target = consistent

| K | cases | robust share | bb | wb | hybrid | wb - prior [CI] p | wb - bb [CI] p | hybrid - prior [CI] p |
|---|---|---|---|---|---|---|---|---|
| 1 | 158 | 0.70 | 0.549 | 0.566 | 0.566 | +0.063 [-0.017, +0.146] p=0.123 | +0.025 [-0.046, +0.104] p=0.495 | +0.063 [-0.017, +0.146] p=0.123 |
| 2 | 158 | 0.70 | 0.535 | 0.526 | 0.512 | -0.020 [-0.105, +0.066] p=0.656 | -0.090 [-0.192, +0.009] p=0.069 | +0.010 [-0.077, +0.095] p=0.825 |
| 4 | 158 | 0.70 | 0.657 | 0.575 | 0.606 | +0.066 [-0.026, +0.154] p=0.163 | -0.071 [-0.186, +0.036] p=0.213 | +0.093 [-0.003, +0.188] p=0.066 |
| 8 | 139 | 0.71 | 0.609 | 0.491 | 0.596 | -0.005 [-0.101, +0.093] p=0.931 | -0.128 [-0.236, -0.021] p=0.012 | +0.101 [-0.011, +0.216] p=0.085 |


Class recalls and break-even robust share (correct target, last seed): accuracy at robust share pi = pi*TPR + (1-pi)*TNR; the arm beats always-robust when pi < TNR/(1-TPR+TNR).

| K | bb TPR/TNR/break-even | wb | hybrid |
|---|---|---|---|
| K1 | 0.47/0.60/0.53 | 0.54/0.60/0.57 | 0.54/0.60/0.57 |
| K2 | 0.60/0.54/0.58 | 0.53/0.65/0.58 | 0.52/0.62/0.56 |
| K4 | 0.63/0.65/0.63 | 0.55/0.67/0.59 | 0.56/0.60/0.58 |
| K8 | 0.65/0.60/0.63 | 0.49/0.55/0.52 | 0.62/0.55/0.59 |
