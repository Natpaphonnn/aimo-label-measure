# Two-stage white-box analysis `qwen_amc_v1` (analysis/run_wb.py)

Samples: 464 over {'Qwen/Qwen3.5-4B': 60} problems per model. Labels: data/labels/case_labels.csv.
Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.

## target = correct

stage-1 target: **correct**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 464 | 0.603 | 0.850 | 0.908 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 22 | 22 | 0.73 | 0.727 | 0.727 (0.000) | 0.636 (0.037) | 0.636 (0.037) | -0.091, p=0.625, nan | -0.091, p=0.625, nan | -0.091, p=0.625, nan |
| K2 | 22 | 22 | 0.73 | 0.727 | 0.727 (0.000) | 0.742 (0.021) | 0.697 (0.021) | +0.045, p=1.000, nan | +0.045, p=1.000, nan | -0.045, p=1.000, nan |
| K4 | 22 | 22 | 0.73 | 0.727 | 0.621 (0.021) | 0.621 (0.057) | 0.621 (0.021) | +0.045, p=1.000, nan | -0.045, p=1.000, nan | +0.000, p=1.000, nan |
| K8 | 0 | | | | | | | | | |

## target = consistent

stage-1 target: **consistent**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 464 | 0.603 | 0.821 | 0.908 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 22 | 22 | 0.73 | 0.727 | 0.727 (0.000) | 0.636 (0.000) | 0.636 (0.000) | -0.091, p=0.500, nan | -0.091, p=0.500, nan | -0.091, p=0.500, nan |
| K2 | 22 | 22 | 0.73 | 0.727 | 0.727 (0.000) | 0.636 (0.037) | 0.652 (0.021) | -0.045, p=1.000, nan | -0.045, p=1.000, nan | -0.045, p=1.000, nan |
| K4 | 22 | 22 | 0.73 | 0.727 | 0.621 (0.021) | 0.667 (0.043) | 0.621 (0.043) | +0.091, p=0.625, nan | +0.000, p=1.000, nan | +0.045, p=1.000, nan |
| K8 | 0 | | | | | | | | | |

