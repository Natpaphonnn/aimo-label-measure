# Two-stage white-box analysis `pilot` (analysis/run_wb.py)

Samples: 120 over {'Qwen/Qwen3.5-4B': 60} problems per model. Labels: data/labels/case_labels.csv.
Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.

## target = correct

stage-1 target: **correct**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 120 | 0.608 | 0.786 | 0.797 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 22 | 22 | 0.73 | 0.727 | 0.727 (0.000) | 0.636 (0.000) | 0.636 (0.000) | -0.091, p=0.500, nan | -0.091, p=0.500, nan | -0.091, p=0.500, nan |
| K2 | 22 | 22 | 0.73 | 0.727 | 0.652 (0.043) | 0.621 (0.057) | 0.712 (0.043) | -0.045, p=1.000, nan | -0.182, p=0.125, nan | +0.091, p=0.625, nan |
| K4 | 0 | | | | | | | | | |
| K8 | 0 | | | | | | | | | |

## target = consistent

stage-1 target: **consistent**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 120 | 0.608 | 0.662 | 0.797 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 22 | 22 | 0.73 | 0.727 | 0.727 (0.000) | 0.712 (0.021) | 0.712 (0.021) | +0.000, p=1.000, nan | +0.000, p=1.000, nan | +0.000, p=1.000, nan |
| K2 | 22 | 22 | 0.73 | 0.727 | 0.652 (0.043) | 0.667 (0.021) | 0.667 (0.043) | +0.091, p=0.625, nan | -0.045, p=1.000, nan | +0.045, p=1.000, nan |
| K4 | 0 | | | | | | | | | |
| K8 | 0 | | | | | | | | | |

