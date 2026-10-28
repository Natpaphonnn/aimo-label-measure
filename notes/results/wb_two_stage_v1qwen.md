# Two-stage white-box analysis `v1qwen` (analysis/run_wb.py)

Samples: 1178 over {'Qwen/Qwen3.5-4B': 169} problems per model. Labels: data/labels/case_labels.csv.
Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.

## target = correct

stage-1 target: **correct**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 962 | 0.406 | 0.897 | 0.875 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 63 | 63 | 0.68 | 0.683 | 0.683 (0.000) | 0.646 (0.020) | 0.646 (0.020) | -0.032, p=0.688, nan | -0.032, p=0.688, nan | -0.032, p=0.688, nan |
| K2 | 63 | 63 | 0.68 | 0.683 | 0.651 (0.000) | 0.646 (0.007) | 0.656 (0.007) | +0.000, p=1.000, 0.14 | -0.032, p=0.727, 0.12 | +0.016, p=1.000, 0.15 |
| K4 | 63 | 63 | 0.68 | 0.683 | 0.640 (0.020) | 0.656 (0.027) | 0.598 (0.015) | +0.000, p=1.000, 0.14 | -0.016, p=1.000, nan | -0.048, p=0.508, 0.13 |
| K8 | 0 | | | | | | | | | |

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.692 | 0.688 | 0.729 |
| wb | 0.613 | 0.675 | 0.671 |
| hybrid | 0.613 | 0.675 | 0.671 |
| prior | 0.692 | 0.688 | 0.729 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.633 | 0.662 | 0.671 |
| wb | 0.600 | 0.625 | 0.708 |
| hybrid | 0.579 | 0.629 | 0.683 |
| prior | 0.692 | 0.688 | 0.729 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.646 | 0.662 | 0.654 |
| wb | 0.613 | 0.608 | 0.646 |
| hybrid | 0.621 | 0.617 | 0.596 |
| prior | 0.692 | 0.688 | 0.729 |

## target = consistent

stage-1 target: **consistent**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 962 | 0.406 | 0.787 | 0.875 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 63 | 63 | 0.68 | 0.683 | 0.683 (0.000) | 0.651 (0.026) | 0.651 (0.026) | -0.032, p=0.500, nan | -0.032, p=0.500, nan | -0.032, p=0.500, nan |
| K2 | 63 | 63 | 0.68 | 0.683 | 0.651 (0.000) | 0.656 (0.033) | 0.614 (0.042) | -0.032, p=0.727, 0.12 | -0.063, p=0.219, nan | -0.095, p=0.070, 0.12 |
| K4 | 63 | 63 | 0.68 | 0.683 | 0.640 (0.020) | 0.667 (0.026) | 0.640 (0.020) | +0.000, p=1.000, nan | -0.016, p=1.000, nan | -0.032, p=0.688, nan |
| K8 | 0 | | | | | | | | | |

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.692 | 0.688 | 0.729 |
| wb | 0.596 | 0.600 | 0.679 |
| hybrid | 0.596 | 0.600 | 0.679 |
| prior | 0.692 | 0.688 | 0.729 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.633 | 0.662 | 0.671 |
| wb | 0.592 | 0.575 | 0.667 |
| hybrid | 0.542 | 0.562 | 0.613 |
| prior | 0.692 | 0.688 | 0.729 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.646 | 0.662 | 0.654 |
| wb | 0.562 | 0.617 | 0.700 |
| hybrid | 0.550 | 0.608 | 0.629 |
| prior | 0.692 | 0.688 | 0.729 |

