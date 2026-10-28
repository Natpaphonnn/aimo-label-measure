# Two-stage white-box analysis `v1all3` (analysis/run_wb.py)

Samples: 2922 over {'Qwen/Qwen3.5-4B': 169, 'Skywork/Skywork-OR1-Math-7B': 109, 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 109} problems per model. Labels: data/labels/case_labels.csv.
Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.

## target = correct

stage-1 target: **correct**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 962 | 0.406 | 0.897 | 0.875 |
| Skywork/Skywork-OR1-Math-7B | 584 | 0.202 | 0.898 | 0.952 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 584 | 0.125 | 0.725 | 0.690 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 128 | 76 | 0.66 | 0.656 | 0.656 (0.000) | 0.651 (0.007) | 0.651 (0.007) | +0.000, p=1.000, nan | +0.000, p=1.000, nan | +0.000, p=1.000, nan |
| K2 | 128 | 76 | 0.66 | 0.656 | 0.630 (0.018) | 0.609 (0.034) | 0.599 (0.030) | -0.023, p=0.629, 0.09 | -0.062, p=0.057, 0.09 | -0.047, p=0.180, 0.09 |
| K4 | 128 | 76 | 0.66 | 0.656 | 0.589 (0.035) | 0.628 (0.010) | 0.591 (0.033) | +0.086, p=0.027, 0.10 | -0.031, p=0.388, 0.08 | +0.008, p=1.000, 0.07 |
| K8 | 80 | 61 | 0.68 | 0.675 | 0.654 (0.016) | 0.658 (0.012) | 0.650 (0.027) | +0.013, p=1.000, 0.14 | -0.025, p=0.500, nan | -0.025, p=0.688, nan |

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.602 | 0.610 | 0.677 | 0.644 |
| wb | 0.624 | 0.594 | 0.675 | 0.637 |
| hybrid | 0.624 | 0.594 | 0.675 | 0.637 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.586 | 0.601 | 0.652 | 0.627 |
| wb | 0.598 | 0.565 | 0.659 | 0.632 |
| hybrid | 0.591 | 0.568 | 0.627 | 0.612 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.596 | 0.609 | 0.631 | 0.599 |
| wb | 0.627 | 0.585 | 0.652 | 0.642 |
| hybrid | 0.624 | 0.576 | 0.632 | 0.621 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K8 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.551 | 0.607 | 0.662 |
| wb | 0.529 | 0.546 | 0.655 |
| hybrid | 0.521 | 0.567 | 0.655 |
| prior | 0.559 | 0.625 | 0.673 |

## target = consistent

stage-1 target: **consistent**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 962 | 0.406 | 0.787 | 0.875 |
| Skywork/Skywork-OR1-Math-7B | 584 | 0.202 | 0.823 | 0.952 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 584 | 0.125 | 0.412 | 0.690 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 128 | 76 | 0.66 | 0.656 | 0.656 (0.000) | 0.659 (0.004) | 0.659 (0.004) | +0.000, p=1.000, nan | +0.000, p=1.000, nan | +0.000, p=1.000, nan |
| K2 | 128 | 76 | 0.66 | 0.656 | 0.630 (0.018) | 0.620 (0.004) | 0.602 (0.023) | +0.000, p=1.000, 0.08 | -0.039, p=0.180, 0.07 | -0.047, p=0.109, 0.07 |
| K4 | 128 | 76 | 0.66 | 0.656 | 0.589 (0.035) | 0.633 (0.013) | 0.607 (0.031) | +0.109, p=0.004, 0.11 | -0.008, p=1.000, nan | +0.023, p=0.453, nan |
| K8 | 80 | 61 | 0.68 | 0.675 | 0.654 (0.016) | 0.662 (0.000) | 0.633 (0.016) | +0.025, p=0.774, 0.12 | -0.013, p=1.000, nan | +0.000, p=1.000, nan |

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.602 | 0.610 | 0.677 | 0.644 |
| wb | 0.614 | 0.586 | 0.646 | 0.639 |
| hybrid | 0.614 | 0.586 | 0.646 | 0.639 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.586 | 0.601 | 0.652 | 0.627 |
| wb | 0.643 | 0.575 | 0.636 | 0.620 |
| hybrid | 0.625 | 0.584 | 0.625 | 0.620 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.596 | 0.609 | 0.631 | 0.599 |
| wb | 0.618 | 0.547 | 0.622 | 0.620 |
| hybrid | 0.619 | 0.576 | 0.601 | 0.584 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K8 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 |
|---|---|---|---|
| bb | 0.551 | 0.607 | 0.662 |
| wb | 0.536 | 0.591 | 0.630 |
| hybrid | 0.528 | 0.611 | 0.650 |
| prior | 0.559 | 0.625 | 0.673 |

