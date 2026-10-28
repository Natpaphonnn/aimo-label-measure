# Two-stage white-box analysis `v2` (analysis/run_wb.py)

Samples: 3080 over {'Qwen/Qwen3.5-4B': 169, 'Skywork/Skywork-OR1-Math-7B': 109, 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 109} problems per model. Labels: data/labels/case_labels.csv.
Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.

## target = correct

stage-1 target: **correct**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 1048 | 0.392 | 0.896 | 0.866 |
| Skywork/Skywork-OR1-Math-7B | 584 | 0.202 | 0.898 | 0.952 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 584 | 0.125 | 0.725 | 0.690 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 128 | 76 | 0.66 | 0.656 | 0.656 (0.000) | 0.651 (0.007) | 0.651 (0.007) | +0.000, p=1.000, nan | +0.000, p=1.000, nan | +0.000, p=1.000, nan |
| K2 | 128 | 76 | 0.66 | 0.656 | 0.630 (0.018) | 0.609 (0.034) | 0.602 (0.028) | -0.023, p=0.629, 0.09 | -0.062, p=0.057, 0.09 | -0.039, p=0.267, 0.08 |
| K4 | 128 | 76 | 0.66 | 0.656 | 0.589 (0.035) | 0.633 (0.011) | 0.596 (0.030) | +0.086, p=0.019, 0.10 | -0.031, p=0.344, 0.07 | +0.016, p=0.754, 0.07 |
| K8 | 121 | 69 | 0.67 | 0.669 | 0.625 (0.022) | 0.645 (0.007) | 0.598 (0.026) | +0.041, p=0.267, 0.09 | -0.033, p=0.125, nan | -0.033, p=0.125, nan |

Holm-adjusted p over the 8 primary comparisons: K1 wb_vs_bb 1.000; K1 wb_vs_prior 1.000; K2 wb_vs_bb 1.000; K2 wb_vs_prior 0.402; K4 wb_vs_bb 0.154; K4 wb_vs_prior 1.000; K8 wb_vs_bb 1.000; K8 wb_vs_prior 0.750

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.602 | 0.610 | 0.677 | 0.644 |
| wb | 0.624 | 0.573 | 0.675 | 0.639 |
| hybrid | 0.624 | 0.573 | 0.675 | 0.639 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.586 | 0.601 | 0.652 | 0.627 |
| wb | 0.602 | 0.575 | 0.655 | 0.637 |
| hybrid | 0.586 | 0.579 | 0.622 | 0.612 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.596 | 0.609 | 0.631 | 0.599 |
| wb | 0.629 | 0.586 | 0.641 | 0.639 |
| hybrid | 0.624 | 0.578 | 0.618 | 0.614 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K8 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.611 | 0.593 | 0.637 | 0.610 |
| wb | 0.596 | 0.577 | 0.635 | 0.649 |
| hybrid | 0.594 | 0.586 | 0.617 | 0.610 |
| prior | 0.637 | 0.609 | 0.643 | 0.683 |

## target = consistent

stage-1 target: **consistent**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 1048 | 0.392 | 0.787 | 0.866 |
| Skywork/Skywork-OR1-Math-7B | 584 | 0.202 | 0.823 | 0.952 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 584 | 0.125 | 0.412 | 0.690 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 128 | 76 | 0.66 | 0.656 | 0.656 (0.000) | 0.659 (0.004) | 0.659 (0.004) | +0.000, p=1.000, nan | +0.000, p=1.000, nan | +0.000, p=1.000, nan |
| K2 | 128 | 76 | 0.66 | 0.656 | 0.630 (0.018) | 0.638 (0.013) | 0.617 (0.006) | +0.039, p=0.125, nan | +0.000, p=1.000, nan | -0.008, p=1.000, nan |
| K4 | 128 | 76 | 0.66 | 0.656 | 0.589 (0.035) | 0.633 (0.011) | 0.591 (0.026) | +0.102, p=0.004, 0.10 | -0.016, p=0.500, nan | +0.016, p=0.625, nan |
| K8 | 121 | 69 | 0.67 | 0.669 | 0.625 (0.022) | 0.656 (0.019) | 0.623 (0.019) | +0.033, p=0.454, 0.10 | -0.041, p=0.062, nan | +0.000, p=1.000, nan |

Holm-adjusted p over the 8 primary comparisons: K1 wb_vs_bb 1.000; K1 wb_vs_prior 1.000; K2 wb_vs_bb 0.750; K2 wb_vs_prior 1.000; K4 wb_vs_bb 0.035; K4 wb_vs_prior 1.000; K8 wb_vs_bb 1.000; K8 wb_vs_prior 0.438

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.602 | 0.610 | 0.677 | 0.644 |
| wb | 0.602 | 0.575 | 0.644 | 0.644 |
| hybrid | 0.602 | 0.575 | 0.644 | 0.644 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.586 | 0.601 | 0.652 | 0.627 |
| wb | 0.629 | 0.573 | 0.634 | 0.632 |
| hybrid | 0.618 | 0.556 | 0.622 | 0.623 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.596 | 0.609 | 0.631 | 0.599 |
| wb | 0.612 | 0.556 | 0.640 | 0.641 |
| hybrid | 0.623 | 0.564 | 0.591 | 0.600 |
| prior | 0.658 | 0.610 | 0.677 | 0.644 |

label-efficiency curve at K8 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.611 | 0.593 | 0.637 | 0.610 |
| wb | 0.602 | 0.566 | 0.625 | 0.683 |
| hybrid | 0.582 | 0.604 | 0.637 | 0.626 |
| prior | 0.637 | 0.609 | 0.643 | 0.683 |

