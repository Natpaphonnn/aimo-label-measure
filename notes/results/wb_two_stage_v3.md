# Two-stage white-box analysis `v3` (analysis/run_wb.py)

Samples: 3912 over {'Qwen/Qwen3.5-4B': 169, 'Skywork/Skywork-OR1-Math-7B': 169, 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 169} problems per model. Labels: data/labels/case_labels.csv.
Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.

## target = correct

stage-1 target: **correct**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 1048 | 0.392 | 0.896 | 0.866 |
| Skywork/Skywork-OR1-Math-7B | 1064 | 0.386 | 0.867 | 0.949 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 936 | 0.186 | 0.844 | 0.759 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 158 | 78 | 0.70 | 0.696 | 0.696 (0.000) | 0.686 (0.015) | 0.686 (0.015) | -0.032, p=0.062, nan | -0.032, p=0.062, nan | -0.032, p=0.062, nan |
| K2 | 158 | 78 | 0.70 | 0.696 | 0.686 (0.003) | 0.677 (0.009) | 0.673 (0.008) | -0.013, p=0.688, nan | -0.025, p=0.125, nan | -0.019, p=0.453, nan |
| K4 | 158 | 78 | 0.70 | 0.696 | 0.673 (0.003) | 0.671 (0.023) | 0.679 (0.021) | -0.032, p=0.383, 0.09 | -0.057, p=0.012, 0.06 | -0.019, p=0.549, 0.06 |
| K8 | 139 | 75 | 0.71 | 0.712 | 0.693 (0.003) | 0.683 (0.000) | 0.691 (0.006) | -0.014, p=0.774, 0.07 | -0.029, p=0.125, nan | +0.000, p=1.000, nan |

Holm-adjusted p over the 8 primary comparisons: K1 wb_vs_bb 0.438; K1 wb_vs_prior 0.438; K2 wb_vs_bb 1.000; K2 wb_vs_prior 0.625; K4 wb_vs_bb 1.000; K4 wb_vs_prior 0.094; K8 wb_vs_bb 1.000; K8 wb_vs_prior 0.625

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.669 | 0.627 | 0.636 | 0.643 |
| wb | 0.654 | 0.623 | 0.640 | 0.643 |
| hybrid | 0.654 | 0.623 | 0.640 | 0.643 |
| prior | 0.669 | 0.627 | 0.636 | 0.643 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.649 | 0.616 | 0.610 | 0.635 |
| wb | 0.616 | 0.619 | 0.643 | 0.654 |
| hybrid | 0.629 | 0.588 | 0.632 | 0.631 |
| prior | 0.669 | 0.627 | 0.636 | 0.643 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.659 | 0.635 | 0.646 | 0.632 |
| wb | 0.649 | 0.616 | 0.638 | 0.637 |
| hybrid | 0.660 | 0.622 | 0.629 | 0.633 |
| prior | 0.669 | 0.627 | 0.636 | 0.643 |

label-efficiency curve at K8 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.620 | 0.661 | 0.733 | 0.727 |
| wb | 0.634 | 0.665 | 0.735 | 0.729 |
| hybrid | 0.589 | 0.659 | 0.727 | 0.706 |
| prior | 0.637 | 0.687 | 0.775 | 0.744 |

## target = consistent

stage-1 target: **consistent**

| model | gold samples | share correct | AUC probe | AUC twin (K=8) |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 1048 | 0.392 | 0.787 | 0.866 |
| Skywork/Skywork-OR1-Math-7B | 1064 | 0.386 | 0.864 | 0.949 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 936 | 0.186 | 0.567 | 0.759 |

| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 158 | 78 | 0.70 | 0.696 | 0.696 (0.000) | 0.686 (0.006) | 0.686 (0.006) | -0.006, p=1.000, nan | -0.006, p=1.000, nan | -0.006, p=1.000, nan |
| K2 | 158 | 78 | 0.70 | 0.696 | 0.686 (0.003) | 0.694 (0.003) | 0.675 (0.012) | +0.013, p=0.500, nan | +0.000, p=1.000, nan | -0.025, p=0.125, nan |
| K4 | 158 | 78 | 0.70 | 0.696 | 0.673 (0.003) | 0.690 (0.005) | 0.669 (0.003) | +0.025, p=0.424, 0.07 | +0.000, p=1.000, nan | +0.000, p=1.000, nan |
| K8 | 139 | 75 | 0.71 | 0.712 | 0.693 (0.003) | 0.688 (0.009) | 0.700 (0.003) | -0.007, p=1.000, 0.07 | -0.022, p=0.250, nan | +0.000, p=1.000, nan |

Holm-adjusted p over the 8 primary comparisons: K1 wb_vs_bb 1.000; K1 wb_vs_prior 1.000; K2 wb_vs_bb 1.000; K2 wb_vs_prior 1.000; K4 wb_vs_bb 1.000; K4 wb_vs_prior 1.000; K8 wb_vs_bb 1.000; K8 wb_vs_prior 1.000

label-efficiency curve at K1 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.669 | 0.627 | 0.636 | 0.643 |
| wb | 0.638 | 0.622 | 0.626 | 0.642 |
| hybrid | 0.638 | 0.622 | 0.626 | 0.642 |
| prior | 0.669 | 0.627 | 0.636 | 0.643 |

label-efficiency curve at K2 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.649 | 0.616 | 0.610 | 0.635 |
| wb | 0.631 | 0.622 | 0.621 | 0.644 |
| hybrid | 0.628 | 0.602 | 0.606 | 0.628 |
| prior | 0.669 | 0.627 | 0.636 | 0.643 |

label-efficiency curve at K4 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.659 | 0.635 | 0.646 | 0.632 |
| wb | 0.630 | 0.607 | 0.616 | 0.645 |
| hybrid | 0.633 | 0.612 | 0.617 | 0.634 |
| prior | 0.669 | 0.627 | 0.636 | 0.643 |

label-efficiency curve at K8 (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):

| arm | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|
| bb | 0.620 | 0.661 | 0.733 | 0.727 |
| wb | 0.613 | 0.642 | 0.749 | 0.730 |
| hybrid | 0.601 | 0.639 | 0.737 | 0.715 |
| prior | 0.637 | 0.687 | 0.775 | 0.744 |

