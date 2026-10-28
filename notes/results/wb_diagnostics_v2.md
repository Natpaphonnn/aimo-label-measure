# Stage-1 diagnostics and label decomposition `v2` (analysis/wb_diagnostics.py)

## A. Probe AUC by block (correct vs incorrect sample; fixed PCA 32, C 0.1; cross-fitted by problem)

| model | S:trace:0 | S:answer:0 | S:trace:1 | S:answer:1 | S:trace:2 | S:answer:2 | selected (inner CV) | twin K=8 |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.870 | 0.658 | 0.895 | 0.699 | 0.874 | 0.726 | 0.896 | 0.866 |
| Skywork/Skywork-OR1-Math-7B | 0.867 | 0.743 | 0.910 | 0.783 | 0.867 | 0.793 | 0.898 | 0.952 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.748 | 0.603 | 0.787 | 0.636 | 0.767 | 0.667 | 0.725 | 0.690 |

Block names: S:<kind>:<j>, kind trace = mean of the last 64 generated tokens, answer = state of the last token; j = 0,1,2 -> layer L/4, L/2, 3L/4.

## B. Within-problem vs between-problem signal

| model | within-problem AUC probe | within-problem AUC twin | between-problem Spearman (problem mean vs solve rate) probe | twin |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.684 | 0.776 | 0.806 | 0.570 |
| Skywork/Skywork-OR1-Math-7B | 0.782 | 0.809 | 0.673 | 0.613 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.562 | 0.702 | 0.350 | 0.034 |

## C. Truncation control (samples that hit the cut get a forced answer)

| model | share finished before cut | acc finished / capped | AUC of 'finished' alone | probe AUC inside capped | inside finished |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.02 | 1.000 / 0.377 | 0.530 | 0.889 | n/a |
| Skywork/Skywork-OR1-Math-7B | 0.06 | 1.000 / 0.154 | 0.640 | 0.858 | n/a |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.01 | 1.000 / 0.114 | 0.548 | 0.741 | n/a |

## D. Transfer across problem sets (probe on the best block, trained on one set, tested on the other)

| model | block | AMC -> test-time wave | test-time wave -> AMC |
|---|---|---|---|
| Qwen/Qwen3.5-4B | S:trace:1 | 0.868 | 0.856 |
| Skywork/Skywork-OR1-Math-7B | S:trace:1 | n/a | n/a |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | S:trace:1 | n/a | n/a |

## E. What explains the robustness label? (121 labelled cases with 8 samples; robust = positive class)

| predictor | AUC Qwen3.5-4B | AUC Skywork-OR1-Math-7B | AUC DeepSeek-R1-0528-Qwen3-8B | pooled AUC | n | CV acc (+model offset) | prior on same cases |
|---|---|---|---|---|---|---|---|
| solve rate at cut (gold, 8 samples) | 0.603 | 0.568 | 0.806 | 0.629 | 87 | 0.640 | 0.655 |
| probe mean (stage 1) | 0.617 | 0.561 | 0.579 | 0.601 | 121 | 0.664 | 0.669 |
| twin mean (agreement) | 0.634 | 0.566 | 0.664 | 0.612 | 121 | 0.634 | 0.669 |
| labellers' base accuracy at 8k (byeolki) | 0.609 | 0.447 | 0.684 | 0.582 | 87 | 0.632 | 0.655 |

Solve rate at our cut vs the labellers' base accuracy at 8k on the same 87 cases: Spearman 0.67, means 0.54 vs 0.89.
