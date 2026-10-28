# Stage-1 diagnostics and label decomposition `v1all3` (analysis/wb_diagnostics.py)

## A. Probe AUC by block (correct vs incorrect sample; fixed PCA 32, C 0.1; cross-fitted by problem)

| model | S:trace:0 | S:answer:0 | S:trace:1 | S:answer:1 | S:trace:2 | S:answer:2 | selected (inner CV) | twin K=8 |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.870 | 0.675 | 0.897 | 0.697 | 0.877 | 0.729 | 0.897 | 0.875 |
| Skywork/Skywork-OR1-Math-7B | 0.867 | 0.743 | 0.910 | 0.783 | 0.867 | 0.793 | 0.898 | 0.952 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.748 | 0.603 | 0.787 | 0.636 | 0.767 | 0.667 | 0.725 | 0.690 |

Block names: S:<kind>:<j>, kind trace = mean of the last 64 generated tokens, answer = state of the last token; j = 0,1,2 -> layer L/4, L/2, 3L/4.

## B. Within-problem vs between-problem signal

| model | within-problem AUC probe | within-problem AUC twin | between-problem Spearman (problem mean vs solve rate) probe | twin |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.676 | 0.768 | 0.803 | 0.579 |
| Skywork/Skywork-OR1-Math-7B | 0.782 | 0.809 | 0.673 | 0.613 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.562 | 0.702 | 0.350 | 0.034 |

## C. Truncation control (samples that hit the cut get a forced answer)

| model | share finished before cut | acc finished / capped | AUC of 'finished' alone | probe AUC inside capped | inside finished |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.03 | 1.000 / 0.391 | 0.532 | 0.890 | n/a |
| Skywork/Skywork-OR1-Math-7B | 0.06 | 1.000 / 0.154 | 0.640 | 0.858 | n/a |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.01 | 1.000 / 0.114 | 0.548 | 0.741 | n/a |

## D. Transfer across problem sets (probe on the best block, trained on one set, tested on the other)

| model | block | AMC -> test-time wave | test-time wave -> AMC |
|---|---|---|---|
| Qwen/Qwen3.5-4B | S:trace:1 | 0.856 | 0.850 |
| Skywork/Skywork-OR1-Math-7B | S:trace:1 | n/a | n/a |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | S:trace:1 | n/a | n/a |

## E. What explains the robustness label? (80 labelled cases with 8 samples; robust = positive class)

| predictor | AUC Qwen3.5-4B | AUC Skywork-OR1-Math-7B | AUC DeepSeek-R1-0528-Qwen3-8B | pooled AUC | n | CV acc (+model offset) | prior on same cases |
|---|---|---|---|---|---|---|---|
| solve rate at cut (gold, 8 samples) | 0.885 | 0.568 | 0.806 | 0.690 | 58 | 0.632 | 0.655 |
| probe mean (stage 1) | 0.846 | 0.561 | 0.579 | 0.633 | 80 | 0.662 | 0.675 |
| twin mean (agreement) | 0.827 | 0.566 | 0.664 | 0.629 | 80 | 0.646 | 0.675 |
| labellers' base accuracy at 8k (byeolki) | 0.692 | 0.447 | 0.684 | 0.590 | 58 | 0.592 | 0.655 |

Solve rate at our cut vs the labellers' base accuracy at 8k on the same 58 cases: Spearman 0.74, means 0.55 vs 0.88.
