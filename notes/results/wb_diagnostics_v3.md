# Stage-1 diagnostics and label decomposition `v3` (analysis/wb_diagnostics.py)

## A. Probe AUC by block (correct vs incorrect sample; fixed PCA 32, C 0.1; cross-fitted by problem)

| model | S:trace:0 | S:answer:0 | S:trace:1 | S:answer:1 | S:trace:2 | S:answer:2 | selected (inner CV) | twin K=8 |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.870 | 0.658 | 0.895 | 0.699 | 0.874 | 0.726 | 0.896 | 0.866 |
| Skywork/Skywork-OR1-Math-7B | 0.892 | 0.783 | 0.881 | 0.808 | 0.879 | 0.821 | 0.867 | 0.949 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.797 | 0.708 | 0.861 | 0.745 | 0.813 | 0.708 | 0.844 | 0.759 |

Block names: S:<kind>:<j>, kind trace = mean of the last 64 generated tokens, answer = state of the last token; j = 0,1,2 -> layer L/4, L/2, 3L/4.

## B. Within-problem vs between-problem signal

| model | within-problem AUC probe | within-problem AUC twin | between-problem Spearman (problem mean vs solve rate) probe | twin |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.684 | 0.776 | 0.806 | 0.570 |
| Skywork/Skywork-OR1-Math-7B | 0.668 | 0.816 | 0.755 | 0.778 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.588 | 0.830 | 0.644 | 0.258 |

## C. Truncation control (samples that hit the cut get a forced answer)

| model | share finished before cut | acc finished / capped | AUC of 'finished' alone | probe AUC inside capped | inside finished |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 0.02 | 1.000 / 0.377 | 0.530 | 0.889 | n/a |
| Skywork/Skywork-OR1-Math-7B | 0.14 | 1.000 / 0.289 | 0.676 | 0.794 | n/a |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 0.04 | 1.000 / 0.155 | 0.598 | 0.806 | n/a |

## D. Transfer across problem sets (probe on the best block, trained on one set, tested on the other)

| model | block | AMC -> test-time wave | test-time wave -> AMC |
|---|---|---|---|
| Qwen/Qwen3.5-4B | S:trace:1 | 0.868 | 0.856 |
| Skywork/Skywork-OR1-Math-7B | S:trace:0 | 0.889 | 0.865 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | S:trace:1 | 0.837 | 0.820 |

## E. What explains the robustness label? (139 labelled cases with 8 samples; robust = positive class)

| predictor | AUC Qwen3.5-4B | AUC Skywork-OR1-Math-7B | AUC DeepSeek-R1-0528-Qwen3-8B | pooled AUC | n | CV acc (+model offset) | prior on same cases |
|---|---|---|---|---|---|---|---|
| solve rate at cut (gold, 8 samples) | 0.603 | 0.643 | 0.806 | 0.651 | 105 | 0.733 | 0.714 |
| probe mean (stage 1) | 0.617 | 0.548 | 0.664 | 0.624 | 139 | 0.686 | 0.712 |
| twin mean (agreement) | 0.634 | 0.644 | 0.664 | 0.638 | 139 | 0.698 | 0.712 |
| labellers' base accuracy at 8k (byeolki) | 0.609 | 0.543 | 0.684 | 0.608 | 105 | 0.708 | 0.714 |

Solve rate at our cut vs the labellers' base accuracy at 8k on the same 105 cases: Spearman 0.68, means 0.57 vs 0.90.
