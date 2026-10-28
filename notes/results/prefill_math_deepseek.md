# Prefill-only features vs official MATH labels (DeepSeek-R1-0528-Qwen3-8B, 137 cases, 36 robust, old any-degrade rule)

Features from one forward pass of the chat prompt (`src/features/prefill.py`, Kaggle T4, bf16->fp16). Labels: `augmented-sample-math-agg`
(alias qwen3-8b:low = DeepSeek-R1-0528-Qwen3-8B; base accuracy 1.0 for 138/141 rows, so these labels are pure degradation labels, driven by
domain transfer and distractors).

| feature | univariate AUC | direction |
|---|---|---|
| problem length (chars / tokens) | 0.74 / 0.72 | longer = more robust |
| zlib compressibility | 0.74 | less compressible = more robust |
| first-generated-token entropy | 0.72 | higher entropy = more robust |
| mean log-prob / perplexity of the problem text | 0.59 | weak |
| Min-K% 10/20/40, Min-K%++ | 0.50-0.55 | none |

Grouped-CV logistic regression on all scalars: accuracy 0.723 vs 0.737 for the per-model prior (always non-robust); paired delta -0.015,
95% CI [-0.044, 0.015], McNemar p = 0.63, wins/losses 1/3. The classifier disagrees with the prior on only 4 of 137 cases, so no
difference is detectable at 80% power (MDE80 undefined: even winning every discordant case would be a 2.9-point gain). The CI bounds
any accuracy gain of the scalar classifier over the prior at **under 1.5 points**; the AUCs below are the informative statistic here. Grouped-CV out-of-fold AUC by feature group: length only 0.718; memorization only 0.624; first-token entropy only 0.622; length+memorization 0.696; length+entropy 0.721; all 0.699. Entropy correlates with length (r = 0.57); first-token top-1 probability and top-5 mass are constant for this model (the token after `<think>` is deterministic).

Reading: within what N = 137 can resolve, memorization/fluency statistics of the problem text do NOT separate robust from non-robust cases for this model on MATH. The only
prefill signal is problem length/compressibility, a problem-level attribute whose sign flips between MATH (longer = robust) and the official
train set (longer = non-robust); on validation/test sets balanced per problem such attributes cannot help. This is negative result N1 for the report.
Caveat: MATH labels use the stricter any-degrade rule and a saturated base accuracy; the AIMO-level mechanism (solve rate) is absent here.
