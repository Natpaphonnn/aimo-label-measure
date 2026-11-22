# Pre-registration of the revision analyses for Submission #21 (written 2026-11-22, before any of them was run)

Conditions: notes/REVIEWS_SUB21_2026-11-21.md (M1-M3, R-S1..S7, R-Q1..Q4). Rule: every result below is reported whatever its
sign or size; numbers in the paper that depend on a re-run are replaced by the re-run values; old values appear only in
notes/REVISION_NOTE.md as "was -> now". Data: the same cached generations and residual states as the submitted report
(runs/pilot, runs/gen, runs/features, runs/kaggle; data/labels/case_labels.csv). No new generations, no GPU.

## P1. Nested two-stage evaluation (R-Q1, M2) -> replaces Table 1 and Table 3 (Appendix B)
- Outer loop = the stage-2 folds as before: grouped 5-fold CV by problem_id over the labelled cases, seeds 0, 1, 2
  (`folds_for(cases.problem_id, 5, seed)`), K = 1, 2, 4, 8, same case sets (158 cases; 139 at K = 8).
- For each outer fold and each model, stage 1 sees only samples of problems that are NOT in the outer test fold (unlabelled
  problems are never outer-test, so they stay in every pool):
  - outer-training cases get stage-1 scores by inner grouped 5-fold cross-fitting inside that pool (same model selection:
    inner 3-fold CV over block x PCA {8, 32} x C {0.01, 0.1} by log-loss);
  - outer-test cases get scores from a probe selected and fitted on the whole pool.
  Stage-1 seed = stage-2 seed, as in the original run.
- Stage 2 is then fitted on the outer-training cases only and scored on the outer-test cases. Arms: prior (per-model majority
  of the outer-training cases), offsets (per-model offsets only), bb, wb, hybrid. Two variants as before: unweighted stage 2
  scored by plain accuracy, class-balanced stage 2 scored by balanced accuracy.
- Reported per arm and K (mean over the 3 seeds): plain accuracy, balanced accuracy, recall of robust, recall of non-robust,
  share of cases predicted robust.
- Table 3 (K = 4, balanced): wb - prior, wb - offsets, bb - prior, wb - bb; paired cluster bootstrap over problems, 10^4
  resamples per seed, 95% CI and two-sided p. Primary family as pre-registered in October (plain accuracy, exact McNemar,
  Holm over wb vs bb and wb vs prior at K = 1, 2, 4, 8) is recomputed on the nested predictions.
- Headline white-box result (M1, R-S3) = wb - offsets at K = 4 (balanced), whatever its value; the abstract states the
  measured wb - bb gap.

## P2. What the probe adds beyond base solvability (R-Q2)
- Stage 2 (same model, matched on solve rate): cases with all 8 samples gold-labelled; strata = (model, number of correct
  samples out of 8 at the cut). Primary: pooled within-stratum AUC of the nested wb case mean (K = 8) for the label
  (robust = positive), i.e. only pairs of cases of the same model with the same solve count are compared; 95% cluster bootstrap
  over problems (10^4). Secondary: nested class-balanced stage 2 with [gold solve rate at the cut] + offsets vs
  [solve rate + wb mean, sd] + offsets (K = 8 cases with gold), paired bootstrap of the difference.
  Decision rule: if the within-stratum AUC CI includes 0.5 and adding wb does not significantly improve on the solve-rate
  arm, the paper's claim becomes "stage 1 estimates base solvability; we find no evidence that it adds anything beyond it".
- Stage 1 (natural vs truncated traces): finished = the sample ended before the cut (n_gen < cut, its 64-token window can
  contain natural final-answer tokens); truncated = reached the cut (window is inside the reasoning). Probes are selected,
  fitted and evaluated separately inside each subset (grouped cross-fitting by problem, same grid); AUC per model and
  subset. Variant for stage 2: a probe trained on truncated samples only scores all samples; nested K = 4 balanced
  wb - offsets and wb - prior are reported next to P1 (not substituted).

## P3. Two-sided solve-rate rules (R-S2) and official labels (R-S1)
- For each solve-rate-type case predictor x in [0, 1] (gold solve rate at the cut, wb mean, agreement mean, labellers' base
  accuracy at 8k where available): (a) monotone AUC of x; (b) AUC of the band score |x - 0.5|; (c) a two-threshold rule
  "non-robust iff a < x < b" with (a, b) on a 0.05 grid chosen by balanced accuracy on the training folds, nested grouped
  5-fold x 3 seeds, balanced accuracy on the test folds; (d) class-balanced logistic on [x, x^2] + offsets, same CV.
  95% cluster bootstrap CIs. Wording rule fixed now: "close to chance" is kept only if the best of (a)-(d) has balanced
  accuracy below 0.60 and its CI includes 0.5 (AUC: below 0.65 with CI including 0.5); otherwise the sentence reports the
  measured value and drops "close to chance".
- R-S1: the same table on the official train-main-v2 cases of the three probe models (34 cases); the labellers' base
  accuracy row is not available there (train-main-v2 publishes is_robust and max_drop only), so the base-accuracy claim is
  restricted to byeolki-protocol labels. Small n is stated with its CI.

## P4. Simulation assumptions and sensitivity (M3, R-S5, R-Q3)
- New script analysis/label_sensitivity.py. Stated assumptions: Bernoulli samples per prompt; families independent given
  the case; per-family degradation probability = scale x observed share of rows with drop >= 0.25 in sample-full
  (10 problems x 8 settings); drop size ~ U(0.3, 1.0); base accuracy from the empirical sample-full distribution (base > 0).
- Sensitivity grid: scale {0.25, 0.5, 0.75, 1.0}; family shares bootstrapped over the 10 sample-full problems (200
  replicates, 5-95% range); drop size U(0.1, 1), U(0.3, 1), U(0.5, 1); correlated families (Gaussian copula, rho = 0.5).
  Outputs: false-NR at p0 = 0.9, oracle agreement, test-retest, solve-rate-only gain, kept share.
- Selection bias: share of cases dropped by the 0.10-0.25 band, and how the kept set differs from all cases (truly degraded
  share, mean p0, robust share). Construct validity: P(robust) for low-p0 cases (a model that always fails is "robust").
- The 0.88-0.94 numbers are re-described as simulated-oracle agreement under these assumptions; the 40-sample
  recommendation is qualified (it reduces sampling variance only).

## P5. Precision parity (R-S6), from cached states
- 24 Qwen samples recomputed in bf16 (Mac) and fp16 (Kaggle T4): cosine per block and the change of the selected probe's
  score; probe AUC on the bf16-only Qwen samples (714) vs the fp16 ones. The evaluation GPUs (compute capability >= 8) run in
  bf16 (src/features/budgeted.py), as the Mac does.
