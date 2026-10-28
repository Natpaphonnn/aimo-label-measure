# Stage-1 white-box probe, Qwen3.5-4B, all AMC samples (2026-10-22 18:50)

Data: 464 samples (60 AMC problems x 7-8; our Kaggle HF traces cut at the test-time length 3072), forced answers at 3072
(`force_at_cuts_hf.py`), teacher-forced states on the Mac (MPS bf16; 120 unpadded + 344 with the padded last chunk, equal up to
bf16 noise). Correct at the cut: 60%. Problem solve rates: 15 never, 24 always, 21 mixed.
Probe: standardise -> PCA -> logistic on one sample's state, block/PCA/C by inner grouped CV, cross-fitted by problem
(5 folds, scores averaged over 3 fold seeds). Code: `analysis/wb_two_stage.py` (crossfit_probe, twin_scores).

| scorer of sample i's correctness | samples used | AUC |
|---|---|---|
| **white-box probe (sample i's own state)** | **1** | **0.853-0.868** |
| black-box twin: share of the other samples agreeing with i | 2 | 0.734 |
| black-box twin | 4 | 0.902 |
| black-box twin | 8 | 0.908 |
| probe + twin (rank average) | 2 | 0.911 |
| probe + twin (rank average) | 4 | **0.945** |
| probe + twin (rank average) | 8 | 0.938 |

Within problem (21 mixed problems, 210 correct-vs-wrong pairs): the probe ranks the correct sample higher in 61.4% of pairs,
the K = 8 twin in 77.6%.

Reading (exploratory; stage 2 / robustness labels not yet evaluated, only 22 labelled Qwen cases on these problems):
1. **Efficiency**: one sample's internal state predicts its correctness better than two-sample agreement (0.86 vs 0.73) and
   approaches 4-8-sample agreement (0.90-0.91).
2. **Complementarity**: probe + agreement beats either alone at every K (0.94-0.95 vs 0.91): the white-box signal adds
   information agreement does not have.
3. **What the probe reads**: mostly problem-level (solve rate), with a real but smaller sample-level component (61% within-problem
   vs 78% for agreement, which compares the actual answers).
