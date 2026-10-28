# White-box stage-1 pilot, Qwen3.5-4B (2026-10-22, Mac MPS bf16)

Data: our Kaggle AMC traces (HF fp16, 8k cap), samples 4 and 5 of all 60 problems = 120 samples, cut to the test-time length 3072
(the first 3072 tokens of a capped sample are a sample with cap 3072); forced answer at 3072 (`force_at_cuts_hf.py`, greedy, 16
tokens) unless the sample finished earlier; states by teacher forcing (`trace_states.py`, layers 7/15/23, window 64).
Correct at the cut: 61% (73/120); answered 98%. Problems: 34 all-correct, 5 mixed, 21 all-wrong.

Stage-1 correctness probe (StandardScaler -> PCA -> logistic, grouped 5-fold by problem; ~96 training samples per fold):

| feature | AUC for sample correctness (3 fold seeds) |
|---|---|
| probe, nested selection of block/PCA/C | 0.786 / 0.764 / 0.767 |
| fixed block trace-mean, layer 7 (L/4) | 0.763 |
| fixed block trace-mean, layer 15 (L/2) | **0.788** |
| fixed block trace-mean, layer 23 (3L/4) | 0.739 |
| answer-state (emitting token), layers 7/15/23 | 0.548 / 0.554 / 0.562 |
| black-box twin at K = 2 (does the other sample agree?) | 0.797 |

Reading: one sample's mid-layer trace state predicts that sample's correctness on NEW problems about as well as agreement between
TWO samples, from only ~96 training samples. The last-token state is near chance: the signal is in the trace, not at the
final token. Within-problem discrimination cannot be judged here (5 mixed problems; the probe ranked the correct sample higher in
3-4 of 5); it needs the full 8 samples per problem (Kaggle states kernel). Exploratory: block choice is not pre-registered beyond
#33/#44 (nested CV picks it inside the folds).
