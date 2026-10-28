# N2: hidden-state prompt probes on DeepSeek's official MATH labels (137 cases, 36 robust)

Features: final prompt token and problem-token mean of every layer (37 x 4096, fp16, one forward pass; `src/features/prefill.py`).
Probe: standardize -> PCA(32) -> L2 logistic (C=0.1), stratified 5-fold x 3 seeds.

| setting | accuracy | AUC |
|---|---|---|
| prior (always non-robust) | 0.737 | 0.50 |
| length / compressibility only | 0.706 | 0.716 |
| per-layer sweep, last token (best layer picked on the test folds: optimistic) | 0.65-0.75 (best 0.752 at L12) | |
| per-layer sweep, mean-pooled (optimistic) | 0.69-0.79 (best 0.788 at L34; already 0.77 at L1) | |
| mean-pooled probe, layer chosen in inner folds (honest) | 0.747 +- 0.014 | 0.756 |
| probe + length, nested | 0.735 +- 0.009 | 0.751 |

Paired tests (exact McNemar; MDE80 = smallest true accuracy gain detectable with 80% power at the observed discordance, `analysis/eval_features.py::mde_paired`):

| comparison | delta | 95% CI | wins/losses | p | MDE80 |
|---|---|---|---|---|---|
| nested probe vs prior | +0.015 | [-0.051, 0.080] | 11/9 | 0.82 | 0.095 |
| probe + length vs length only | +0.022 | [-0.036, 0.073] | 9/6 | 0.61 | 0.083 |

So: **no accuracy gain larger than about 9 points (vs prior) or 8 points (vs length) was detected**; smaller gains cannot be ruled out
at N = 137. Layers chosen across the 15 outer folds: 4 to 32, no stable band.

Reading (phrased within the detectable range): the per-layer sweep looks like a 5-point gain only because the layer is selected on the evaluation; with nested selection the
probe ranks somewhat (AUC 0.76) but does not beat the majority rule in accuracy, and it adds nothing measurable beyond problem length.
A mean-pooled embedding-layer probe already reaching 0.77 points to topic/lexical content (which MATH problems get domain-transferred
badly), not to a reasoning-state signal. Consistent with byeolki's finding that prompt probes track solve rate, not robustness; here solve
rate is saturated (base accuracy 1.0 on 138/141 rows), so there is nothing for them to read. Code: `analysis/p1_probe_vs_length.py`,
`analysis/train_solve_probe.py`; raw numbers in `probe_vs_length_math_deepseek.json`, `probe_math_deepseek_layers.json`.
