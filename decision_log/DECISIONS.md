# Decisions log

## 2026-10-15 (P0)

1. **Interface**: build against `are_robust(model_id, reasoning_effort=..., problems=...)`; treat `problems` as original statements; accept any effort string; return a native list of bools; wrap every batch in try/except with a prior fallback.
2. **[Updated by #18]** **Authoritative label rule for self-labelling**: max over perturbation types of (base accuracy - type accuracy); robust iff <= 0.10, non-robust iff >= 0.25, discard the band. Pending the Discord answer on never-solved cases, self-labelling will **exclude cases with base accuracy < 0.5** and report sensitivity to that cut (0.3 / 0.5 / 0.7) and to the byeolki "never-solved = robust" convention.
3. **Track focus**: one method, submitted to both tracks every time. Scientific effort goes to the Small track (the four models we can self-label on free GPUs). Main track reuses the same model-agnostic features computed on the worker for gpt-oss-120b, with per-effort priors from the 45 labelled gpt-oss rows in train-main-v2 and the 10 reconstructed gpt-oss:low cases; no gpt-oss training data will be generated locally.
4. **Validation protocol (pre-registered)**: grouped CV by problem; report pooled, within-model, and difficulty-controlled (solve-rate covariate) accuracy; a feature counts only if it beats the per-(model, effort) prior under a paired test. Never tune on the Codabench board (14/21 cases, probe-able).
5. **First uploads**: `dist/prior-only-v1-main.zip` and `dist/prior-only-v1-small.zip` (constant True per config given current priors) to validate the pipeline on both tasks. Expected scores 0.571 / 0.500.
6. **Leakage hygiene**: exclude AIME 2024-2026 and all `aimo-2025-reference` problems from self-labelling pools (organizer advice 2026-10-09; val problems are public). Never redistribute private data; publish only our own generated variants, labels and metrics.
7. **Repo**: MIT, external clones not vendored (`tools/clone_external.sh`), data re-fetched by script, deterministic zips via `tools/build_submission.py`. No commits unless the author asks.

## 2026-10-15 (P1 start)

8. **Pivot from "label everything" to "solve rate first"** (justified by `notes/results/label_noise_ceiling.md`): with 10 samples per prompt, a case whose true solve probability is 0.9 and that has NO true degradation is still labelled non-robust 17% of the time and dropped 30%; robust therefore means "solves the original with probability near 1 and no family truly degrades". The primary training target is the per-config solve rate on ORIGINAL problems (10 samples each, realistic token budget); the robust/non-robust mapping is calibrated on the 71 official labels plus reconstructed sample-full cases. Variants (no-solution first, then rephrase/domain) are generated only for solved problems in a second stage to learn the residual.
9. **Token budget**: byeolki's 8k cap truncated 92-95% of Qwen3.5-4B and DeepSeek samples, so their "unsolved" counts are budget artifacts. We use a 24k-32k cap with forced answers for the thinking models and record both forced and unforced correctness.
10. **Generation backends**: Qwen3.5-4B on the Mac via mlx-lm (bf16, batched) overnight on AC power only; 7-8B models on Kaggle (vLLM if it installs on the offered GPU, else HF generate); gpt-oss-120b never generated locally.
11. **Test-time design hypothesis**: estimate each config's solve rate on the original (prefill probe + 2 short budgeted samples), and, because every config is asked about the same problems, cache cross-config features so that within-problem ranking (balanced labels) can be exploited. To be validated in P2.

## 2026-10-16 (the author's direction after N2)

12. **Stop work on prompt-only probes.** N1 and N2 are kept for the report, with MDE80 next to every p-value and phrased as "no effect larger than X detected".
13. **Milestone M1, due 2026-10-20**: a test-time feature that beats the per-(model, effort) prior WITHIN model under grouped CV (paired test, MDE reported), on competition-level problems, plus a runtime estimate showing it fits the 3600 s server budget for both tracks. Route: per-(model, problem) solve rate on the original (short budgeted samples at test time) and the expert no-solution family.
14. M1 evidence ladder: (a) GPU-free simulation on byeolki's competition-level generations (features from 2 of 8 original samples, labels from the disjoint 6 + variants); (b) the same on our Mac Qwen3.5-4B generations at a larger budget, adding no-solution variants (60 authored AMC problems); (c) runtime from measured server-class throughput (BAS logs on the RTX PRO 6000) and our own Codabench run.

## 2026-10-16 (after the prior-only uploads)

15. **[WITHDRAWN 2026-10-16, see #17]** **Calibration target is a balanced set, not train-main-v2's 73% robust.** Both validation sets are about half robust (Main 10/21, Small 7/14), consistent with the organizers' per-problem and per-model balancing. Decisions therefore use within-config ranking (predict robust for the top half of each batch by score, as BAS does) or a threshold fitted for a 50% positive rate, never the train prior. The prior-only bundle is effectively always-True and stays only as a pipeline check.
16. **New ingestion program (2026-10-15)** is unknown and broke a quarter of Main runs. Until we know what changed, every model-loading bundle gets a cheap Main-track canary upload before any real candidate, and must keep per-batch try/except fallbacks.

## 2026-10-16 (organizer answers on Discord, FACTS section 0)

17. **Per-case absolute decisions.** The server calls `are_robust` once per case and the test set is not balanced (D1, D2), so
    #15 is withdrawn: no ranking within a call, no forced positive rate. Each case gets `score >= threshold`, with the score
    calibrated and the threshold fitted per config (or shared with per-model offsets) in grouped CV on self-labels plus official
    train rows. Bundles keep module-level caches but must be correct when every call is a fresh process; both modes are tested
    by `tests/test_per_case_protocol.py` through `tools/emulate_per_case.py`.
18. **Self-labelling convention (D3, D9).** Primary: band 0.10/0.25 over the answer-preserving families (no expert_no_solution,
    matching train-main-v2), excluding cases with base accuracy < 0.5. Sensitivity: base cut 0.3 / 0.7; a stricter band
    (robust <= 0.05, non-robust >= 0.30); labels that include expert_no_solution. All reported.
19. **Evaluation fixes (analysis/eval_features.py).** (a) Primary baseline is now the in-sample per-model majority: the
    leave-one-problem-out majority flips when a model's robust share is near 0.5 and then scores below the base rate (K = 3,
    cut 0.3: 0.515 against a 0.62 robust share), inflating gains. It is slightly optimistic for the baseline, so comparisons
    against it are conservative. (b) Seeds now change the folds: whole problems are assigned to folds at random per seed. Before,
    sklearn's GroupKFold ignored the shuffled row order and every seed gave the same folds, so the reported seed means were
    single runs. M1 numbers before 2026-10-16 are superseded by `notes/results/m1_agreement_signal.md` (v2). N1 and N2 have
    robust shares far from 0.5 (26%), so (a) barely affects them; they will be recomputed with (b) before the report.
20. **[Numbers superseded by #26]** **Runtime design under the per-case limit.** About 150 s of compute per case (`AIMO_PER_CASE_S`; 200 s/case on val Main).
    From the BAS server logs, HF `generate` decodes at about 31 steps/s for Qwen3.5-4B and 9 / 9 / 5.5 / 7.5 steps/s for
    Skywork / DeepSeek / Olmo / gpt-oss at batch 18-36, i.e. roughly 4.5k tokens per sample for Qwen and 0.8-1.4k for the
    7-8B models in 150 s, whatever K is. Width is nearly free, depth is not: the test-time feature is K parallel short samples
    with a forced answer at the cut. Whether agreement still predicts at 0.5-2k tokens is the deciding experiment
    (force_at_cuts on night-1, cuts 512 / 1024 / 2048 / 4096). The real steps/s at batch 4-16 is to be measured with a timing
    canary upload that logs only timing (no problem text) before any real candidate.
21. **M1 confirmatory specification, pre-registered now (before our own labels exist).** Pooled logistic regression on the
    label-free agreement features (agreement, distinct answers, null answers, truncation, log length) plus model one-hot,
    grouped 5-fold CV by problem x 3 seeds, K = the largest value that fits the measured runtime (expected 4), label convention
    #18 primary, compared with the in-sample per-model prior by exact McNemar; MDE80 reported. The byeolki simulation that
    motivated this is exploratory (16 specifications were looked at).
22. **Nomination (D6).** The P3 rule (written by Oct 27) chooses 2-3 bundles per track that differ in mechanism (e.g. the
    best agreement model, the same with the no-solution component, a conservative prior-plus-agreement variant), not the
    single best val score: the best private score counts.
23. **No-solution component (D4, D9).** Optional and gated: validated on val-sample and our no-solution variants, it may only
    flip a case to non-robust when its own evidence is strong, and its effect is reported separately. A helper model
    (gpt-oss-120b writing test-time variants) is a P2 option only if the base route is done.
24. **No compute grant (D7).** The draft stays in notes/ but is not part of the plan.
25. **Priors.** `submissions/prior-only` stays frozen at the uploaded version (train-main-v2 23791d9b). New bundles take
    priors from the current revision via `tools/make_priors.py` (fa8af758: Skywork 0.75 instead of 0.67).

## 2026-10-16 afternoon (timing canary 991764, FACTS D10)

26. **[Budget part superseded by #29]** **Fresh process per case; budget and targets.** No in-memory state survives between cases, so every case loads its model
    (3.5-6.5 s) and nothing can be cached across configs in memory. `PER_CASE_S` = 160 s after import (191 s of wall clock per
    case available on val). Measured short-context decode is 50-71 steps/s for the small models and 25 for gpt-oss, so the M1
    target moves from ~1k to **2-4k tokens per sample with K = 4-8** (gpt-oss ~2-3k), pending the long-context measurement.
    Every generate call carries `max_time` so that one slow case cannot eat the shared job budget.
27. **Timing canary v2 before any real candidate** (one Main upload): long-context speed (to 4096 tokens, batch 4/8/16) and
    whether a temp file written by one case process is visible to the next. If it is, a file-based clock could bank unused
    time across cases and cache work across configs; until then each case must stand alone.
28. **M1 evaluation at realistic budgets.** Night-1 (Qwen3.5-4B, 8 samples x 8k) supports features from K <= 4 samples cut
    at 2048 / 4096 tokens, with labels from the other 4 samples at the full 8k budget (labels need long budgets: the organizers
    sample up to 100k tokens). K = 8 needs at least 12 samples per problem: a later run on a subset. The pre-registered
    specification of #21 stands, with the cut fixed by the canary v2 result.

## 2026-10-16 evening (#qa 15:30-15:44, FACTS D11)

29. **Global time keeper.** The cap is global and scales with unique problems, so a fixed per-case allot is wrong.
    `runtime.JobClock` shares the budget across the fresh case processes through a state file in the temp dir (keyed by the
    parent pid and bundle path, expired after 15 min idle, problem texts stored only as hashes): 240 s earned per new problem
    (val rate 262.5), overhead reserved for every possible future case, the rest shared over expected cases (c0 = 3 on Main,
    4 on Small), prior answer below 15 s. Simulated worst case 0.912 of the budget over 13 scenarios
    (`analysis/sim_jobclock.py`, unit-tested). Price: ~30% use of the val budget, and up to half prior answers on a Main test
    with 7 cases per problem ordered by config. Canary v2 now logs the clock's dry-run view on the server; if the temp dir does
    not persist there, bundles switch to the stateless fallback (AIMO_STATELESS, ~24 s per case, safe in any order).
    `CaseClock` remains only in the canaries.
30. **Generalisation first (D11).** Every candidate is evaluated on held-out problems (grouped by problem: our self-labelled
    pool, official train rows and val-sample as separate held-out sets); nominations follow the held-out evidence under the P3
    rule, never the val board. The report gets an explicit "val board vs held-out" section (our own board and held-out numbers,
    plus the organizers' report that the 100% Small entry fell to chance on 16 new problems).
31. **Variable-budget features.** The allot per case will range from ~25 s to ~200 s depending on the test's cases per problem,
    so the method picks K and the token cap from the allot at run time (via the measured steps/s), and the classifier is trained
    per budget level (night-1 cut points 512-4096 give these levels for Qwen3.5-4B). Agreement at the smallest budget that the
    test can force must be measured, not assumed.

## 2026-10-17 (White-Box Awards, FACTS D12; night-1 outcome)

32. **Hybrid is the main line.** Answer agreement alone is black-box and cannot win the White-Box award, so the primary
    candidate becomes agreement + an in-trace hidden-state probe captured during the SAME K-sample generation at no extra
    generation cost. Agreement-only stays as a separate nominated bundle (it may still be the best black-box method), and at
    least one white-box-eligible bundle is nominated per track.
33. **White-box feature specification (fixed before any self-label exists).** During generation, forward hooks on the decoder
    layers at depths L/4, L/2, 3L/4 record the residual stream (layer output) at every decode step. Per sample: (a) the mean
    over the last 64 generated tokens before the cut ("trace state"), (b) the state at the last position, i.e. the token that
    emits the forced or natural answer ("answer state"). Per case (K samples): the mean of (a) and of (b) across samples, and
    the dispersion of (a) across samples (mean pairwise cosine distance, K >= 2). Probe: PCA to 16 components fitted inside
    the training fold, then logistic regression; layer and C chosen by inner grouped CV, never on the outer fold (the N2
    lesson). Offline these states are recomputed by teacher-forcing the saved token ids through the same model
    (`src/features/trace_states.py`): identical to the generation-time states up to numerical noise, and it gives the states
    at every cut point (512-4096) from one forward pass. Prompt-only states are excluded (N2: they read the topic).
34. **Pre-registered ablation.** Four arms on the same held-out problems (grouped 5-fold by problem x 3 seeds, label
    convention #18, in-sample per-model prior as the floor): prior; agreement-only (#21 features); probe-only (#33 features);
    hybrid (both, one logistic model). Primary white-box claim: hybrid beats agreement-only on paired exact McNemar with the
    bootstrap CI and MDE80 reported; "adds signal" means p < 0.05 AND the CI excludes 0. Secondary: probe-only vs prior, and
    probe dispersion alone vs prior. If the primary claim fails, the white-box bundle is still nominated (eligibility needs the
    component to be used, not to win) but the report says plainly that no effect larger than the MDE was found. Token
    log-probs and entropy are reported as black-box features until the organizers rule otherwise.
35. **Night-1 (2026-10-16) outcome and fix.** 48 minutes before the Mac left AC power: 16 of 960 samples, 15 at the 8k cap,
    forced answer correct in 10 of 15, throughput falling from 95 to 45 tok/s (batch 8, bf16 MLX) as contexts grew. At 45
    tok/s the planned 7.9M tokens need ~49 h: infeasible. Fix to be measured before night-2: 4-bit MLX quantisation of
    Qwen3.5-4B (documented deviation; the per-config label convention is ours anyway), larger batch (16), and a 6144 cap with
    forced answers; the AMC subset (60 problems) first because its no-solution variants are already authored. The launcher now
    re-arms after a stop instead of exiting, so a lost AC connection costs only the gap.
36. **Budget for labels vs budget for features (2026-10-17 measurements).** 4-bit MLX Qwen3.5-4B at batch 16 gives 92 tok/s,
    the same aggregate rate as bf16 at batch 8 (95 falling to 45 at 8k contexts): decode on the M5 is compute-bound, so
    quantisation and larger batches do not raise throughput; the Mac yields ~2M tokens per night. Forced answers on the 16
    night-1 samples: correct 5/16 at 512, 5/16 at 1024, 6/16 at 2048, 6/16 at 4096 tokens vs 11/16 at 8192. A 4k cap would
    halve Qwen's measured solve rate and turn "slow solves" into exclusions or non-robust labels, so **labels keep the 8192
    cap** (byeolki's budget; a documented deviation from the organizers' 100k) and **test-time features use the cuts**. Recipe:
    60 AMC originals x 8 samples at 8k (~3.9M tokens, two nights, resumable), then the authored no-solution + rephrase/domain
    variants for solved problems. The own-label M1 evidence therefore arrives ~Oct 22, after the Oct 20 milestone; the Oct 20
    deliverable is the byeolki-based evidence plus the budget-cut analysis on our originals.
37. **Self-labelling moves to Kaggle (the author, 2026-10-17; cap raised to 25 GPU-h/week while Name That LLM is paused).** Qwen3.5-4B
    runs with HF generate in fp16 (the architecture is not in any vLLM release old enough to ship CUDA-12 wheels for T4; the
    current wheel is CUDA-13 and broke Kaggle's torch on 2026-10-15): one worker process per T4, batch 16, 8k cap, forced
    answers batched, `token_ids` saved for the white-box states. Two shards (samples 0-3 and 4-7 of the 60 AMC originals),
    7.5 h budget each so the kernel ends cleanly before Kaggle's 12 h limit (a killed kernel loses its output). The Mac takes
    sample indices 8-11 of the same problems tonight (12 samples per problem lowers label noise, DECISIONS #8), under
    `caffeinate -i`. A bounded smoke kernel tests pinned `vllm==0.8.5` on T4 for Skywork; if it runs, Skywork and DeepSeek
    use vLLM, Olmo-3 (newer architecture) falls back to HF. Target: own-label M1 + white-box ablation evidence by Oct 22-23.

### 38. Kaggle HF path: forced answers one prefix at a time; vLLM 0.10.2 needs transformers<5 (2026-10-17 16:40 local time)
**Evidence.** amc-a/b v2 logs: batch-8 sampling at 8k on a T4 finished (≈50 tok/s per GPU, 8x8k in ~21 min) but
`forced_answers` prefilled 8 truncated 8k traces at once and OOMed inside `torch_chunk_gated_delta_rule` (Qwen3.5 linear
attention without flash-linear-attention casts the whole prefix's q/k/v to fp32: the failed allocations were 0.8-1.06 GiB,
i.e. 8 x 8192 x 4096 x 4 B). Records were written only after forcing, so both chunks were lost (0.8 GPU-h).
vLLM smoke v2: 0.10.2 installs and the V0 engine initialises on sm75, but transformers 5.x removed
`all_special_tokens_extended` which its tokenizer wrapper calls.
**Decision.** (a) kernel template: forcing runs with `force_chunk`=1 prefixes per call, each wrapped in an OOM/any-exception
guard that yields `forced_answer=None` for that prefix; the forcing loop itself is wrapped so sampled records (with
`token_ids`) are always written; tok/s is logged right after sampling. (b) vLLM smoke v3 pins `vllm==0.10.2 "transformers<5"`.
(c) Shards a/b re-pushed as v3 (same jobs); shard b queued behind the ~10 min smoke so the vLLM decision for
Skywork/DeepSeek/Olmo is not delayed by 6 h. Budget check: ~3.5 GPU-h used this week, ≤19 planned, cap 25.
**Why it matters for the plan.** The HF path for Qwen3.5-4B is ≈50 tok/s per T4 at batch 8: 60 problems x 8 samples x 8k
≈ 3.9M tokens ≈ 11 GPU-h on two kernels of two GPUs ≈ 5.5 h wall each. Variants for solved problems need a similar budget,
so the Oct 22-23 target holds only if vLLM works for the three Qwen2/Qwen3-architecture models.

### 39. hybrid-v1 bundle: one code path, three arms chosen by the artifact (2026-10-17 17:00 local time)
`submissions/hybrid-v1/` samples K traces per case at the official settings (T=1, top_k 40, top_p 0.95, cut tokens) with
`hooks.ResidualRecorder` attached to layers L/4, L/2, 3L/4, then (outside the hooks) greedy forced answers for truncated samples.
Features = agreement statistics over the final answers (agree_top, agree_natural, n_distinct, share_answered, share_finished,
share_nan, mean_len_frac, forced_share, log_steps) + per-layer trace/answer mean states + dispersion. The artifact
(`analysis/fit_hybrid_artifact.py`) carries arm ∈ {hybrid, agreement, probe}, the chosen state block, PCA(16), scaler and logistic
weights; the bundle's manual logit reproduces sklearn to 1e-6 (fitter selftest). So the pre-registered ablation (#34) ships as
three bundles that differ ONLY in the artifact, which is exactly the with/without comparison the White-Box Awards ask for.
Safety: missing artifact / short allotment / any exception -> per-config prior; the cut shrinks with the allotment
(`steps_per_s` x 0.85) and `max_time` bounds generation; JobClock shared across the fresh processes. Verified: per-case protocol
on both tracks (prior path), and the full pipeline on Qwen3.5-4B (MPS, K=2, cut 48): hooks recorded 47 steps, forced answers
correct, 30 s including load. Offline feature definitions mirror the bundle (agreement from the first K samples at the cut via
force_at_cuts output; n_steps = min(cut, longest sample) - 1). Still to measure on the server: K=8 at cut 2048 within the
allotment (canary v2 numbers), gpt-oss behaviour with hooks (`model.layers` path exists; MXFP4 decode 24.6 steps/s).

### 40. Offline states + forced answers at cuts move to Kaggle too (2026-10-17 17:20 local time)
`kaggle/jobs/states_cuts_job.py` (packed with `tools/kaggle_pack.py --kernel-source <gen kernel>`; the gen kernels' outputs are
mounted under /kaggle/input/<slug>/ so nothing is uploaded by hand) runs `src/selflabel/force_at_cuts_hf.py` (torch port of the
MLX script, batch 1 because of the Qwen3.5 fp32 linear-attention prefill) and `src/features/trace_states.py` on a T4.
Check on 3 Mac records (MPS, cuts 512/1024): 5/6 forced answers identical to the MLX run; the one difference (10 vs 11 at
1024 tokens) is greedy-decode divergence between backends/dtypes, so forced answers at cuts are backend-dependent at the
margin and the report must say which backend produced the labels (Kaggle HF fp16 for Kaggle traces, MLX bf16 for Mac traces).
Kernel `kx-states-qwen35-amc` is rendered (not pushed) and will be pushed when shards a/b have finished.

### 41. Self-labelling backends per model on Kaggle (2026-10-17 17:10 local time)
Skywork-OR1-Math-7B and DeepSeek-R1-0528-Qwen3-8B: vLLM 0.10.2 (V0, eager, TP=2, fp16), 60 AMC x 8 samples at 8k in one
kernel each (~3-4 h expected at 265 tok/s). Qwen3.5-4B: HF per-GPU workers (shards a/b v3). Olmo-3-7B-Think: HF single worker
with device_map=auto over both T4s (`--single-worker`), or vLLM if a 0.10.2 load attempt succeeds (not expected); decided after the
first four kernels report. gpt-oss-120b: no self-labels (does not fit); features at test time only. Kernels are pushed by a
watcher as slots free (Kaggle allows 2 sessions); the states+cuts kernel waits until actual GPU hours are known so the 25 h/week
cap is respected (expected ~21 h after DeepSeek).

### 42. No shared state between cases -> static allotment; canary v3 (2026-10-17 evening)
**Evidence.** Canary v2 (995120): every case starts with an empty temp dir and ppid 1; the shared JobClock never sees a second
case (its dry-run allot was ~24 s). Decode curves at 4k tokens are in runtime_estimate v5 / FACTS D14.
**Decision.** (a) `runtime.JobClock` defaults to a STATIC allotment of 110 s per case (`STATIC_ALLOT_S`, env AIMO_STATIC_ALLOT_S);
the shared mode is kept but only used with an explicit persistent `state_dir`. Rationale: val = 200 s per case by the published rule;
110 s uses ~63% of the val budget and survives a test with up to ~1.67x val's cases-per-problem ratio; the residual risk (a test
much denser in cases per problem than val) is stated in the report and is the subject of the open #qa question.
(b) `submissions/timing-canary-v3` (no model, <1 s per case) probes 11 locations for a surviving marker, dumps env names and
timeout/case-like values, /proc/1, uptime and monotonic time. If something persists, JobClock gets that `state_dir` and the
adaptive clock returns; if an env variable carries the case count or deadline, the clock uses it directly.
(c) Per-model K/cut defaults (hybrid-v1 `MODEL_RUN`): Qwen 8x3072, Skywork 8x3072, Olmo 6x2048, DeepSeek 8x2560, gpt-oss 4x1536;
to be re-checked against the self-label signal at those cuts when the Kaggle data arrive (fitter `--cut`).

### 43. Qwen3.5-4B originals done on Kaggle; forcing moves to chunked prefill; venues (2026-10-18 01:30 local time)
**Data.** Shards a/b v3: 464/464 records with token_ids (60 AMC problems x 8 samples, samples 0-3 minus the Mac's 16 + samples
4-7), ~51 tok/s per T4 at batch 8; 84% of samples hit the 8k cap; all 76 samples that finished are correct; forced answers at 8k
are missing because the one-at-a-time prefill still OOMed (4.2 GiB fp32 copy in Qwen3.5's torch linear attention). Natural base
>= 0.5 on only 13/60 problems, so the labels depend on forced answers (Mac night-1: 10/15 forced answers correct at 8k).
**Decisions.** (a) Forced answers everywhere now use a chunked prefill (2048-token chunks through the KV cache, then greedy
step by step): kernel template, `force_at_cuts_hf.py`; the MLX script already chunks. (b) Forced answers at cuts 512/1024/2048/
3072/4096/8192 for all Qwen records are computed on the Mac this morning (`tools/morning_cuts.sh`, armed: waits for the night
generation to end or 07:40, then MLX, ~2-3 h, AC). (c) Hidden states for the 464 Kaggle records go to the Kaggle states-only
kernel (`kx-states-qwen35-amc`, SKIP_CUTS=1, ~2 h), pushed automatically after Skywork and DeepSeek finish ONLY if the projected
weekly total stays <= 25 GPU-h (14.5 used + Skywork + DeepSeek + 2.5); otherwise the author decides. Mac states (79 s per trace on MPS)
remain the fallback for the Mac-generated samples. (d) Kaggle records carry model_id "Qwen/Qwen3.5-4B", Mac records the local
path: build_labels / select_variants must normalise the id (TODO when merging).

## 2026-10-22 (restart after the forced quit; website update of Oct 16-21)

### 44. Rules re-read: the White-Box Award needs a SUBSTANTIAL white-box effect; plan re-centred on the author's goal (Writeup + White-Box)
**Evidence.** Website HEAD 0bb0dbb (2026-10-21): a submission is considered for its track's White-Box Award (+$1,000) if it uses
white-box signals "and its technical report shows, through a dedicated ablation, that this signal has a substantial effect on
the submission's performance". Writeup prizes are now $4,000, "judged on their scientific contribution", with stated interest in
**generalization of interpretability methods, actionable interpretability, negative results and efficient methods**. Leaderboard
prizes cut to Main 2,500/?/500, Small 1,500/700/?. Report rules: reproducible from the report alone, training data + codebase +
bundle documented, the submission may depend only on documented resources; award candidates review up to 3 reports.
**Consequences.** (a) #34's fallback "nominate the white-box bundle even if the probe adds nothing" no longer wins anything: the
white-box submission must be designed so that the internal signal carries the decision. (b) The case-level probe of #33 has
only ~170 labelled small-model cases to learn from (byeolki 124 + official 43); 16 PCA dims x layers on that is the setting in
which competitors' probes fell to chance (FACTS §9). (c) The report's strongest assets are already in hand: the label-noise
ceiling (negative result about the benchmark itself), val-board vs held-out generalization, and the server-time budget.
**Decision: two-stage white-box solve-rate estimator (main white-box line).**
1. Stage 1, per model, white-box: a **sample-level correctness probe** on the residual stream (answer-position state and
   trace-mean state at L/4, L/2, 3L/4; PCA inside folds, logistic) predicting whether that sample's final answer (natural or
   forced at the test-time cut) is correct. Labels are gold answers, so every generated sample is a training example
   (~1,400 AMC samples x cuts now, more with the new generations): no perturbation variants are needed for stage 1.
2. Stage 2, shared across models: robust = logistic(mean p_correct over K samples, its dispersion, model offset), a handful of
   parameters calibrated on the labelled cases (byeolki band labels, official train-main-v2). This follows from our own finding
   that "robust" ~ "solves the original with probability ~1" (label_noise_ceiling.md).
3. Ablation arms on the same held-out problems (grouped CV by problem, #34 statistics): prior; **black-box twin** (stage 1
   replaced by sample answer-agreement, identical stage 2); white-box; hybrid. Plus the efficiency curve: accuracy vs K = 1, 2,
   4, 8 for white-box and black-box at equal decode budget. The white-box submission is the arm with the white-box stage;
   its report ablation is "white-box stage -> black-box twin" and "-> prior". All arms reported whatever the sign.
4. The case-level PCA probe of #33 stays as a secondary arm (already in the bundle).
**Data needed.** Test-time-style samples (K = 8 at the per-model cut, token_ids, natural + forced answers) on the 73 byeolki
problems outside our AMC set and the 36 train-main-v2 problems, for Qwen / Skywork / DeepSeek (Olmo: train-main-v2 only if
budget allows; gpt-oss: none, its cases use agreement + prior). The 29 AMC problems shared with byeolki reuse our 8k traces
cut at the test-time length (the first c tokens of a sample are a sample with cap c). States by teacher forcing on Kaggle.
**Server check first.** `dist/hybrid-v1-pipecanary-v1-main.zip` runs the full hybrid-v1 pipeline (K samples, hooks, forced
answers) on every val case and answers the prior (expected 0.48): first measurement of K x cut per model on the server and of
gpt-oss with hooks.

### 45. Correction: solve rate alone does NOT recover most of the label (2026-10-22)
Re-reading `label_noise_ceiling.md` section B: the v0 reading ("knowing only the true solve rate already recovers most of the
ceiling") contradicted its own table. Re-run with a majority column and the train-main-v2 convention (6 types, no
expert_no_solution): the solve-rate oracle beats the majority by only +2 to +9 points (0.54->0.63 / 0.65->0.70 at pi scale 0.5;
below majority at scale 1.0), while a predictor that knows the true per-type degradation reaches 0.88-0.94. In this simulated
world the label is mostly about genuine degradation, not solve rate. Caveat: the simulation draws degradations independently of
p0; on real data (M1, byeolki) agreement beat the prior by ~10 points while the held-out solve rate had AUC only 0.53-0.59 for
Qwen/Skywork, i.e. agreement carries stability information beyond the solve rate.
Consequences: (a) the report's claim 1 becomes "labels are noisy and solve rate explains only part of them" (outline v1 title
to be revised); (b) stage 1 of #44 gets a second, gold-free target: **consistency** (does this sample's answer equal the
plurality answer of the other samples?). A probe for it is a white-box distillation of agreement; at K = 1 it is the
efficiency test (can one sample's internals stand in for K-sample agreement?). Both targets are run and reported; the
pre-registered primary remains correctness (#44), consistency is secondary.

### 46. Nomination rule, draft (to be frozen by 2026-10-27, before any candidate's board score is seen)
Facts: at most 5 nominations per track, the BEST private score among them counts (D6, D11); the White-Box Award goes to the
highest-ranked white-box submission whose report shows a substantial ablation effect (#44). Nominating more distinct candidates
can only help the leaderboard prize, and nominating several white-box variants helps the award.
Rule (per track, same bundles; Small gets small.txt; on Main gpt-oss cases use the prior in every candidate):
1. **wb-K8**: two-stage, stage-1 target and layer as chosen by the pre-registered CV (#44: correctness primary), K = 8. White-box.
2. **hybrid-K8**: wb + bb stage-2 features. White-box (ablation reported against bb).
3. **bb-K8**: black-box twin only (the best black-box method; also the ablation partner of 1-2).
4. **wb-alt**: the stage-1 variant the CV ranks second among {correctness, consistency} x {trace, answer} blocks, K = 8. White-box.
5. **wb-K-efficient**: the wb arm at the smallest K whose held-out accuracy is within 1 point of wb-K8, with longer cuts if the
   server timing (pipeline canary) allows. White-box.
Gates (all must hold to nominate; otherwise the slot goes to the next candidate or the prior): passes the per-case emulator on
both tracks; its pipeline-canary-measured worst case fits <= 60% of the val time budget; held-out CV accuracy >= prior.
The val board is used only to check validity (coverage 1.0, 0 invalid), never to choose among candidates (D11, #30).
Fallback if Codabench is unreachable near the deadline: the first real candidates (1-3) are uploaded by 2026-10-29.

### 47. Two nominations only; label-efficiency becomes a headline result (2026-10-23, FACTS D15-D18)
(a) **Nomination rule (replaces the 5-slot draft of #46; to be frozen by 2026-10-27 before any candidate's board score is
seen):** slot 1 = the white-box arm with the best held-out CV accuracy among {wb, hybrid} whose report ablation (white-box ->
black-box twin, and -> prior) is reported in full; slot 2 = the best remaining arm by held-out CV among {wb, hybrid, bb}, so
that a black-box method is nominated only if it is clearly better (>= 2 points held-out) than both white-box arms. Ties go
to the white-box arm (award eligibility). Gates of #46 unchanged (emulator, canary timing <= 60% of budget, CV >= prior).
(b) **Label-efficiency curve** (organizers want methods that generalize from 10s of examples, D16): stage-2 accuracy on
held-out problems vs the number of labelled training cases (10, 20, 50, 100, all; 20 random draws each), for wb, bb and
hybrid; stage 1 uses no robustness labels (gold answers; the consistency target uses none at all). Reported as a main figure.
(c) The val board is a validity check only; nothing is tuned or keyed on val problems (D15).

### 48. Report deadline moved to Nov 5 AoE: write in parallel (2026-10-23, FACTS D19)
Only ~5 days separate the competition close (Nov 1 12:55 UTC) from the report deadline (Nov 6 18:59 local time). Plan:
Oct 24-26 data (Kaggle week 2 + Mac) and the full ablation; Oct 27 freeze the nomination rule (#47) and the report's
claims list; Oct 27-29 first real uploads; **Oct 26-31 LaTeX draft of every section that does not depend on the test
(setting, label instrument, method, data, held-out ablation, label-efficiency, limitations)**; Nov 1-4 add the board/test
numbers available, polish, reproducibility appendix, open-source release; Nov 5 submit (local time morning of Nov 6 at the latest).
the author: OpenReview profile now; he is the reciprocal reviewer (3 reviews by Nov 12); voluntary reviewing could earn an
Outstanding Reviewer Award (NeurIPS registration).

### 49. PROPOSAL (2026-10-25 night, for the author; freeze by 2026-10-27 before any candidate's board score is seen)
Evidence (v2, `notes/results/wb_two_stage_v2.md`): no arm passes the #46 gate "held-out CV >= prior" (best wb 0.656-0.659 vs
prior 0.656-0.669, all within MDE80 ~0.09); wb > bb at K4 on the consistency target (Holm p = 0.035). Read literally, #46/#47
would give both slots to the prior and forfeit White-Box eligibility. But the best private score of the two nominations counts
(D6, D18), so a prior slot already protects the leaderboard and the gate's purpose is met by the pair.
Proposed amendment: **slot 1 = prior-only** (leaderboard floor), **slot 2 = the white-box arm with the best held-out CV**
(selection among wb/hybrid x target x K exactly as #47a, gate "CV >= prior" replaced by "CV >= prior - MDE80/2" so a
statistically indistinguishable white-box arm qualifies); both tracks; bb is uploaded to the val board for the ablation only.
Val-board uploads needed for the report (D20): wb (slot 2 candidate), bb, hybrid; prior exists (991595/991597).
Open with the organizers (D18): whether the White-Box Award considers a team's white-box nomination when its other nomination
ranks higher.
Addendum (21:35, FACTS D22): the site says every submission is considered for the White-Box Award automatically, but the
ablation must show the white-box signal has a "substantial effect" on THAT submission's performance. So the white-box slot should
go to the arm whose with/without-probe ablation is clearest, not only the highest CV: on v2 that is wb at K4 with the
consistency target (wb 0.633 vs its twin 0.589, +10.2 points, Holm p = 0.035), while wb-K8 (0.656) has only +4 vs its twin
(p = 0.27). Choice between the two to be made with the author on v3 (Skywork/DeepSeek AMC states added), disclosed in the report as a
choice made on held-out CV, before any board score.

### 50. Label shift: score stage 2 by balanced accuracy too (2026-10-25 21:40, before any candidate's board score)
Trigger: the val board is balanced (Main 10/21, Small 7/14 robust; prior-only scored 0.48) while our labelled cases are 66-71%
robust, and the organizers give no guarantee on the test label mix (D18). Under a balanced test the per-model majority scores
~0.5, so "CV accuracy >= prior" on our imbalanced cases is the wrong gate. Added `analysis/wb_balanced.py` (stage 2 with
class_weight='balanced', balanced accuracy, cluster bootstrap by problem). First result (`notes/results/wb_balanced_v2b.md`,
146 cases incl. Skywork AMC): **bb (agreement) beats the prior by +12 points of balanced accuracy at K4 and K8 (95% CI
[+0.02, +0.22], p = 0.016 / 0.028)**; wb is 0.51-0.57 and adds nothing to bb (hybrid 0.59-0.61 vs bb 0.61-0.63). The earlier
"wb > bb at K4" (#49 addendum) was an artefact of thresholding at the majority class: bb fell below the prior there because its
unweighted stage 2 ran into the class imbalance. Consequences for #49: the leaderboard slot should be judged on balanced
accuracy (bb-K4/K8 is the strongest candidate), and the White-Box ablation does not currently show a substantial white-box
effect under either metric. To be confirmed on v3 (DeepSeek AMC states) and decided with the author.

### 51. Recommendation on v3 (2026-10-25 22:40; for the author's choice, freeze by Oct 27, no candidate board score seen)
v3 balanced accuracy (`notes/results/wb_balanced_v3.md`, 158 cases, 78 problems, stage 2 class-balanced, K = 4):
bb 0.657 (prior +0.137, 95% CI [+0.04, +0.24], p = 0.005); **wb (correctness probe) 0.619 (prior +0.106, CI [+0.01, +0.20],
p = 0.033)**; hybrid 0.610; wb vs bb -0.030 (CI [-0.13, +0.06]). Under unweighted accuracy on our 70%-robust cases nothing
beats the majority; the val board is balanced, so #50 makes balanced accuracy the selection metric.
Applying #47a with this metric: slot 1 (white-box, best CV among wb/hybrid) = **wb-K4-balanced**; slot 2 = **bb-K4-balanced**
(black-box, >= 2 points above both white-box arms). Both gates pass on balanced CV (>= prior 0.5). K = 4 rather than 8: higher
balanced CV for every arm and half the samples (runtime tie-break). White-Box ablation for the report: wb-K4 with vs without its
probe (-> prior: -10.6 points balanced, CI excludes 0; -> black-box twin: -3.0, n.s.), reported in full, plus the val-board
scores of wb, bb, hybrid and prior. Risk: if the test is as imbalanced as our labels, both balanced arms lose ~2-3 points of
accuracy to the always-robust prior.
Robustness (23:45): wb-K4 balanced accuracy over three stage-1 cross-fitting seeds 0.619 / 0.600 / 0.622, gain over the prior
+0.106 / +0.092 / +0.118, every 95% CI above 0 (`notes/results/wb_balanced_v3_s1seed{1,2}.md`). At K = 1 (one sample) wb is
0.576 / 0.594 / 0.630 vs bb 0.549.
Break-even (00:15 Oct 26): with class recalls TPR/TNR the accuracy at robust share pi is pi*TPR + (1-pi)*TNR. wb-K4 beats
always-robust when pi < 0.59, bb-K4 when pi < 0.63. Evidence on the test mix conflicts: D2 says the test is "NOT balanced"
(no forced 50% rate), the val sets (built, per byeolki's relay, the same way as the test) are 0.48/0.50, the official train
is 0.74 robust, our labels 0.70. Hence the honest framing for the author: the balanced arms win if the test looks like val, the
always-robust prior wins if it looks like train. Note #51's slot proposal (wb + bb) bets on the val-like mix; the alternative
(wb + prior) hedges: the best of the two nominations counts.

### 52. FROZEN nomination (the author, 2026-10-26 morning; recorded before any candidate's board score was seen)
the author's decision on #51: **per track (Main and Small), nominate (1) twostage-wb-k4 v1 (white-box, class-balanced stage 2, K = 4)
and (2) the constant "robust" prior**. If the organizers allow 3 nominations per track (D18 says 2), **add twostage-bb-k4 v1 as
the third**. Rationale (#51 break-even): wb wins if the test is val-like (robust share < 0.59), the prior if it is train-like
(0.74); the best private score of the nominations counts, and wb keeps White-Box eligibility with its ablation (-10.6 points
balanced without the probe).
Slot 2 needs no new upload: prior-only v1 is already on the board (Main 991595, Small 991597; constant True for every known
model and the pooled fallback, stdlib only, per-call safe because it returns one verdict per problem in the list). The wb,
bb and hybrid zips are uploaded for their val scores, which go into the report only and cannot change this choice.
Open (D18, the author may ask in #qa): whether the White-Box Award considers the wb nomination if the prior nomination ranks higher.
Addendum (2026-10-26, the author): Codabench is still down and the OpenReview form asks for "Leaderboard Submission IDs", so the
prior-only v1 zips (unchanged, re-checked) are re-submitted on the replacement UI to give the nominated pair IDs in the current
system. Re-submitting the same frozen bundle does not change the choice.

### 53. Validation ablation of the submitted wb arm (2026-10-26, approved by the author; nominations #52 unchanged)
Server logs (`notes/results/val_board_v1.md`) show the three Main arms make identical predictions on all 21 cases (13 cases are
gpt-oss/Olmo, decided by the shared bb_any stage 2; seeded sampling gives identical samples; on the 8 white-box cases p differs
but never crosses 0.5 differently) and the Small arms differ on 1 of 14 cases. The board's wb-vs-prior gap therefore cannot be
credited to the probe, and the 14.3/7.1-point "validation ablation" written into the report on Oct 26 was wrong (removed).
Direct validation ablation: `twostage-wb-k4-noprobe` = the uploaded wb artifact plus `ablate_models` = the three white-box
models, which then answer the prior (robust; priors 0.67-0.75) without sampling; Olmo/gpt-oss unchanged. Its score minus wb's
is exactly the effect of the probe on the board: wb - noprobe = (#white-box-model cases wb gets right) - (#robust among them).
Pre-registered expectation: a few cases either way (8 white-box cases on Main, 11 on Small); reported whatever the sign.

### #52 addendum 2 (2026-10-26 22:30): organizer answer on the White-Box ranking
PMMon confirmed on Discord (FACTS D23) that with a black-box baseline and a white-box method both submitted via Codabench,
the white-box submission is ranked on its own for the White-Box Award. The frozen pair stands unchanged. Because the answer
and Josef's 21:37 announcement both point to the official Codabench queue, the wb-K4 zips (same files, sha 57ea3888618a /
93c4da45ef26) are uploaded there too; the prior already has Codabench IDs 991595 / 991597.

### #54 (2026-10-27): post-hoc check from the self-review — offsets-only baseline
The self-review noticed that at K = 1 the black-box twin is constant, so the K=1 bb row (0.549) is stage 2 with the per-model
offsets only: a per-model constant fit with balanced class weights already beats the 0.5 majority. Against this baseline the
probe adds +7.0 points of balanced accuracy at K = 4 (CI [-4.3, +19.0], p = 0.24; analysis/wb_vs_offsets.py). Not
pre-registered; reported in the paper next to the pre-registered wb -> prior ablation (+10.6, p = 0.033), with the White-Box
evidence described as suggestive rather than conclusive. Nominations (#52) unchanged.
#54 update (2026-10-27, 10^4 paired cluster-bootstrap resamples per stage-2 seed, notes/results/wb_vs_offsets_v3.md):
wb - prior +11.8/+13.3/+10.6 (every CI excludes 0, p 0.010-0.036; mean +11.9); wb - offsets +6.9/+7.4/+6.8 (every CI
includes 0, p 0.14-0.24); bb - prior mean +15.7; wb - bb mean -3.8 (n.s.). The report previously paired 3-seed-mean
accuracies (0.619/0.657) with last-seed deltas (+10.6/+13.7/-3.0); it now reports seed means (+11.9/+15.7/-3.8) with per-seed
CIs (Appendix table). The held-out effect vs the fair baseline does NOT exclude 0, so it is not made the primary claim; the
pre-registered wb -> prior ablation stays primary, the offsets comparison is reported as post hoc. Nominations unchanged.
