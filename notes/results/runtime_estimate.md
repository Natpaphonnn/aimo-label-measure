# Runtime estimate

## v5 (2026-10-17 evening): long-context decode measured on the server (canary v2 995120) and the static allotment

Canary v2 (Main, 0.48, 0 invalid, ~2,500 s of 4,200): sampled decode at the official settings up to 4,096 new tokens, timestamps every
256 tokens (steps/s = decode steps per second for the whole batch; tokens/s = steps/s x batch).

| model | batch | steps/s start -> at 4k | 4,096 tokens take | note |
|---|---|---|---|---|
| Qwen3.5-4B | 8 | ~52 flat | ~79 s | linear attention: no slowdown with context |
| Skywork-OR1-Math-7B | 4 | 74 -> 67 | ~58 s | |
| Olmo-3-7B-Think | 8 | 55 -> 21 | ~3.9k in 130 s | strong slowdown; smaller batch or shorter cap |
| DeepSeek-R1-0528-Qwen3-8B | 16 | 55 -> 28 | ~105 s | |
| gpt-oss-120b low/medium | 4 | ~25 flat | ~3.3k in 130 s | |
| gpt-oss-120b high | 16 | 25 -> 21 | ~3.0k in 130 s | |

Peak memory <= 64 GB (gpt-oss), fine on the 96 GB card. Load per case 3.5-6.5 s (canary v1).

**No state persists between cases.** Every case saw an empty temp dir (`tmp_seen=false`, `tmp_count=1`, `ppid=1`), so the shared
JobClock (v4) can never see a second case and would fall back to ~24 s per case. Canary v3 (`submissions/timing-canary-v3`) probes
/app/output, /app/input, /app, the program dir, cwd, $HOME, /dev/shm, /var/tmp, the temp dir and /root for a marker that survives,
dumps environment variable names (and the values of timeout/case/count-like names), /proc/1, uptime and `time.monotonic()`.

**Static allotment (now the default, `runtime.STATIC_ALLOT_S` = 110 s).** The published rule gives 4,200 s for the 21 val Main cases:
262.5 s per unique problem = 200 s per case at val's 1.31 cases per problem. The organizers rescale the test cap by the same rule and
say that passing val means fitting the test, so the val ratio is the best available estimate of the test ratio. 110 s per case plus
~10 s of process overhead uses ~63% of the budget on val and stays within the cap for a test with up to 1.67 x val's cases per
problem (2.2 cases per problem). Beyond that the tail cases are lost (train-main-v2's 2.28 cases per problem would overrun by ~4%).
We state this risk in the report; a persisting location found by canary v3 would restore the adaptive clock.

**Per-model K and cut inside 110 s** (load + forced answers + reserve ~25 s -> ~85 s of generation; `hybrid-v1/features.MODEL_RUN`):

| model | K | cut (tokens) | expected generation | basis |
|---|---|---|---|---|
| Qwen3.5-4B | 8 | 3072 | ~60 s | 52 steps/s flat |
| Skywork | 8 | 3072 | ~50 s | 74 -> 67 at b4; b8 assumed ~60 |
| Olmo-3-7B-Think | 6 | 2048 | ~50 s | 55 -> ~35 over the first 2k |
| DeepSeek | 8 | 2560 | ~55 s | 55 -> 28 at b16; b8 assumed ~45 |
| gpt-oss-120b (all efforts) | 4 | 1536 | ~65 s | ~25 flat, load 6.5 s |

The cut shrinks automatically when the allotment is smaller (`steps_per_s` x 0.85) and `max_time` bounds generation. Still to check
once the Kaggle self-labels are in: that the agreement and white-box signals survive at these cuts (`fit_hybrid_artifact.py --cut`).

## v4 (2026-10-16 evening): one global budget per unique problem, shared across fresh processes

Organizers (#qa, FACTS D11): one cap for the whole evaluation, scaled by unique problems (val Main 4200 s / 16 = 262.5 s per
problem), no per-call limit, cases per problem vary on the test. Every case is a fresh process, so the budget is shared through a
state file in the temp dir: `runtime.JobClock`. Each newly seen problem earns 240 s; spent = wall clock since the job's first case;
overhead (10 s) is reserved for every POSSIBLE future case (up to 7 per problem on Main, 4 on Small); the rest is shared equally over
the cases EXPECTED for problems already seen (3 per problem on Main and 4 on Small until more are observed). A case below 15 s
answers its prior. Simulation (`analysis/sim_jobclock.py`; fake clock, every case uses its whole allot, 8-9 s of real overhead):

| scenario | cases | used / budget | mean allot s | min allot s | prior fallbacks |
|---|---|---|---|---|---|
| val Main 21 cases / 16 problems, random | 21 | 0.323 | 56.6 | 50.0 | 0 |
| val Small 14 cases / 11 problems (assumed), random | 14 | 0.279 | 49.5 | 45.0 | 0 |
| test 40 problems x 7 cases (main), grouped by problem | 280 | 0.912 | 26.2 | 2.0 | 4 |
| test 40 problems x 7 cases (main), grouped by config | 280 | 0.912 | 26.8 | 0.1 | 152 |
| test 40 problems x 7 cases (main), random | 280 | 0.912 | 27.4 | 10.6 | 29 |
| test 40 problems x 4 cases (small), grouped by problem | 160 | 0.912 | 51.9 | 45.0 | 0 |
| test 40 problems x 4 cases (small), grouped by config | 160 | 0.912 | 51.9 | 45.0 | 0 |
| test 40 problems x 4 cases (small), random | 160 | 0.912 | 51.9 | 45.0 | 0 |
| test 40 problems x 3 cases (main), grouped by problem | 120 | 0.76 | 58.5 | 50.0 | 0 |
| test 40 problems x 3 cases (main), grouped by config | 120 | 0.76 | 58.5 | 50.0 | 0 |
| test 40 problems x 3 cases (main), random | 120 | 0.76 | 58.5 | 50.0 | 0 |
| test 40 problems x 1-7 cases (main), random | 164 | 0.627 | 33.8 | 12.7 | 21 |
| stateless fallback, 40 x 7 main | 280 | 0.861 | 24.3 | 24.3 | 0 |

Reading: never above 0.912 of the true budget, in any ordering. The cost of not knowing the test's cases per problem: on val the
clock uses only ~30% of the budget (~50-57 s per case), and a Main test with 7 cases per problem ordered by config would answer half
its cases with the prior. The stateless fallback (if the temp dir did not persist) is safe for any order at ~24 s per case.
Consequence: features must work at a variable budget (about 25-200 s, i.e. K x tokens chosen from the allot; DECISIONS #31).
Open questions for the organizers: typical cases per unique problem on the test, and the case order.

## v3 (2026-10-16 afternoon): measured on the worker (timing canary 991764)

Every case runs in a fresh process under one shared job budget: 4200 s for the 21 val Main cases with a 180 s safety margin, i.e.
191 s of wall clock per case, including process start and imports before our module loads. `runtime.PER_CASE_S` is now 160 s
(after import), leaving ~30 s per case of buffer; model load is 3.5-6.5 s per case because nothing persists between cases.

| model | steps/s (canary, short context) | load s | tokens per sample in 140 s of generation | if long context halves the rate |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 50.6 | 3.5 | 7084 | 3542 |
| Skywork/Skywork-OR1-Math-7B | 70.9 | 3.6 | 9926 | 4963 |
| allenai/Olmo-3-7B-Think | 58.4 | 3.6 | 8176 | 4088 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 60.4 | 3.8 | 8456 | 4228 |
| openai/gpt-oss-120b | 24.6 | 6.5 | 3444 | 1722 |

Measured at batch 8 with 192 new tokens and a short prompt (bf16, RTX PRO 6000 Blackwell; gpt-oss's first call in a job is ~20%
slower because of kernel warm-up). The v2 floor taken from the BAS logs (0.8-1.2k tokens for the 7-8B models) was ~6x too
pessimistic: BAS ran batch 18-36 at 5-6k-token contexts. Unknown: the slowdown at 2-4k-token contexts and at batch 16, worst
expected for Qwen3.5-4B, whose linear-attention layers fall back to the torch implementation (flash-linear-attention missing).
Timing canary v2 measures this (up to 4096 tokens, a timestamp every 256 tokens, batch 4/8/16).

Working target until then (DECISIONS #26): K = 4-8 samples, cap 4096 tokens for the four small models and ~2-3k for gpt-oss,
each with a forced answer at the cut and a hard stop from `max_time` in `generate`.

## v2 (2026-10-16 morning, superseded by v3: BAS-based floor): per-problem protocol

The server now calls `are_robust` once per case, with a time limit that scales with the number of cases (val Main: 1 h 10 min / 21 cases = 200 s; the organizers quote ~170 s per case for the test and promise that passing val means fitting the test). Our bundles use `runtime.CaseClock` with 150 s per case and bank unused time. With one problem per call, batching is only across the K samples (and test-time variants) of that problem, so the binding quantity is decode **steps per second**, not aggregate tokens per second:

| model | steps/s at batch 18-36 (BAS) | tokens per sample in 150 s (load 4 s, prefill + forced answer 10 s) | in 120 s |
|---|---|---|---|
| Qwen/Qwen3.5-4B | 30.7 | 4178 | 3256 |
| Skywork/Skywork-OR1-Math-7B | 8.8 | 1193 | 929 |
| allenai/Olmo-3-7B-Think | 5.5 | 752 | 586 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 8.7 | 1189 | 926 |
| openai/gpt-oss-120b | 7.5 | 1023 | 797 |

These step rates were measured at batch 18-36 with 5-6k-token contexts (HF `generate`, no static cache); at batch 4-8 and 1k contexts they should be higher, so the table is a floor. Reading: K = 4-8 parallel samples cost about the same wall time as K = 2; the cap must be about 1k tokens for the 7-8B models and gpt-oss, about 4k for Qwen3.5-4B, each with a forced answer at the cut. Next measurement: a timing canary upload (prior-only answers, logs steps/s per model at batch 4 and 8, no problem text).

## v1 (2026-10-16 morning, superseded: assumed one call per (model, effort) batch)

| problems per config | K | token cap | Small track total s | Main track total s (adds gpt-oss low/med/high) |
|---|---|---|---|---|
| 5 | 2 | 2048 | 369 | 674 |
| 5 | 2 | 4096 | 721 | 1184 |
| 5 | 2 | 8192 | 1426 | 1988 |
| 5 | 4 | 2048 | 721 | 1320 |
| 5 | 4 | 4096 | 1426 | 2339 |
| 5 | 4 | 8192 | 2836 | 3949 **over** |
| 15 | 2 | 2048 | 1074 | 1965 |
| 15 | 2 | 4096 | 2131 | 3495 **over** |
| 15 | 2 | 8192 | 4246 **over** | 5909 **over** |
| 15 | 4 | 2048 | 2131 | 3903 **over** |
| 15 | 4 | 4096 | 4246 **over** | 6961 **over** |
| 15 | 4 | 8192 | 8477 **over** | 11790 **over** |
| 30 | 2 | 2048 | 2131 | 3903 **over** |
| 30 | 2 | 4096 | 4246 **over** | 6961 **over** |
| 30 | 2 | 8192 | 8477 **over** | 11790 **over** |
| 30 | 4 | 2048 | 4246 **over** | 7778 **over** |
| 30 | 4 | 4096 | 8477 **over** | 13895 **over** |
| 30 | 4 | 8192 | 16937 **over** | 23553 **over** |

Assumptions: aggregate throughput per model as measured on the worker by BAS (Qwen 552, Skywork 309, Olmo 206, DeepSeek 216, gpt-oss 226 tok/s at 18-36 concurrent sequences), every sample hitting the cap (thinking models), gpt-oss traces 1.5k/3k/6k tokens at low/medium/high, +3% forced-answer pass, +15% shared-GPU margin, 420 s safety reserve. Throughput should rise with 60-120 concurrent sequences, so the table is conservative.

Reading: at validation-set scale (3-5 problems per config) K = 4 at 4096 tokens fits both tracks; at the paper's ~30 problems per config only K = 2 at 2048 fits Small, and Main needs less (e.g. K = 2 at 1024 for the 7-8B models). The submission must therefore choose K and the cap at run time from len(problems) and the remaining budget (the runtime module already allots time per batch). The test size is Discord question 5.
