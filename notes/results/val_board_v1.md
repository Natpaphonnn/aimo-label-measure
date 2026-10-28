# Validation board, per-case decisions from the server logs (runs/val_logs, not in git)

Parsed by scratch parse_val.py from the six Full Logs (2026-10-26). Columns: case order as served; k, cut; agreement top share per arm; stage-2 path per arm (wb/bb/hybrid); p_robust per arm; predicted robust per arm; seconds. No labels are available per case; only the board totals.

```

### main: cases {'wb': 21, 'bb': 21, 'hybrid': 21}
i model effort | k cut agree(wb/bb/hy) | arm_used wb/bb/hy | p wb/bb/hy | robust wb/bb/hy | sec
1 Qwen3.5-4B default | 4 3072 [0.0, 0.0, 0.0] | wb bb hybrid | 0.3933 0.3164 0.3132 | [0, 0, 0] | [58.8, 58.2, 58.7]
2 Qwen3.5-4B default | 4 3072 [0.75, 0.75, 0.75] | wb bb hybrid | 0.2918 0.487 0.3717 | [0, 0, 0] | [58.2, 58.7, 58.2]
3 Skywork-OR1- default | 4 3072 [0.0, 0.0, 0.0] | wb bb hybrid | 0.4083 0.4346 0.3747 | [0, 0, 0] | [48.6, 48.6, 48.6]
4 Skywork-OR1- default | 4 3072 [0.25, 0.25, 0.25] | wb bb hybrid | 0.4594 0.4346 0.405 | [0, 0, 0] | [48.4, 48.5, 48.6]
5 Skywork-OR1- default | 4 3072 [1.0, 1.0, 1.0] | wb bb hybrid | 0.8335 0.7623 0.8381 | [1, 1, 1] | [47.3, 47.2, 47.3]
6 Olmo-3-7B-Th default | 4 2048 [0.5, 0.5, 0.5] | bb_any bb_any bb_any | 0.4013 0.4013 0.4013 | [0, 0, 0] | [43.6, 43.8, 43.7]
7 Olmo-3-7B-Th default | 4 2048 [0.75, 0.75, 0.75] | bb_any bb_any bb_any | 0.5213 0.5213 0.5213 | [1, 1, 1] | [43.9, 43.9, 44.0]
8 DeepSeek-R1- default | 4 2560 [0.0, 0.0, 0.0] | wb bb hybrid | 0.3112 0.2685 0.2554 | [0, 0, 0] | [49.0, 49.0, 49.0]
9 DeepSeek-R1- default | 4 2560 [0.25, 0.25, 0.25] | wb bb hybrid | 0.3287 0.2685 0.2649 | [0, 0, 0] | [48.9, 48.9, 48.9]
10 DeepSeek-R1- default | 4 2560 [0.25, 0.25, 0.25] | wb bb hybrid | 0.3254 0.2685 0.2633 | [0, 0, 0] | [49.3, 49.2, 49.1]
11 gpt-oss-120b high | 4 1536 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [75.4, 75.2, 75.5]
12 gpt-oss-120b low | 4 1536 [0.5, 0.5, 0.5] | bb_any bb_any bb_any | 0.4013 0.4013 0.4013 | [0, 0, 0] | [60.0, 59.7, 59.3]
13 gpt-oss-120b low | 4 1536 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [68.9, 68.2, 69.1]
14 gpt-oss-120b low | 4 1536 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [67.0, 66.6, 66.9]
15 gpt-oss-120b medium | 4 1536 [0.75, 0.75, 0.75] | bb_any bb_any bb_any | 0.5213 0.5213 0.5213 | [1, 1, 1] | [66.8, 67.1, 66.9]
16 gpt-oss-120b medium | 4 1536 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [71.4, 70.9, 71.6]
17 gpt-oss-120b high | 4 1536 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [67.4, 67.9, 68.4]
18 gpt-oss-120b low | 4 1536 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [68.3, 68.0, 68.5]
19 gpt-oss-120b medium | 4 1536 [0.5, 0.5, 0.5] | bb_any bb_any bb_any | 0.4013 0.4013 0.4013 | [0, 0, 0] | [68.2, 68.8, 68.3]
20 gpt-oss-120b low | 4 1536 [1.0, 1.0, 1.0] | bb_any bb_any bb_any | 0.6807 0.6807 0.6807 | [1, 1, 1] | [68.0, 68.2, 68.1]
21 gpt-oss-120b high | 4 1536 [0.5, 0.5, 0.5] | bb_any bb_any bb_any | 0.4013 0.4013 0.4013 | [0, 0, 0] | [66.5, 66.4, 66.7]
cases where predictions differ: 0 | robust counts {'wb': 4, 'bb': 4, 'hybrid': 4}
missing two-stage line: {'wb': 0, 'bb': 0, 'hybrid': 0}

### small: cases {'wb': 14, 'bb': 14, 'hybrid': 14}
i model effort | k cut agree(wb/bb/hy) | arm_used wb/bb/hy | p wb/bb/hy | robust wb/bb/hy | sec
1 Qwen3.5-4B default | 4 3072 [0.25, 0.25, 0.25] | wb bb hybrid | 0.5269 0.3164 0.3887 | [1, 0, 0] | [57.8, 58.8, 58.0]
2 Qwen3.5-4B default | 4 3072 [0.0, 0.0, 0.0] | wb bb hybrid | 0.3933 0.3164 0.3132 | [0, 0, 0] | [57.2, 58.6, 57.5]
3 Qwen3.5-4B default | 4 3072 [0.5, 0.5, 0.5] | wb bb hybrid | 0.2884 0.3705 0.2935 | [0, 0, 0] | [58.0, 58.4, 57.7]
4 Qwen3.5-4B default | 4 3072 [0.75, 0.75, 0.75] | wb bb hybrid | 0.3123 0.487 0.386 | [0, 0, 0] | [57.9, 59.1, 57.9]
5 Qwen3.5-4B default | 4 3072 [0.75, 0.75, 0.75] | wb bb hybrid | 0.2918 0.487 0.3717 | [0, 0, 0] | [57.3, 57.9, 57.3]
6 Skywork-OR1- default | 4 3072 [0.0, 0.0, 0.0] | wb bb hybrid | 0.4083 0.4346 0.3747 | [0, 0, 0] | [48.5, 48.6, 48.5]
7 Skywork-OR1- default | 4 3072 [0.25, 0.25, 0.25] | wb bb hybrid | 0.4594 0.4346 0.405 | [0, 0, 0] | [48.4, 48.5, 48.4]
8 Skywork-OR1- default | 4 3072 [1.0, 1.0, 1.0] | wb bb hybrid | 0.8335 0.7623 0.8381 | [1, 1, 1] | [47.1, 47.3, 47.3]
9 Olmo-3-7B-Th default | 4 2048 [0.25, 0.25, 0.25] | bb_any bb_any bb_any | 0.3417 0.3417 0.3417 | [0, 0, 0] | [44.2, 44.4, 44.2]
10 Olmo-3-7B-Th default | 4 2048 [1.0, 1.0, 1.0] | bb_any bb_any bb_any | 0.6807 0.6807 0.6807 | [1, 1, 1] | [43.8, 43.8, 43.9]
11 Olmo-3-7B-Th default | 4 2048 [0.5, 0.5, 0.5] | bb_any bb_any bb_any | 0.4013 0.4013 0.4013 | [0, 0, 0] | [44.3, 44.6, 44.3]
12 DeepSeek-R1- default | 4 2560 [0.25, 0.25, 0.25] | wb bb hybrid | 0.3501 0.2685 0.2765 | [0, 0, 0] | [49.0, 49.2, 49.0]
13 DeepSeek-R1- default | 4 2560 [0.0, 0.0, 0.0] | wb bb hybrid | 0.3112 0.2685 0.2554 | [0, 0, 0] | [49.1, 49.2, 48.9]
14 DeepSeek-R1- default | 4 2560 [0.25, 0.25, 0.25] | wb bb hybrid | 0.3287 0.2685 0.2649 | [0, 0, 0] | [48.8, 49.0, 48.8]
cases where predictions differ: 1 | robust counts {'wb': 3, 'bb': 2, 'hybrid': 2}
missing two-stage line: {'wb': 0, 'bb': 0, 'hybrid': 0}
```

Findings: no fallback, timeout or invalid case in any run; Main predictions identical across wb/bb/hybrid (21/21); Small differ on case 1 (Qwen, wb robust p=0.527, correct per the 8/14 vs 7/14 totals). From totals: Main wb predicts 4 robust of which 3 are robust, 10 of its 17 non-robust calls are right; Small wb 2 of 3 robust calls right, 6 of 11 non-robust calls right.

## wb without its probe (#53): Main b018b198 = 0.4762 (10/21), Small 24e6d3d6 = 0.5000 (7/14)
Derivation (CONFIRMED from the no-probe logs 2026-10-26: no fallback; the 8 / 11 white-box-model cases answered the prior
without loading the model; every gpt-oss/Olmo case has the same agreement, p_robust and decision as wb): white-box-model cases Main 8 / Small 11; wb predicts robust on 1 / 2 of them, the no-probe arm on all.
wb - noprobe = (#WB cases wb right) - (#robust among WB cases) = +3 / +1 cases => discordant 7 / 9, wb right on 5 / 5
(exact McNemar p = 0.45 / 1.0; pooled 10 vs 6, p = 0.45). On the bb_any cases (13 / 3) agreement is exactly as accurate as
always-robust (bb_any correct = noprobe total - robust WB = robust among bb_any cases).

## Codabench re-run of wb-K4 Main (1011261, 2026-10-27, official queue) = 0.524 (11/21) vs temporary queue 064c9648 = 0.619 (13/21)
Log pasted by the author (not saved as a file); per case (agree_top, p_robust, robust) compared with runs/val_logs/064c9648.log:
- cases 1-10 (Qwen x2, Skywork x3, Olmo x2, DeepSeek x3): bit-identical p_robust and decisions.
- 11 gpt-oss cases: agree_top differs in 9 of 11 (cases 12-18, 20, 21); decisions differ in 4 (12, 13, 18 -> robust; 15 -> non-robust).
  Robust predictions 4 -> 6; net -2 correct. Small (1009859 = 0.571) reproduced exactly (no gpt-oss on Small).
- Reading: gpt-oss-120b sampling (hub triton MoE kernels) is not reproducible across runs/workers despite the per-problem seed;
  the white-box path is. Every differing case is on the bb_any path shared by wb/bb/hybrid/no-probe, so the held-out ablation is
  unaffected; the board carries ~2 cases of run-to-run noise on Main.
