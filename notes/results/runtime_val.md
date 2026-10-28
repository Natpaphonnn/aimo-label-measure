# Validation-server runtime of wb-K4 (analysis/runtime_table.py)

Seconds per case, median [min, max]. call = are_robust wall time inside the process (model load + sampling + probe + stage 2);
the harness adds process start and imports per case. Logs of the temporary queue (RTX PRO 6000 Blackwell, per-case fresh process).

## main: 21 cases, whole job 21 min (no-probe ablation: 14 min)

| model | cases | K x cut | load | generation | call |
|---|---|---|---|---|---|
| Qwen3.5-4B | 2 | 4 x 3072 | 3.7 [3.6, 3.7] | 53.1 [52.9, 53.3] | 58.5 [58.2, 58.8] |
| Skywork-OR1-Math-7B | 3 | 4 x 3072 | 3.7 [3.6, 3.7] | 42.9 [42.9, 43.1] | 48.4 [47.3, 48.6] |
| Olmo-3-7B-Think | 2 | 4 x 2048 | 3.7 [3.7, 3.7] | 38.4 [38.3, 38.5] | 43.8 [43.6, 43.9] |
| DeepSeek-R1-0528-Qwen3-8B | 3 | 4 x 2560 | 3.8 [3.8, 3.9] | 43.4 [43.2, 43.5] | 49.0 [48.9, 49.3] |
| gpt-oss-120b (high) | 3 | 4 x 1536 | 6.6 [6.6, 6.6] | 57.6 [57.2, 63.5] | 67.4 [66.5, 75.4] |
| gpt-oss-120b (low) | 5 | 4 x 1536 | 6.6 [6.6, 6.7] | 57.4 [52.7, 58.0] | 68.0 [60.0, 68.9] |
| gpt-oss-120b (medium) | 3 | 4 x 1536 | 6.6 [6.6, 6.6] | 57.3 [57.0, 60.4] | 68.2 [66.8, 71.4] |
| all | 21 | | | | 60.0 [43.6, 75.4]; mean 59.2 |
| prior-only case (no-probe ablation, model not loaded) | 8 | | | | 0.0 [0.0, 0.0] |

## small: 14 cases, whole job 12 min (no-probe ablation: 2 min)

| model | cases | K x cut | load | generation | call |
|---|---|---|---|---|---|
| Qwen3.5-4B | 5 | 4 x 3072 | 3.6 [3.5, 3.7] | 52.1 [51.9, 52.6] | 57.8 [57.2, 58.0] |
| Skywork-OR1-Math-7B | 3 | 4 x 3072 | 3.7 [3.6, 3.8] | 42.8 [42.8, 43.0] | 48.4 [47.1, 48.5] |
| Olmo-3-7B-Think | 3 | 4 x 2048 | 3.7 [3.7, 3.7] | 38.8 [38.5, 38.8] | 44.2 [43.8, 44.3] |
| DeepSeek-R1-0528-Qwen3-8B | 3 | 4 x 2560 | 3.9 [3.8, 3.9] | 43.3 [43.2, 43.3] | 49.0 [48.8, 49.1] |
| all | 14 | | | | 48.9 [43.8, 58.0]; mean 50.8 |
| prior-only case (no-probe ablation, model not loaded) | 11 | | | | 0.0 [0.0, 0.0] |

## pooled (Main + Small; report Table A1)

| model | cases | load | generation | call |
|---|---|---|---|---|
| Qwen3.5-4B | 7 | 3.6 [3.5, 3.7] | 52.2 [51.9, 53.3] | 57.9 [57.2, 58.8] |
| Skywork-OR1-Math-7B | 6 | 3.7 [3.6, 3.8] | 42.9 [42.8, 43.1] | 48.4 [47.1, 48.6] |
| Olmo-3-7B-Think | 5 | 3.7 [3.7, 3.7] | 38.5 [38.3, 38.8] | 43.9 [43.6, 44.3] |
| DeepSeek-R1-0528-Qwen3-8B | 6 | 3.8 [3.8, 3.9] | 43.3 [43.2, 43.5] | 49.0 [48.8, 49.3] |
| gpt-oss-120b | 11 | 6.6 [6.6, 6.7] | 57.4 [52.7, 63.5] | 68.0 [60.0, 75.4] |
