# Stage 1 on finished vs truncated samples, probes fitted separately (analysis/stage1_finished_truncated.py)

| model | subset | gold samples | correct share | problems | AUC [95% CI] |
|---|---|---|---|---|---|
| qwen | finished | 9 | 1.00 | 5 | one class only |
| qwen | truncated | 1023 | 0.38 | 133 | 0.879 [0.837, 0.916] |
| skywork | finished | 145 | 1.00 | 30 | one class only |
| skywork | truncated | 919 | 0.29 | 123 | 0.798 [0.726, 0.855] |
| deepseek | finished | 34 | 1.00 | 10 | one class only |
| deepseek | truncated | 902 | 0.16 | 130 | 0.774 [0.706, 0.836] |
