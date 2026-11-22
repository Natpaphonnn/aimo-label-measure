# Precision parity: bf16 (Mac, as the evaluation GPUs) vs fp16 (Kaggle T4) states (analysis/precision_parity.py)

24 Qwen3.5-4B samples recomputed on both. Cosine between the two copies per block (cut 3072):

| block | min cosine | median | max abs relative L2 difference |
|---|---|---|---|
| S:trace:0 | 0.999998 | 0.999999 | 0.0024 |
| S:trace:1 | 0.999991 | 0.999997 | 0.0043 |
| S:trace:2 | 0.999975 | 0.999994 | 0.0072 |
| S:answer:0 | 0.999932 | 0.999978 | 0.0122 |
| S:answer:1 | 0.999530 | 0.999966 | 0.0310 |
| S:answer:2 | 0.996890 | 0.999963 | 0.0788 |

Probe score of the same sample from its bf16 and its fp16 states (probe: PCA 32, C 0.1, trained on the Qwen gold samples of other problems):

- S:trace:0: max |p_bf16 - p_fp16| = 0.0021, mean 0.0006
- S:trace:1: max |p_bf16 - p_fp16| = 0.0059, mean 0.0008
- S:trace:2: max |p_bf16 - p_fp16| = 0.0031, mean 0.0008

Cross-fitted probe AUC (inner model selection, as in the paper) on Qwen samples by the precision of their stored states:

- bf16: 962 gold samples, AUC 0.895
- fp16: 86 gold samples, AUC 0.900
