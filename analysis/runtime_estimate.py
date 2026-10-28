#!/usr/bin/env python3
"""Runtime estimate for K budgeted samples per (config, problem) on the evaluation worker (RTX PRO 6000, HF generate).
Throughput: aggregate tok/s measured on that worker by the BAS submissions (CVasilopoulos/aimo-bas report Table 4, 18-36 concurrent
sequences, ~6k-token traces). Conservative: assumes every sample runs to the cap except gpt-oss (mean 1.5k tokens at low effort,
scaled x2 / x4 for medium / high). Overheads: model load (2-4 s) + forced-answer pass (~3% of generation) + 15% shared-GPU margin."""
import itertools
TPS = {"Qwen/Qwen3.5-4B": 552, "Skywork/Skywork-OR1-Math-7B": 309, "allenai/Olmo-3-7B-Think": 206, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 216, "openai/gpt-oss-120b": 226}
GPT_TOKENS = {"low": 1500, "medium": 3000, "high": 6000}
SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
def seconds(model, n, K, cap, effort=None):
    toks = n * K * (min(cap, GPT_TOKENS[effort]) if effort else cap)
    return (toks / TPS[model]) * 1.03 * 1.15 + 4
print("| problems per config | K | token cap | Small track total s | Main track total s (adds gpt-oss low/med/high) |")
print("|---|---|---|---|---|")
for n, K, cap in itertools.product([5, 15, 30], [2, 4], [2048, 4096, 8192]):
    small = sum(seconds(m, n, K, cap) for m in SMALL)
    main = small + sum(seconds("openai/gpt-oss-120b", n, K, cap, e) for e in GPT_TOKENS)
    flag = lambda s: f"{s:.0f}" + (" **over**" if s > 3600 - 420 else "")
    print(f"| {n} | {K} | {cap} | {flag(small)} | {flag(main)} |")

# --- Per-problem protocol (organizers, 2026-10-16): one call per case, ~150 s usable per case (200 s/case on val Main) ---
# Decode steps/s on the worker from the BAS logs (report Table 4: sequences x tokens per trace / generation seconds, batch 18-36).
STEPS = {"Qwen/Qwen3.5-4B": 18 * 6144 / 200, "Skywork/Skywork-OR1-Math-7B": 36 * 5384 / 614, "allenai/Olmo-3-7B-Think": 36 * 5516 / 997,
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 24 * 6144 / 703, "openai/gpt-oss-120b": 30 * 1489 / 198}
STEPS = {m: v / n for (m, v), n in zip(STEPS.items(), [18, 36, 36, 24, 30])}
print("\n| model | steps/s at batch 18-36 (BAS) | tokens per sample in 150 s (load 4 s, prefill + forced answer 10 s) | in 120 s |")
print("|---|---|---|---|")
for m, r in STEPS.items():
    print(f"| {m} | {r:.1f} | {r * (150 - 14):.0f} | {r * (120 - 14):.0f} |")

# --- v3: measured on the worker by timing canary 991764 (batch 8, 192 new tokens, short prompt, fresh process per case) ---
CANARY = {"Qwen/Qwen3.5-4B": (50.6, 3.5), "Skywork/Skywork-OR1-Math-7B": (70.9, 3.6), "allenai/Olmo-3-7B-Think": (58.4, 3.6),
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": (60.4, 3.8), "openai/gpt-oss-120b": (24.6, 6.5)}
print("\n| model | steps/s (canary, short context) | load s | tokens per sample in 140 s of generation | if long context halves the rate |")
print("|---|---|---|---|---|")
for m, (r, ld) in CANARY.items():
    print(f"| {m} | {r} | {ld} | {r * 140:.0f} | {r * 70:.0f} |")
