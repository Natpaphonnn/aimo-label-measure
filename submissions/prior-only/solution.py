"""Prior-only baseline for the AIMO Interpretability Challenge.

Predicts robustness from a per-(model_id, reasoning_effort) class prior estimated on the
public train set aimo-interp/train-main-v2 (revision 23791d9b, labelled rows only).
No model is loaded. This is the floor every learned method must beat, and it doubles as a
pipeline check: it exercises the exact Codabench entry-point contract.

Interface (verified against components/ingestion_program/ingestion.py at de79405):
    are_robust(model_id, reasoning_effort=..., problems=[...]) -> list[bool]
`problems` are the ORIGINAL problem statements; no perturbation type is given.
"""

from __future__ import annotations

import sys
import time

# P(robust | model_id, reasoning_effort) on labelled train-main-v2 rows (robust / labelled).
PRIORS: dict[tuple[str, str], tuple[float, int]] = {
    ("Qwen/Qwen3.5-4B", "default"): (8 / 12, 12),
    ("Skywork/Skywork-OR1-Math-7B", "default"): (2 / 3, 3),
    ("allenai/Olmo-3-7B-Think", "default"): (7 / 9, 9),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"): (7 / 10, 10),
    ("openai/gpt-oss-120b", "low"): (5 / 7, 7),
    ("openai/gpt-oss-120b", "medium"): (10 / 15, 15),
    ("openai/gpt-oss-120b", "high"): (13 / 15, 15),
}
POOLED_PRIOR = 52 / 71  # all labelled rows
THRESHOLD = 0.5


def prior_for(model_id: str, reasoning_effort: str) -> float:
    key = (str(model_id), str(reasoning_effort or "default"))
    if key in PRIORS:
        return PRIORS[key][0]
    # Same model, unknown effort: pool that model's configurations.
    rows = [(p, n) for (m, _), (p, n) in PRIORS.items() if m == key[0]]
    if rows:
        return sum(p * n for p, n in rows) / sum(n for _, n in rows)
    return POOLED_PRIOR


def are_robust(model_id: str, reasoning_effort: str = "default", problems: list[str] | None = None) -> list[bool]:
    t0 = time.time()
    problems = list(problems or [])
    p = prior_for(model_id, reasoning_effort)
    verdict = bool(p >= THRESHOLD)
    out = [verdict for _ in problems]
    print(
        f"[prior-only] model={model_id} effort={reasoning_effort} n={len(problems)} "
        f"prior={p:.3f} verdict={verdict} elapsed={time.time() - t0:.3f}s",
        file=sys.stderr,
        flush=True,
    )
    return out
